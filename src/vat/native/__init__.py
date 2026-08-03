"""Signalsmith Stretch 自作ラッパーのビルドとctypesバインディング。

初回利用時に wrapper.cpp をシステムのC++コンパイラでビルドし、
ユーザーキャッシュに共有ライブラリを保存する（ソース変更で自動再ビルド）。
コンパイラが無い環境では RuntimeError → `--engine world` を案内する。
"""

from __future__ import annotations

import ctypes
import hashlib
import os
import shutil
import subprocess
import sys
import threading
from pathlib import Path

import numpy as np

_HERE = Path(__file__).parent
_LIB: ctypes.CDLL | None = None
_BUILD_LOCK = threading.Lock()  # サービスのウォームアップと本処理の同時ビルド防止


def _source_hash() -> str:
    h = hashlib.sha256()
    for p in sorted(_HERE.rglob("*.cpp")) + sorted(_HERE.rglob("*.h")):
        h.update(p.read_bytes())
    return h.hexdigest()[:16]


def _cache_dir() -> Path:
    import platformdirs

    d = Path(platformdirs.user_cache_dir("vocal-align-tune"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def _lib_suffix() -> str:
    if sys.platform == "darwin":
        return ".dylib"
    if sys.platform == "win32":
        return ".dll"
    return ".so"


_MSVC_CL = ["cl", "/nologo", "/utf-8", "/std:c++17", "/O2", "/EHsc", "/LD"]


def _msvc_vcvars_command(src: str, out: str) -> str | None:
    """PATHに無いMSVCをvswhereで探し、vcvars64経由でclを呼ぶコマンド文字列を返す。

    リストではなく文字列で返す（subprocessのリスト→コマンドライン変換は
    cmd.exe の引用符規則と非互換のため）。
    """
    import os

    vswhere = (Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"))
               / "Microsoft Visual Studio" / "Installer" / "vswhere.exe")
    if not vswhere.exists():
        return None
    r = subprocess.run(
        [str(vswhere), "-products", "*", "-sort",
         "-requires", "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
         "-property", "installationPath"],
        capture_output=True, text=True)
    for line in r.stdout.splitlines():
        vcvars = Path(line.strip()) / "VC" / "Auxiliary" / "Build" / "vcvars64.bat"
        if vcvars.exists():
            cl = " ".join(_MSVC_CL)
            return (f'cmd /s /c ""{vcvars}" >nul 2>&1 && '
                    f'{cl} "{src}" /Fe:"{out}""')
    return None


def _compile_commands(src: str, out: str) -> list[list[str] | str]:
    """利用可能なコンパイラごとのビルドコマンド候補（優先順）。"""
    if sys.platform == "win32":
        gnu = ["-std=c++17", "-O3", "-shared", src, "-o", out,
               "-static-libgcc", "-static-libstdc++"]
        cmds = [
            ["g++", *gnu],
            ["clang++", *gnu],
            # cl はVS開発者プロンプト等、環境変数設定済みの場合のみPATHに居る
            [*_MSVC_CL, src, f"/Fe:{out}"],
        ]
        msvc = _msvc_vcvars_command(src, out)
        if msvc:
            cmds.append(msvc)
        return cmds
    gnu = ["-std=c++17", "-O3", "-shared", "-fPIC", src, "-o", out]
    return [["c++", *gnu], ["g++", *gnu], ["clang++", *gnu]]


def _bundled_lib() -> Path | None:
    """プリビルトDLLがあればそれを使う（配布パッケージ用。ビルド不要）。

    優先順: 環境変数 VAT_STRETCH_LIB → PyInstaller同梱（実行ファイルと同じ場所）
    """
    env = os.environ.get("VAT_STRETCH_LIB")
    if env and Path(env).exists():
        return Path(env)
    if getattr(sys, "frozen", False):
        p = Path(sys.executable).parent / f"vatstretch{_lib_suffix()}"
        if p.exists():
            return p
    return None


def _resolve_lib() -> Path:
    bundled = _bundled_lib()
    if bundled is not None:
        return bundled
    return _build()


def _build() -> Path:
    with _BUILD_LOCK:
        return _build_locked()


def _build_locked() -> Path:
    lib_path = _cache_dir() / f"vatstretch-{_source_hash()}{_lib_suffix()}"
    if lib_path.exists():
        return lib_path
    print("Signalsmith Stretchラッパーをビルド中…（初回のみ、数十秒かかることがあります）",
          file=sys.stderr, flush=True)
    src = str(_HERE / "wrapper.cpp")
    candidates = [c for c in _compile_commands(src, str(lib_path))
                  if isinstance(c, str) or shutil.which(c[0])]
    if not candidates:
        hint = ("g++ (MSYS2/w64devkit) か Visual Studio Build Tools を導入するか、"
                if sys.platform == "win32" else "")
        raise RuntimeError(
            f"C++コンパイラが見つかりません。{hint}--engine world を使用してください"
        )
    errors = []
    for cmd in candidates:
        label = cmd[0] if isinstance(cmd, list) else "msvc"
        try:
            # cl は .obj 等の中間ファイルをCWDに出すためキャッシュディレクトリで実行
            subprocess.run(cmd, check=True, capture_output=True, text=True,
                           encoding="utf-8", errors="replace",
                           cwd=str(_cache_dir()))
            return lib_path
        except subprocess.CalledProcessError as e:
            # cl はエラーをstdoutに出す
            errors.append(f"[{label}] {e.stderr or e.stdout}")
    raise RuntimeError(
        "Signalsmith Stretchラッパーのビルドに失敗しました:\n" + "\n".join(errors)
    )


def is_available() -> bool:
    """stretchエンジンが利用可能か（同梱DLL or ビルド済み or その場でビルド成功）。"""
    try:
        _resolve_lib()
        return True
    except RuntimeError:
        return False


def _load() -> ctypes.CDLL:
    global _LIB
    if _LIB is not None:
        return _LIB
    lib = ctypes.CDLL(str(_resolve_lib()))
    lib.vs_create.restype = ctypes.c_void_p
    lib.vs_create.argtypes = [ctypes.c_float]
    lib.vs_destroy.argtypes = [ctypes.c_void_p]
    lib.vs_reset.argtypes = [ctypes.c_void_p]
    lib.vs_input_latency.restype = ctypes.c_int
    lib.vs_input_latency.argtypes = [ctypes.c_void_p]
    lib.vs_output_latency.restype = ctypes.c_int
    lib.vs_output_latency.argtypes = [ctypes.c_void_p]
    lib.vs_set_transpose_factor.argtypes = [ctypes.c_void_p, ctypes.c_float, ctypes.c_float]
    lib.vs_set_formant_factor.argtypes = [ctypes.c_void_p, ctypes.c_float, ctypes.c_int]
    fptr = ctypes.POINTER(ctypes.c_float)
    lib.vs_process.argtypes = [ctypes.c_void_p, fptr, ctypes.c_int, fptr, ctypes.c_int]
    lib.vs_flush.argtypes = [ctypes.c_void_p, fptr, ctypes.c_int]
    _LIB = lib
    return lib


class StretchStream:
    """モノラルのストリーミングStretch。process()を繰り返し呼び、最後にflush()。"""

    def __init__(self, sample_rate: float):
        self._lib = _load()
        self._h = self._lib.vs_create(ctypes.c_float(sample_rate))

    def __del__(self):
        if getattr(self, "_h", None):
            self._lib.vs_destroy(self._h)
            self._h = None

    @property
    def input_latency(self) -> int:
        return self._lib.vs_input_latency(self._h)

    @property
    def output_latency(self) -> int:
        return self._lib.vs_output_latency(self._h)

    def set_transpose_factor(self, multiplier: float, tonality_limit: float = 0.0) -> None:
        self._lib.vs_set_transpose_factor(
            self._h, ctypes.c_float(multiplier), ctypes.c_float(tonality_limit))

    def process(self, chunk: np.ndarray, out_samples: int) -> np.ndarray:
        x = np.ascontiguousarray(chunk, dtype=np.float32)
        out = np.zeros(out_samples, dtype=np.float32)
        fptr = ctypes.POINTER(ctypes.c_float)
        self._lib.vs_process(
            self._h,
            x.ctypes.data_as(fptr), len(x),
            out.ctypes.data_as(fptr), out_samples,
        )
        return out

    def flush(self, out_samples: int) -> np.ndarray:
        out = np.zeros(out_samples, dtype=np.float32)
        self._lib.vs_flush(
            self._h, out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), out_samples)
        return out
