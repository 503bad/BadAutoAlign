"""シフト/ストレッチエンジン（P1）。

- SignalsmithEngine: 周波数領域＋位相コヒーレンス＋フォルマント処理（MIT）。
  ストリーミングAPIでチャンクごとにレート/シフト量を変えられるため、
  可変レートワープと時間変化ピッチシフトの両方をこれ一つで行う。
- WorldEngine: F0/スペクトル包絡/非周期性に分解して再合成（修正BSD）。
  A/B比較用フォールバック。

librosa.effects.pitch_shift と単純リサンプリングは使用禁止（仕様）。
"""

from __future__ import annotations

import numpy as np

from .config import Config
from .detect import PitchTrack
from .timing import WarpMap


def get_engine(cfg: Config) -> "BaseEngine":
    if cfg.engine == "stretch":
        from . import native

        if not native.is_available():
            cfg.warn(
                "C++コンパイラが無くstretchエンジンをビルドできないため"
                "worldエンジンにフォールバックします")
            return WorldEngine(cfg)
        return SignalsmithEngine(cfg)
    if cfg.engine == "world":
        return WorldEngine(cfg)
    raise ValueError(f"未知のエンジン: {cfg.engine}")


class BaseEngine:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def pitch_shift(self, audio: np.ndarray, sr: int, track: PitchTrack,
                    curve_semitones: np.ndarray) -> np.ndarray:
        """フレームごとの補正量カーブ（半音）でピッチシフト。長さは入力と同一。"""
        raise NotImplementedError

    def time_warp(self, audio: np.ndarray, sr: int, warp: WarpMap) -> np.ndarray:
        """区分線形ワープマップで可変レートストレッチ。長さは入力と同一
        （マップ両端が固定されている前提）。"""
        raise NotImplementedError


# ---------------------------------------------------------------- Signalsmith

class SignalsmithEngine(BaseEngine):
    """自作ラッパー（vat.native）によるストリーミング処理。

    SignalsmithStretch本体のストリーミングAPIは process(in, n_in, out, n_out) の
    サイズ比が伸縮率を決める。チャンクごとにトランスポーズ量を更新できるため、
    時間変化するピッチシフトと可変レートワープの両方をこれで行う。
    """

    def pitch_shift(self, audio, sr, track, curve_semitones):
        from .native import StretchStream

        hop = self.cfg.hop
        s = StretchStream(float(sr))
        latency = s.input_latency + s.output_latency
        x = audio.astype(np.float32)
        n = len(audio)
        chunks = []
        frame = 0
        for pos in range(0, n, hop):
            chunk = x[pos: pos + hop]
            semi = float(curve_semitones[min(frame, len(curve_semitones) - 1)])
            s.set_transpose_factor(2.0 ** (semi / 12.0))
            chunks.append(s.process(chunk, len(chunk)))
            frame += 1
        chunks.append(s.process(np.zeros(s.input_latency, dtype=np.float32),
                                s.input_latency))
        chunks.append(s.flush(s.output_latency))
        out = np.concatenate(chunks)
        return _fix_length(out[latency:], n)

    def time_warp(self, audio, sr, warp):
        from .native import StretchStream

        s = StretchStream(float(sr))
        latency = s.input_latency + s.output_latency
        x = audio.astype(np.float32)
        n = len(audio)
        src_pts = (warp.src * sr).round().astype(int)
        dst_pts = (warp.dst * sr).round().astype(int)
        src_pts[-1], dst_pts[-1] = n, n  # 両端固定（丸め誤差の吸収）
        chunks = []
        max_chunk = 2048
        for k in range(len(src_pts) - 1):
            ls = src_pts[k + 1] - src_pts[k]
            ld = dst_pts[k + 1] - dst_pts[k]
            if ls <= 0 or ld <= 0:
                continue
            fed = emitted = 0
            while fed < ls:
                m = min(max_chunk, ls - fed)
                want = int(round((fed + m) * ld / ls)) - emitted
                chunk = x[src_pts[k] + fed: src_pts[k] + fed + m]
                chunks.append(s.process(chunk, max(want, 0)))
                fed += m
                emitted += max(want, 0)
        chunks.append(s.process(np.zeros(s.input_latency, dtype=np.float32),
                                s.input_latency))
        chunks.append(s.flush(s.output_latency))
        out = np.concatenate(chunks)
        return _fix_length(out[latency:], n)


# ---------------------------------------------------------------- WORLD

class WorldEngine(BaseEngine):
    FRAME_PERIOD_MS = 5.0

    def _analyze(self, audio: np.ndarray, sr: int):
        import pyworld as pw

        x = audio.astype(np.float64)
        f0, t = pw.harvest(x, sr, f0_floor=self.cfg.fmin,
                           f0_ceil=self.cfg.fmax * 1.2,
                           frame_period=self.FRAME_PERIOD_MS)
        f0 = pw.stonemask(x, f0, t, sr)
        sp = pw.cheaptrick(x, f0, t, sr)
        ap = pw.d4c(x, f0, t, sr)
        return f0, t, sp, ap

    def pitch_shift(self, audio, sr, track, curve_semitones):
        import pyworld as pw

        f0, t, sp, ap = self._analyze(audio, sr)
        curve = np.interp(t, track.times, curve_semitones)
        f0_new = np.where(f0 > 0, f0 * 2.0 ** (curve / 12.0), 0.0)
        y = pw.synthesize(f0_new, sp, ap, sr, self.FRAME_PERIOD_MS)
        return _fix_length(y.astype(np.float32), len(audio))

    def time_warp(self, audio, sr, warp):
        import pyworld as pw

        f0, t, sp, ap = self._analyze(audio, sr)
        src_t = warp.dst_to_src(t)  # 出力フレームごとの参照元時刻
        idx = src_t / (self.FRAME_PERIOD_MS / 1000.0)
        idx = np.clip(idx, 0, len(f0) - 1)
        i0 = np.floor(idx).astype(int)
        i1 = np.minimum(i0 + 1, len(f0) - 1)
        w = (idx - i0)[:, None]
        sp_w = np.ascontiguousarray(sp[i0] * (1 - w) + sp[i1] * w)
        ap_w = np.ascontiguousarray(ap[i0] * (1 - w) + ap[i1] * w)
        # F0は有声/無声の混在補間を避ける（最近傍で有声性を保ち、有声同士のみ対数補間）
        near = np.where(w[:, 0] < 0.5, i0, i1)
        f0_w = f0[near]
        both = (f0[i0] > 0) & (f0[i1] > 0)
        f0_w[both] = np.exp(
            np.log(f0[i0][both]) * (1 - w[both, 0]) + np.log(f0[i1][both]) * w[both, 0]
        )
        y = pw.synthesize(f0_w, sp_w, ap_w, sr, self.FRAME_PERIOD_MS)
        return _fix_length(y.astype(np.float32), len(audio))


def _fix_length(x: np.ndarray, n: int) -> np.ndarray:
    if len(x) >= n:
        return x[:n]
    return np.pad(x, (0, n - len(x)))
