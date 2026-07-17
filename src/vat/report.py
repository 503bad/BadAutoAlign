"""JSONレポートとスペクトログラム比較画像の出力。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def write_report(report: dict, path: str, input_wav: str,
                 output_wav: str | None) -> None:
    p = Path(path)
    if output_wav:
        imgs = _write_spectrograms(input_wav, output_wav, p)
        if imgs:
            report["spectrograms"] = imgs
    p.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                 encoding="utf-8")
    print(f"レポート: {p}")


def _write_spectrograms(input_wav: str, output_wav: str,
                        report_path: Path) -> list[str] | None:
    """処理前後のスペクトログラム比較PNG。matplotlib未導入なら省略。"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[info] matplotlib未導入のためスペクトログラム画像は省略します")
        return None

    from .audio import load_wav

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    for ax, (title, path) in zip(
        axes, [("before", input_wav), ("after", output_wav)]
    ):
        audio, sr = load_wav(path)
        _spec(ax, audio, sr)
        ax.set_title(f"{title}: {Path(path).name}")
        ax.set_ylabel("Hz")
    axes[-1].set_xlabel("s")
    out = report_path.with_suffix(".spectrogram.png")
    fig.tight_layout()
    fig.savefig(out, dpi=100)
    plt.close(fig)
    return [str(out)]


def _spec(ax, audio: np.ndarray, sr: int, n_fft: int = 2048, hop: int = 512) -> None:
    n_frames = max(1, (len(audio) - n_fft) // hop + 1)
    win = np.hanning(n_fft)
    frames = np.stack([audio[i * hop: i * hop + n_fft] * win for i in range(n_frames)])
    mag = np.abs(np.fft.rfft(frames, axis=1)).T
    db = 20 * np.log10(np.maximum(mag, 1e-6))
    extent = (0, n_frames * hop / sr, 0, sr / 2)
    ax.imshow(db, origin="lower", aspect="auto", extent=extent,
              vmin=db.max() - 90, vmax=db.max(), cmap="magma")
    ax.set_ylim(0, 8000)
