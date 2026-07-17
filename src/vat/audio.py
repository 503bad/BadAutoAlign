"""WAV入出力とフレーム単位の基本特徴量（RMS・ZCR）."""

from __future__ import annotations

import numpy as np
import soundfile as sf


def load_wav(path: str) -> tuple[np.ndarray, int]:
    """WAVをfloat32モノラルで読み込む。ステレオは警告なしでダウンミックス
    （プロトタイプ仕様。将来はチャンネル別処理）。"""
    audio, sr = sf.read(path, dtype="float32", always_2d=True)
    mono = audio.mean(axis=1) if audio.shape[1] > 1 else audio[:, 0]
    return np.ascontiguousarray(mono, dtype=np.float32), int(sr)


def save_wav(path: str, audio: np.ndarray, sr: int) -> None:
    sf.write(path, np.asarray(audio, dtype=np.float32), sr, subtype="FLOAT")


def frame_rms(audio: np.ndarray, hop: int, frame_length: int) -> np.ndarray:
    """中心揃えのフレームRMS。フレーム数は 1 + len//hop（検出器のグリッドと一致させる）。"""
    n_frames = 1 + len(audio) // hop
    pad = frame_length // 2
    x = np.pad(audio.astype(np.float64), (pad, pad + frame_length))
    out = np.empty(n_frames)
    for i in range(n_frames):
        seg = x[i * hop : i * hop + frame_length]
        out[i] = np.sqrt(np.mean(seg * seg) + 1e-12)
    return out


def frame_zcr(audio: np.ndarray, hop: int, frame_length: int) -> np.ndarray:
    """ゼロ交差率（0〜1）。無声摩擦音の検出補助（P2）。"""
    n_frames = 1 + len(audio) // hop
    pad = frame_length // 2
    x = np.pad(audio, (pad, pad + frame_length))
    signs = np.signbit(x)
    flips = (signs[1:] != signs[:-1]).astype(np.float64)
    out = np.empty(n_frames)
    for i in range(n_frames):
        out[i] = flips[i * hop : i * hop + frame_length - 1].mean()
    return out


def db_to_lin(db: float) -> float:
    return 10.0 ** (db / 20.0)


def lin_to_db(x: np.ndarray | float) -> np.ndarray | float:
    return 20.0 * np.log10(np.maximum(x, 1e-12))
