"""P-center（知覚的な頭＝「芯」）の推定。

有声化の開始点（しゃくり・ガナリの起点）ではなく、リズム知覚上の基準点を
推定する。文献に基づく2つの手掛かりを併用し、遅い方を芯とする:

1. ピッチ到達点 — F0がその音節の安定音高±50セント圏に入り持続し始めた点
   （歌唱転写研究の慣行。しゃくりを構造的に除外）
2. 中帯域(300-3000Hz)エネルギー包絡の最速上昇点
   （Scott 1998 / Rathcke+ 2024 のP-centerモデル。母音開始の主要相関量）

ガイド・ボーカル両側に同じ検出器を適用することで、検出器自体のバイアスを
相殺する（片側だけ有声化開始点を使うと系統的な後ろズレが生じる）。
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import butter, sosfiltfilt

from .audio import frame_rms
from .config import Config

SEARCH_AHEAD_S = 0.25    # 芯を探す最大範囲（音節頭から）
SEARCH_BACK_S = 0.04     # エネルギー上昇は音節頭の少し手前から探す
PITCH_TOL_SEMI = 0.5     # ピッチ到達の許容（±50セント）
MIN_RISE_DB = 6.0        # これ未満の起伏しかない窓ではエネルギー手掛かりを使わない


def midband_envelope(audio: np.ndarray, sr: int, cfg: Config) -> np.ndarray:
    """300-3000Hz帯域のフレームRMS包絡（dB、σ≈15ms平滑）。"""
    sos = butter(4, [300.0, min(3000.0, sr / 2 * 0.9)], btype="bandpass",
                 fs=sr, output="sos")
    band = sosfiltfilt(sos, audio.astype(np.float64))
    env = frame_rms(band.astype(np.float32), cfg.hop, cfg.frame_length)
    env_db = 20.0 * np.log10(np.maximum(env, 1e-8))
    sigma = max(1e-3, 0.015 * sr / cfg.hop)
    return gaussian_filter1d(env_db, sigma=sigma, mode="nearest")


def energy_rise_time(env_db: np.ndarray, dt: float,
                     t_start: float, t_end: float) -> float | None:
    """窓内でエネルギー包絡が最も速く立ち上がる時刻。起伏が小さい窓ではNone。"""
    i0 = max(0, int((t_start - SEARCH_BACK_S) / dt))
    i1 = min(len(env_db), int(min(t_end, t_start + SEARCH_AHEAD_S) / dt) + 1)
    if i1 - i0 < 3:
        return None
    win = env_db[i0:i1]
    if win.max() - win.min() < MIN_RISE_DB:
        return None  # レガート内など、明確な立ち上がりが無い
    slope = np.gradient(win)
    k = int(np.argmax(slope))
    if slope[k] <= 0:
        return None
    return (i0 + k) * dt


def pitch_arrival_time(times: np.ndarray, semis: np.ndarray, voiced: np.ndarray,
                       t_start: float, t_end: float, dt: float) -> float | None:
    """F0が音節の安定音高±50セント圏に入り、30ms以上持続し始めた最初の時刻。"""
    i0 = max(0, int(t_start / dt))
    i1 = min(len(semis), int(min(t_end, t_start + SEARCH_AHEAD_S + 0.15) / dt) + 1)
    if i1 - i0 < 3:
        return None
    # 安定音高 = 音節後半60%の有声フレーム中央値
    j0 = i0 + int((i1 - i0) * 0.4)
    ref = semis[j0:i1][voiced[j0:i1]]
    if len(ref) == 0:
        ref = semis[i0:i1][voiced[i0:i1]]
        if len(ref) == 0:
            return None
    target = float(np.median(ref))
    hold = max(2, int(0.03 / dt))
    ok = voiced[i0:i1] & (np.abs(semis[i0:i1] - target) <= PITCH_TOL_SEMI)
    run = 0
    for k, flag in enumerate(ok):
        run = run + 1 if flag else 0
        if run >= hold:
            return (i0 + k - hold + 1) * dt
    return None


def core_time(env_db: np.ndarray, semis: np.ndarray, voiced: np.ndarray,
              t_start: float, t_end: float, dt: float) -> float:
    """音節/ノートの芯。手掛かりの遅い方を採用し、探索範囲にクランプする。"""
    e = energy_rise_time(env_db, dt, t_start, t_end)
    p = pitch_arrival_time(np.arange(len(semis)) * dt, semis, voiced,
                           t_start, t_end, dt)
    cands = [c for c in (e, p) if c is not None]
    if not cands:
        return t_start
    core = max(cands)
    return float(np.clip(core, t_start, t_start + SEARCH_AHEAD_S))
