"""音素的な音響特徴（話者非依存）。

ガイド（合成ボーカル）と実録ボーカルの音色差に頑健な「発音の同一性」の
手掛かりとして、CMVN正規化したMFCC＋ΔMFCCを使う。絶対的な音色では
なく母音・子音の対比が距離に反映されるため、同じ歌詞を歌った2音声の
フレーム対応付け（DTW）に適する。
"""

from __future__ import annotations

import numpy as np

from .config import Config


def phonetic_features(audio: np.ndarray, sr: int, cfg: Config,
                      active: np.ndarray | None = None) -> np.ndarray:
    """フレームごとの正規化MFCC特徴 (n_frames, 24)。

    active: 正規化統計に使うフレームのマスク（無音を統計から除くため）。
    """
    import librosa

    n_frames = 1 + len(audio) // cfg.hop
    mfcc = librosa.feature.mfcc(
        y=audio.astype(np.float32), sr=sr, n_mfcc=13,
        n_fft=cfg.frame_length, hop_length=cfg.hop,
    )[:, :n_frames]
    mfcc = mfcc[1:]  # c0（全体エネルギー）は除外
    delta = librosa.feature.delta(mfcc)
    feat = np.vstack([mfcc, delta]).T  # (n_frames, 24)

    ref = feat
    if active is not None and active.any():
        n = min(len(feat), len(active))
        if active[:n].any():
            ref = feat[:n][active[:n]]
    mu = ref.mean(axis=0)
    sd = ref.std(axis=0) + 1e-6
    return (feat - mu) / sd


def dtw_cost_matrix(
    feat_a: np.ndarray, semis_a: np.ndarray, voiced_a: np.ndarray,
    feat_b: np.ndarray, semis_b: np.ndarray, voiced_b: np.ndarray,
) -> np.ndarray:
    """フレーム間コスト (len_a, len_b)。

    - 音素特徴のユークリッド距離（主）
    - ピッチクラス差（オクターブ折返し、両者有声時のみ。副）
    - 有声/無声の不一致ペナルティ
    """
    na, nb = len(feat_a), len(feat_b)
    dim = feat_a.shape[1]
    # (na, nb) 距離: ||x||^2 + ||y||^2 - 2xy
    sq = (
        np.sum(feat_a**2, axis=1)[:, None]
        + np.sum(feat_b**2, axis=1)[None, :]
        - 2.0 * feat_a @ feat_b.T
    )
    cost = np.sqrt(np.maximum(sq, 0.0)) / np.sqrt(dim)

    pa = np.nan_to_num(semis_a, nan=0.0)
    pb = np.nan_to_num(semis_b, nan=0.0)
    diff = pa[:, None] - pb[None, :]
    fold = np.abs((diff + 6.0) % 12.0 - 6.0)  # ピッチクラス距離（オクターブ非依存）
    both = voiced_a[:, None].astype(bool) & voiced_b[None, :].astype(bool)
    cost += np.where(both, np.minimum(fold, 4.0) * 0.25, 0.0)

    cost += np.abs(voiced_a[:, None].astype(float) - voiced_b[None, :].astype(float)) * 1.5
    return cost
