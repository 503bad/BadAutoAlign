"""ボーカルの音節（モーラ）単位セグメンテーション。

日本語歌唱を前提に、1音節 ≒ 1ガイドノートの構造を使った離散アライメントの
ための単位を切り出す。音節境界の手掛かり:
  1. 有声化開始点（無声子音・息継ぎの後の母音頭）
  2. 母音品質の変化点（CMVN-MFCCのノベルティピーク。レガートの母音遷移）
  3. ピッチの跳躍（同一母音で音高だけ変わる場合）
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import Config
from .detect import PitchTrack
from .features import phonetic_features


@dataclass
class Syllable:
    onset: float                 # 音節頭（母音開始）[s]（セグメント先頭基準）
    end: float
    semitone: float              # 音高中央値（MIDIスケール、nan可）
    feat: np.ndarray = field(repr=False, default=None)  # 母音特徴の重心

    @property
    def duration(self) -> float:
        return self.end - self.onset


def detect_syllables(audio: np.ndarray, sr: int, cfg: Config,
                     track: PitchTrack) -> list[Syllable]:
    from scipy.signal import find_peaks

    n = track.n_frames
    dt = float(track.times[1] - track.times[0]) if n > 1 else cfg.hop / sr
    feats = phonetic_features(audio, sr, cfg, active=track.voiced)
    n = min(n, len(feats))
    semis = track.semitones()[:n]
    voiced = track.voiced[:n]

    min_syl = max(2, int(0.05 / dt))        # 音節の最短 50ms
    runs = _voiced_runs(voiced, bridge=max(1, int(0.03 / dt)))

    boundaries: list[int] = []
    for s, e in runs:
        boundaries.append(s)
        if e - s < 2 * min_syl:
            continue
        # 母音品質ノベルティ（前後60msの平均特徴の距離）
        w = max(2, int(0.06 / dt))
        nov = np.zeros(e - s)
        for i in range(s + w, e - w):
            a = feats[i - w:i].mean(axis=0)
            b = feats[i:i + w].mean(axis=0)
            nov[i - s] = np.linalg.norm(b - a) / np.sqrt(feats.shape[1])
        peaks, _ = find_peaks(nov, distance=max(1, int(0.09 / dt)),
                              height=0.5, prominence=0.3)
        boundaries.extend(int(s + p) for p in peaks)
        # ピッチ跳躍（>0.8半音、前後30ms中央値）
        pw = max(2, int(0.03 / dt))
        for i in range(s + min_syl, e - min_syl):
            left = semis[max(s, i - pw):i]
            right = semis[i:i + pw]
            if not (np.isfinite(left).any() and np.isfinite(right).any()):
                continue
            a = np.nanmedian(left)
            b = np.nanmedian(right)
            if abs(b - a) > 0.8:
                boundaries.append(i)

    boundaries = sorted(set(boundaries))
    # 近すぎる境界を統合
    merged: list[int] = []
    for b in boundaries:
        if not merged or b - merged[-1] >= min_syl:
            merged.append(b)

    syllables: list[Syllable] = []
    for k, b in enumerate(merged):
        # この境界が属する有声ランの終端まで、または次の境界まで
        e_run = next((e for s, e in runs if s <= b < e), None)
        if e_run is None:
            continue
        nxt = merged[k + 1] if k + 1 < len(merged) and merged[k + 1] < e_run else e_run
        if nxt - b < min_syl:
            continue
        seg_semis = semis[b:nxt]
        med = float(np.nanmedian(seg_semis)) if np.isfinite(seg_semis).any() else float("nan")
        centroid = feats[b:nxt].mean(axis=0)
        syllables.append(Syllable(onset=b * dt, end=nxt * dt,
                                  semitone=med, feat=centroid))

    return _merge_fragments(syllables, audio, sr, cfg, dt)


def _merge_fragments(syllables: list[Syllable], audio: np.ndarray, sr: int,
                     cfg: Config, dt: float) -> list[Syllable]:
    """過分割対策: 連続していて同一音高・境界にエネルギー谷が無い断片は
    同一音節とみなしてマージする（ビブラート等によるノベルティ誤検出の吸収）。

    同一音高で歌い直す連続モーラ（「ああ」等）は再アーティキュレーションの
    エネルギー谷が境界に現れるため、谷がある場合は分割を保持する。
    """
    from .audio import frame_rms, lin_to_db

    if len(syllables) < 2:
        return syllables
    rms_db = lin_to_db(frame_rms(audio, cfg.hop, cfg.frame_length))

    def has_energy_dip(t: float, left: Syllable, right: Syllable) -> bool:
        j = int(round(t / dt))
        w = max(1, int(0.02 / dt))
        lo = max(0, j - w)
        hi = min(len(rms_db), j + w + 1)
        if hi <= lo:
            return False
        junction = float(np.min(rms_db[lo:hi]))
        li = slice(int(round(left.onset / dt)), max(int(round(left.end / dt)), lo))
        ri = slice(min(int(round(right.onset / dt)) + 1, hi), int(round(right.end / dt)))
        cores = []
        for sl in (li, ri):
            if sl.stop > sl.start:
                cores.append(float(np.median(rms_db[sl])))
        return bool(cores) and junction < min(cores) - 3.0

    out = [syllables[0]]
    for syl in syllables[1:]:
        prev = out[-1]
        contiguous = syl.onset - prev.end < 0.025
        same_pitch = (np.isfinite(prev.semitone) and np.isfinite(syl.semitone)
                      and abs(prev.semitone - syl.semitone) < 0.6)
        if contiguous and same_pitch and not has_energy_dip(prev.end, prev, syl):
            dur_a, dur_b = prev.duration, syl.duration
            total = dur_a + dur_b
            feat = (prev.feat * dur_a + syl.feat * dur_b) / total
            semi = (prev.semitone * dur_a + syl.semitone * dur_b) / total
            out[-1] = Syllable(onset=prev.onset, end=syl.end,
                               semitone=semi, feat=feat)
        else:
            out.append(syl)
    return out


def _voiced_runs(voiced: np.ndarray, bridge: int) -> list[tuple[int, int]]:
    """有声区間。bridge フレーム以下の短い無声（促音・軽い子音）は跨いで結合。"""
    runs: list[tuple[int, int]] = []
    start = None
    gap = 0
    for i, v in enumerate(voiced):
        if v:
            if start is None:
                start = i
            gap = 0
        elif start is not None:
            gap += 1
            if gap > bridge:
                runs.append((start, i - gap + 1))
                start, gap = None, 0
    if start is not None:
        runs.append((start, len(voiced) - gap))
    return [(s, e) for s, e in runs if e > s]
