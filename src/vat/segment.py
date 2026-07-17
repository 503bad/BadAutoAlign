"""T1: 無音ベースのセクション分割とフレーズ対応付け。"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .audio import frame_rms, db_to_lin
from .config import Config
from .guide import Note


@dataclass
class Phrase:
    start_sample: int
    end_sample: int
    sr: int

    @property
    def start(self) -> float:
        return self.start_sample / self.sr

    @property
    def end(self) -> float:
        return self.end_sample / self.sr


@dataclass
class PhrasePair:
    phrase: Phrase
    notes: list[Note] = field(default_factory=list)


def split_audio_phrases(audio: np.ndarray, sr: int, cfg: Config) -> list[Phrase]:
    """RMSゲートでフレーズ分割。無音 >= min_gap_ms を区切りとする。"""
    hop = cfg.hop
    rms = frame_rms(audio, hop, cfg.frame_length)
    active = rms >= db_to_lin(cfg.silence_thresh_db)
    min_gap_frames = max(1, int(cfg.min_gap_ms / 1000.0 * sr / hop))
    min_phrase_frames = max(2, int(0.05 * sr / hop))  # 50ms未満の孤立音は無視

    spans: list[tuple[int, int]] = []  # (開始サンプル, 終了サンプル)
    start = None
    gap = 0
    for i, a in enumerate(active):
        if a:
            if start is None:
                start = i
            gap = 0
        elif start is not None:
            gap += 1
            if gap >= min_gap_frames:
                end = i - gap + 1
                if end - start >= min_phrase_frames:
                    spans.append((start * hop, end * hop))
                start, gap = None, 0
    if start is not None:
        end = len(active) - gap
        if end - start >= min_phrase_frames:
            spans.append((start * hop, end * hop))

    # 前後の無音へマージンを広げる。ワープでノートを動かす余地（max_shift分）と
    # 子音・余韻の取りこぼし防止のため。隣接フレーズとは重ねない。
    margin = int((cfg.max_shift_ms / 1000.0 + 0.05) * sr)
    phrases: list[Phrase] = []
    for k, (s, e) in enumerate(spans):
        lo = 0 if k == 0 else (spans[k - 1][1] + s) // 2
        hi = len(audio) if k == len(spans) - 1 else (e + spans[k + 1][0]) // 2
        phrases.append(Phrase(max(lo, s - margin), min(hi, e + margin), sr))
    return phrases


def split_note_phrases(notes: list[Note], cfg: Config) -> list[list[Note]]:
    """MIDI側: ノート間ギャップ >= min_gap_ms でフレーズ分割。"""
    if not notes:
        return []
    gap = cfg.min_gap_ms / 1000.0
    groups: list[list[Note]] = [[notes[0]]]
    for n in notes[1:]:
        if n.start - groups[-1][-1].end >= gap:
            groups.append([n])
        else:
            groups[-1].append(n)
    return groups


def match_phrases(audio_phrases: list[Phrase], note_groups: list[list[Note]],
                  cfg: Config) -> tuple[list[PhrasePair], list[Phrase]]:
    """ガイドノートを時間オーバーラップで音声フレーズへ割り当てる（頭出し済み前提）。

    ガイドと音声でフレーズ粒度が一致しない（例: 合成ガイドは息継ぎが短く
    長い連続フレーズになる）ため、フレーズ同士の1対1対応ではなく、
    ノート単位で「最もよく重なる音声フレーズ」に割り当てる。

    戻り値: (ノートが割り当てられたペア, ノートの無いオーディオフレーズ)
    対応が取れないものは未処理でスルー（警告ログ）。
    """
    tol = cfg.max_shift_ms / 1000.0 + 0.05
    notes = [n for grp in note_groups for n in grp]
    by_phrase: dict[int, list[Note]] = {i: [] for i in range(len(audio_phrases))}
    for n in notes:
        best_i, best_ov = None, 0.0
        for i, ph in enumerate(audio_phrases):
            ov = min(n.end, ph.end + tol) - max(n.start, ph.start - tol)
            if ov > best_ov:
                best_i, best_ov = i, ov
        # ノートの半分（最大100ms）以上重なっていなければ対応不能とみなす
        if best_i is None or best_ov < min(0.5 * n.duration, 0.1):
            cfg.warn(
                f"ノート {n.start:.2f}-{n.end:.2f}s に対応する音声が無いためスルーします"
            )
            continue
        by_phrase[best_i].append(n)

    pairs: list[PhrasePair] = []
    unmatched: list[Phrase] = []
    for i, ph in enumerate(audio_phrases):
        if by_phrase[i]:
            pairs.append(PhrasePair(phrase=ph, notes=by_phrase[i]))
        else:
            unmatched.append(ph)
            cfg.warn(
                f"フレーズ {ph.start:.2f}-{ph.end:.2f}s に対応するガイドノートが無いためスルーします"
            )
    return pairs, unmatched
