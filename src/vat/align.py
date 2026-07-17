"""ガイドノート列 ⇔ ボーカル音節列の離散アライメント。

日本語歌唱の「1ノート ≒ 1モーラ」構造を前提に、順序保存・挿入/欠落
ペナルティ付きのNeedleman-Wunsch型DPで対応付ける。フレームDTWと違い、
1音節が複数ノートに「にじむ」ことがなく、音数が一致すれば先頭からの
1:1対応に自然に収束する（掛け違い対策の核）。

判断材料と判定結果は AlignmentEntry の表として返し、レポートに出力する
（人間/AIによる事後レビュー・裁定工程を後付けできるようにするため）。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import Config
from .guide import Note
from .syllables import Syllable

GAP = 0.9          # ノート/音節を対応なしにするコスト
GOOD_MATCH = 0.65  # これ未満のコストなら高信頼


@dataclass
class AlignmentEntry:
    note_index: int              # フレーズ内インデックス
    note_start: float            # 絶対時刻 [s]
    note_pitch: float
    syl_onset: float | None      # 対応音節の頭（フレーズ先頭基準）。None=対応なし
    syl_semitone: float | None
    cost: float | None
    decision: str                # matched | note_skipped | low_confidence
    syl_index: int | None = None # 対応音節のインデックス

    def to_dict(self) -> dict:
        return {
            "note_index": self.note_index,
            "note_start_s": round(self.note_start, 3),
            "note_pitch": round(self.note_pitch, 2),
            "syllable_onset_s": None if self.syl_onset is None else round(self.syl_onset, 3),
            "syllable_semitone": None if self.syl_semitone is None else round(self.syl_semitone, 2),
            "cost": None if self.cost is None else round(self.cost, 3),
            "decision": self.decision,
        }


def align_notes_to_syllables(
    notes: list[Note],
    syllables: list[Syllable],
    phrase_start: float,
    cfg: Config,
    guide_feats: list[np.ndarray] | None = None,
) -> tuple[list[tuple[float, float]], list[bool], list[AlignmentEntry]]:
    """戻り値: (matched=(音節頭, ノート頭ローカル時刻)のリスト, 信頼フラグ, 対応表)。

    guide_feats: ノートごとの母音特徴重心（ガイド音声がある場合）。
    """
    ng, nv = len(notes), len(syllables)
    if ng == 0 or nv == 0:
        return [], [], [
            AlignmentEntry(i, n.start, n.pitch, None, None, None, "note_skipped")
            for i, n in enumerate(notes)
        ]

    note_local = [n.start - phrase_start for n in notes]
    cost = _pair_costs(notes, note_local, syllables, cfg, guide_feats)

    # Needleman-Wunsch（コスト最小化）
    D = np.full((ng + 1, nv + 1), np.inf)
    D[0, :] = np.arange(nv + 1) * GAP
    D[:, 0] = np.arange(ng + 1) * GAP
    move = np.zeros((ng + 1, nv + 1), dtype=np.int8)  # 1=match 2=skip_note 3=skip_syl
    for i in range(1, ng + 1):
        for j in range(1, nv + 1):
            cands = (
                (D[i - 1, j - 1] + cost[i - 1, j - 1], 1),
                (D[i - 1, j] + GAP, 2),
                (D[i, j - 1] + GAP, 3),
            )
            D[i, j], move[i, j] = min(cands, key=lambda t: t[0])

    pairs: dict[int, int] = {}
    i, j = ng, nv
    while i > 0 or j > 0:
        m = move[i, j] if i > 0 and j > 0 else (2 if i > 0 else 3)
        if m == 1:
            pairs[i - 1] = j - 1
            i, j = i - 1, j - 1
        elif m == 2:
            i -= 1
        else:
            j -= 1

    matched: list[tuple[float, float]] = []
    confident: list[bool] = []
    table: list[AlignmentEntry] = []
    for k, note in enumerate(notes):
        if k in pairs:
            s = syllables[pairs[k]]
            c = float(cost[k, pairs[k]])
            ok = c <= GOOD_MATCH
            matched.append((s.onset, note_local[k]))
            confident.append(ok)
            table.append(AlignmentEntry(
                k, note.start, note.pitch, s.onset,
                None if not np.isfinite(s.semitone) else s.semitone,
                c, "matched" if ok else "low_confidence",
                syl_index=pairs[k]))
        else:
            table.append(AlignmentEntry(
                k, note.start, note.pitch, None, None, None, "note_skipped"))
    return matched, confident, table


def _pair_costs(notes: list[Note], note_local: list[float],
                syllables: list[Syllable], cfg: Config,
                guide_feats: list[np.ndarray] | None) -> np.ndarray:
    """ノートi×音節jの対応コスト。

    - ピッチクラス差（オクターブ非依存）: 主要な手掛かり
    - 位置差: 頭出し済み前提。max_shift程度までは無罰、それ以上で急増
    - 長さ比: 補助
    - 母音特徴距離（ガイド音声がある場合のみ）: 同じ発音かどうか
    """
    ng, nv = len(notes), len(syllables)
    cost = np.zeros((ng, nv))

    v_feats = None
    if guide_feats is not None:
        vf = np.array([s.feat for s in syllables])
        gf = np.array(guide_feats)
        # 双方の特徴をそれぞれzスコア化（話者・音色差の正規化）
        vf = (vf - vf.mean(0)) / (vf.std(0) + 1e-6)
        gf = (gf - gf.mean(0)) / (gf.std(0) + 1e-6)
        v_feats, g_feats = vf, gf

    free = cfg.max_shift_ms / 1000.0 + 0.05
    for i, note in enumerate(notes):
        for j, syl in enumerate(syllables):
            c = 0.0
            if np.isfinite(syl.semitone):
                fold = abs((syl.semitone - note.pitch + 6.0) % 12.0 - 6.0)
                c += min(fold, 3.0) / 3.0 * 0.55
            else:
                c += 0.3
            dpos = abs(syl.onset - note_local[i])
            # 許容域内でも近い候補をわずかに優先（迷ったら動かさない方向）
            c += min(dpos / free, 1.0) * 0.15
            c += min(max(0.0, dpos - free) * 2.5, 1.5)
            dur_ratio = abs(np.log2(max(syl.duration, 0.03) / max(note.duration, 0.03)))
            c += min(dur_ratio, 2.0) * 0.1
            if v_feats is not None:
                d = np.linalg.norm(g_feats[i] - v_feats[j]) / np.sqrt(v_feats.shape[1])
                c += min(d, 2.0) * 0.15
            cost[i, j] = c
    return cost
