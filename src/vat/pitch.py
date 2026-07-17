"""ピッチ補正カーブの生成（P2/P3）。

フレーズ単位で「フレームごとの補正量（半音）」カーブを作る。
- ノートごとに検出F0中央値とターゲットの差分（オフセット）のみ補正 → 揺らぎは保持
- ガウシアン平滑でノート境界のジャンプを除去
- ノート先頭 attack_preserve_ms はランプイン（しゃくり保持）
- 無声・低信頼度・ノート外フレームは補正0（ソフトマスクでクロスフェード）
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import gaussian_filter1d

from .config import Config
from .detect import PitchTrack
from .guide import Note


@dataclass
class NoteReport:
    index: int
    midi_pitch: float
    start: float                 # [s] 絶対時刻
    end: float
    detected_median_hz: float | None = None
    offset_cents_before: float | None = None
    applied_cents: float | None = None
    timing_shift_ms: float | None = None
    timing_applied: bool = False
    timing_applied_ms: float | None = None   # この位置で実際に適用された移動量
    timing_residual_ms: float | None = None  # 適用後もまだ残っているズレ（信頼できる計測がある場合のみ）
    anchor_src_s: float | None = None   # ボーカル側の芯（補正前タイムライン、絶対時刻）
    anchor_dst_s: float | None = None   # ガイド側の芯（絶対時刻）
    manual: bool = False                # 手動アンカー由来か
    skip_reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "midi_pitch": self.midi_pitch,
            "start_s": round(self.start, 4),
            "end_s": round(self.end, 4),
            "detected_median_hz": _r(self.detected_median_hz),
            "offset_cents_before": _r(self.offset_cents_before),
            "applied_cents": _r(self.applied_cents),
            "timing_shift_ms": _r(self.timing_shift_ms),
            "timing_applied": self.timing_applied,
            "timing_applied_ms": _r(self.timing_applied_ms),
            "timing_residual_ms": _r(self.timing_residual_ms),
            "anchor_src_s": None if self.anchor_src_s is None else round(self.anchor_src_s, 4),
            "anchor_dst_s": None if self.anchor_dst_s is None else round(self.anchor_dst_s, 4),
            "manual": self.manual,
            "skip_reasons": self.skip_reasons,
        }


def _r(x):
    return None if x is None else round(float(x), 2)


def build_correction_curve(
    track: PitchTrack,
    notes: list[Note],
    phrase_start: float,
    cfg: Config,
    reports: list[NoteReport],
    guide_curve: tuple[np.ndarray, np.ndarray] | None = None,
) -> np.ndarray:
    """フレーズ内の各フレームの補正量（半音）を返す。

    track はフレーズ切り出し音声に対する検出結果（時刻はフレーズ先頭基準）。
    notes / reports の時刻は絶対時刻。guide_curve=(times, midi) を渡すと
    curveモード（ガイドF0カーブ転写）になる。
    """
    n = track.n_frames
    dt = float(track.times[1] - track.times[0]) if n > 1 else cfg.hop / 48000.0
    semis = track.semitones()
    usable = track.voiced & (track.conf >= cfg.min_voiced_conf)

    raw = np.zeros(n)
    note_mask = np.zeros(n)

    for rep, note in _iter_note_reports(notes, reports):
        s = int(round((note.start - phrase_start) / dt))
        e = int(round((note.end - phrase_start) / dt))
        s, e = max(0, s), min(n, e)
        if e <= s:
            rep.skip_reasons.append("note_outside_phrase")
            continue
        seg_usable = usable[s:e]
        if not seg_usable.any():
            rep.skip_reasons.append("low_confidence")
            continue
        med = float(np.median(semis[s:e][seg_usable]))
        rep.detected_median_hz = 440.0 * 2 ** ((med - 69.0) / 12.0)

        if guide_curve is not None:
            # curveモード: ガイドF0カーブとの差分をフレームごとに転写
            gt, gm = guide_curve
            g_local = np.interp(track.times[s:e] + phrase_start, gt,
                                np.nan_to_num(gm, nan=0.0))
            g_valid = g_local > 0
            offset_frames = np.where(g_valid & seg_usable, g_local - semis[s:e], 0.0)
            offset = float(np.median(offset_frames[g_valid & seg_usable])) if (g_valid & seg_usable).any() else 0.0
            rep.offset_cents_before = offset * 100.0
            if abs(offset) * 100.0 > cfg.max_correction_cents:
                rep.skip_reasons.append("offset_exceeds_guard")
                continue
            raw[s:e] = np.where(seg_usable, offset_frames, 0.0) * _attack_ramp(e - s, dt, cfg)
        else:
            # noteモード: 中央値オフセットのみ補正（ビブラート等は保持）
            offset = note.pitch - med
            rep.offset_cents_before = offset * 100.0
            if abs(offset) * 100.0 > cfg.max_correction_cents:
                rep.skip_reasons.append("offset_exceeds_guard")
                cfg.warn(
                    f"ノート{rep.index}: オフセット{offset * 100:.0f}centがガードを超過、スキップ"
                )
                continue
            raw[s:e] = offset * _attack_ramp(e - s, dt, cfg)

        note_mask[s:e] = 1.0
        rep.applied_cents = float(np.median(raw[s:e][seg_usable])) * 100.0 * cfg.pitch_strength

    strength_curve = raw * cfg.pitch_strength

    # P3: 時間方向の平滑化（ノート境界の急峻なジャンプを除去）
    sigma_frames = max(1e-3, cfg.pitch_smooth_ms / 1000.0 / dt)
    smooth = gaussian_filter1d(strength_curve, sigma=sigma_frames, mode="nearest")

    # P2: 無声フレームはシフト比1.0で素通し。境界は短いクロスフェード
    fade_frames = max(1e-3, cfg.voicing_fade_ms / 1000.0 / dt)
    voiced_soft = gaussian_filter1d(usable.astype(float), sigma=fade_frames, mode="nearest")
    note_soft = gaussian_filter1d(note_mask, sigma=fade_frames, mode="nearest")

    return smooth * voiced_soft * note_soft


def _attack_ramp(n_frames: int, dt: float, cfg: Config) -> np.ndarray:
    """ノート先頭 attack_preserve_ms で補正強度を0→1にランプイン。"""
    ramp_frames = int(cfg.attack_preserve_ms / 1000.0 / dt)
    ramp = np.ones(n_frames)
    k = min(ramp_frames, n_frames)
    if k > 0:
        ramp[:k] = np.linspace(0.0, 1.0, k, endpoint=False)
    return ramp


def _iter_note_reports(notes: list[Note], reports: list[NoteReport]):
    by_key = {(round(r.start, 6), round(r.end, 6)): r for r in reports}
    for note in notes:
        rep = by_key.get((round(note.start, 6), round(note.end, 6)))
        if rep is not None:
            yield rep, note
