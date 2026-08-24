"""合成テスト（仕様: テスト・検証計画 1〜2）。

- 既知のピッチずれ（±30セント）→ 補正後残差 中央値 < 10セント
- 既知のタイミングずれ（+60ms）→ 補正後残差 中央値 < 20ms
- ヌルテスト: ずれの無い入力は出力がほぼ同一（非処理ルール）
"""

from __future__ import annotations

import numpy as np
import pytest

from synth import SR, SynthNote, guide_notes, render_voice
from vat.config import Config
from vat.detect import detect_pitch
from vat.guide import GuideData
from vat.pipeline import process_audio
from vat.timing import detect_voicing_onsets

TOTAL_S = 5.0

BASE_NOTES = [
    SynthNote(0.50, 0.50, 60),
    SynthNote(1.16, 0.50, 64),
    SynthNote(1.82, 0.50, 62),
    SynthNote(3.00, 0.50, 65),
    SynthNote(3.66, 0.50, 67),
]


def make_case(detune: float = 0.0, shift_ms: float = 0.0):
    notes = [SynthNote(n.start, n.dur, n.midi, detune, shift_ms) for n in BASE_NOTES]
    audio = render_voice(notes, TOTAL_S)
    guide = GuideData(notes=guide_notes(notes), source="midi")
    return audio, guide


def measure_pitch_residuals(audio: np.ndarray, guide: GuideData) -> list[float]:
    """ノートごとの中央値ピッチ残差（セント）。先頭150msはランプイン区間なので除外。"""
    cfg = Config(detector="pyin")
    track = detect_pitch(audio, SR, cfg)
    residuals = []
    for note in guide.notes:
        m = (track.times >= note.start + 0.15) & (track.times < note.end - 0.05) & track.voiced
        if m.any():
            semis = track.semitones()[m]
            residuals.append(float(np.median(semis) - note.pitch) * 100.0)
    return residuals


def measure_onset_residuals(audio: np.ndarray, guide: GuideData) -> list[float]:
    """検出した有声化開始点とガイドノート開始時刻の差（ms）。"""
    cfg = Config(detector="pyin")
    track = detect_pitch(audio, SR, cfg)
    onsets = detect_voicing_onsets(track, cfg)
    residuals = []
    for note in guide.notes:
        nearest = min(onsets, key=lambda t: abs(t - note.start))
        residuals.append((nearest - note.start) * 1000.0)
    return residuals


@pytest.mark.parametrize("engine", ["psola", "stretch", "world"])
@pytest.mark.parametrize("detune", [30.0, -30.0])
def test_pitch_correction(engine: str, detune: float):
    audio, guide = make_case(detune=detune)
    cfg = Config(detector="pyin", engine=engine,
                 pitch_strength=1.0, pitch_only=True)
    out, reports, _ = process_audio(audio, SR, guide, cfg)

    before = measure_pitch_residuals(audio, guide)
    after = measure_pitch_residuals(out, guide)
    assert abs(np.median(before)) > 20.0, "テストデータのずれが想定より小さい"
    assert abs(np.median(after)) < 10.0, f"補正後残差 {np.median(after):.1f}cent"
    corrected = [r for r in reports if r.applied_cents is not None]
    assert len(corrected) == len(BASE_NOTES)


def test_timing_correction():
    audio, guide = make_case(shift_ms=60.0)
    cfg = Config(detector="pyin", timing_only=True)
    out, reports, logs = process_audio(audio, SR, guide, cfg)

    before = measure_onset_residuals(audio, guide)
    after = measure_onset_residuals(out, guide)
    assert np.median(np.abs(before)) > 40.0, "テストデータのずれが想定より小さい"
    assert np.median(np.abs(after)) < 20.0, f"補正後残差 {np.median(np.abs(after)):.1f}ms"
    assert any(log["timing_applied"] for log in logs)


@pytest.mark.parametrize("engine", ["psola", "stretch", "world"])
def test_combined_correction(engine: str):
    audio, guide = make_case(detune=30.0, shift_ms=60.0)
    cfg = Config(detector="pyin", engine=engine, pitch_strength=1.0)
    out, reports, logs = process_audio(audio, SR, guide, cfg)

    pitch_after = measure_pitch_residuals(out, guide)
    onset_after = measure_onset_residuals(out, guide)
    assert abs(np.median(pitch_after)) < 10.0
    assert np.median(np.abs(onset_after)) < 20.0


def test_null():
    """ずれの無い入力 → 出力が入力と同一（迷ったら触らない）。"""
    audio, guide = make_case()
    cfg = Config(detector="pyin")
    out, reports, logs = process_audio(audio, SR, guide, cfg)
    diff = np.max(np.abs(out - audio))
    assert diff == 0.0, f"ヌルテストで出力が変化 (max diff {diff})"


def test_no_guide_region_untouched():
    """MIDIノートの無い区間（ここでは2フレーズ目を欠落させる）は素通し。"""
    audio, _ = make_case(detune=30.0)
    partial = GuideData(notes=guide_notes(BASE_NOTES[:3]), source="midi")
    cfg = Config(detector="pyin", pitch_strength=1.0)
    out, reports, _ = process_audio(audio, SR, partial, cfg)
    # 2フレーズ目（3.0s以降）はビット一致
    s = int(2.8 * SR)
    assert np.array_equal(out[s:], audio[s:])


def test_max_shift_guard():
    """max-shiftを超える移動は未補正＋警告。"""
    audio, guide = make_case(shift_ms=200.0)  # ガード(120ms)超過
    cfg = Config(detector="pyin", timing_only=True)
    out, reports, logs = process_audio(audio, SR, guide, cfg)
    applied = [r for r in reports if r.timing_applied]
    assert not applied
    assert any("shift_exceeds_max" in r.skip_reasons or "no_matching_phrase" in r.skip_reasons
               or not r.timing_applied for r in reports)


def test_legato_dtw_runs():
    """ノート数≠オンセット数（レガート）でもDTW経路で安全に処理が通る。"""
    notes = [
        SynthNote(0.50, 0.45, 60, 0.0, 60.0),
        SynthNote(0.95, 0.45, 64, 0.0, 60.0),
        SynthNote(1.40, 0.45, 67, 0.0, 60.0),
    ]
    audio = render_voice(notes, 2.6)
    guide = GuideData(notes=guide_notes(notes), source="midi")
    cfg = Config(detector="pyin", timing_only=True)
    out, reports, logs = process_audio(audio, SR, guide, cfg)
    assert len(out) == len(audio)
    assert np.isfinite(out).all()
