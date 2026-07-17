"""ガイドアダプタのWAV対応（M5）: 擬似ノート化と note/curve モード。"""

from __future__ import annotations

import numpy as np
import soundfile as sf
import pytest

from synth import SR, SynthNote, guide_notes, render_voice
from test_correction import measure_pitch_residuals
from vat.cli import main
from vat.config import Config
from vat.guide import GuideData, load_wav_guide

NOTES = [
    SynthNote(0.50, 0.50, 60),
    SynthNote(1.16, 0.50, 64),
    SynthNote(2.20, 0.50, 67),
]


def test_wav_guide_pseudo_notes(tmp_path):
    """ガイドWAVから擬似ノート列が抽出できる。"""
    guide_wav = tmp_path / "guide.wav"
    sf.write(guide_wav, render_voice(NOTES, 3.5), SR, subtype="FLOAT")
    cfg = Config(detector="pyin")
    guide = load_wav_guide(str(guide_wav), cfg)
    assert guide.source == "wav"
    assert len(guide.notes) == len(NOTES)
    for note, ref in zip(guide.notes, NOTES):
        assert abs(note.pitch - ref.midi) < 0.15, f"擬似ノート音高 {note.pitch} vs {ref.midi}"
        assert abs(note.start - ref.start) < 0.06


@pytest.mark.parametrize("target", ["note", "curve"])
def test_wav_guide_correction(tmp_path, target):
    """+30セントずれた入力をWAVガイドで補正できる（note/curve両モード）。"""
    guide_wav = tmp_path / "guide.wav"
    wav_in = tmp_path / "in.wav"
    wav_out = tmp_path / "out.wav"
    sf.write(guide_wav, render_voice(NOTES, 3.5), SR, subtype="FLOAT")
    detuned = [SynthNote(n.start, n.dur, n.midi, detune_cents=30.0) for n in NOTES]
    sf.write(wav_in, render_voice(detuned, 3.5), SR, subtype="FLOAT")

    rc = main([
        "process", str(wav_in), str(guide_wav), "-o", str(wav_out),
        "--detector", "pyin", "--pitch-only", "--pitch-strength", "1.0",
        "--pitch-target", target,
    ])
    assert rc == 0
    out, _ = sf.read(wav_out, dtype="float32")
    ref_guide = GuideData(notes=guide_notes(NOTES), source="midi")
    after = measure_pitch_residuals(out, ref_guide)
    assert abs(np.median(after)) < 12.0, f"補正後残差 {np.median(after):.1f}cent"


def test_wav_guide_timing_acoustic_dtw(tmp_path):
    """WAVガイドのタイミング補正（音響特徴DTW経路）。

    同一音高が続く音節を含むケース: F0だけでは区別できないが、
    音響特徴＋アンカーピッチ照合で正しく補正できること。
    """
    from test_correction import measure_onset_residuals

    notes = [
        SynthNote(0.50, 0.40, 60),
        SynthNote(1.00, 0.40, 60),   # 同一音高の連続
        SynthNote(1.50, 0.40, 64),
        SynthNote(2.05, 0.40, 62),
    ]
    shifted = [SynthNote(n.start, n.dur, n.midi, shift_ms=80.0) for n in notes]
    guide_wav = tmp_path / "guide.wav"
    wav_in = tmp_path / "in.wav"
    wav_out = tmp_path / "out.wav"
    sf.write(guide_wav, render_voice(notes, 3.2), SR, subtype="FLOAT")
    sf.write(wav_in, render_voice(shifted, 3.2), SR, subtype="FLOAT")

    rc = main([
        "process", str(wav_in), str(guide_wav), "-o", str(wav_out),
        "--detector", "pyin", "--timing-only",
    ])
    assert rc == 0
    out, _ = sf.read(wav_out, dtype="float32")
    ref_guide = GuideData(notes=guide_notes(notes), source="midi")
    before = measure_onset_residuals(render_voice(shifted, 3.2), ref_guide)
    after = measure_onset_residuals(out, ref_guide)
    assert np.median(np.abs(before)) > 50.0
    assert np.median(np.abs(after)) < 25.0, f"補正後残差 {np.median(np.abs(after)):.1f}ms"


def measure_arrival_residuals(audio, ref_notes) -> list[float]:
    """ピッチ到達点（芯）とガイドノート開始時刻の差（ms）。"""
    from vat.detect import detect_pitch
    from vat.pcenter import pitch_arrival_time

    cfg = Config(detector="pyin")
    track = detect_pitch(audio, SR, cfg)
    dt = float(track.times[1] - track.times[0])
    semis = np.nan_to_num(track.semitones(), nan=0.0)
    out = []
    for n in ref_notes:
        arr = pitch_arrival_time(track.times, semis, track.voiced,
                                 max(0.0, n.start - 0.05), n.start + 0.4, dt)
        if arr is not None:
            out.append((arr - n.start) * 1000.0)
    return out


def test_wav_guide_scoop_robust(tmp_path):
    """しゃくり（発声がノート音高より低く始まる）があっても、
    エネルギー包絡ベースのタイミング補正が乱れないこと。

    ボーカルは全ノートしゃくり付き＋80ms遅れ。補正後はエネルギーの
    タイミング（オンセット）がガイドに揃うこと。しゃくり自体は表現として
    保持される（芯=ピッチ到達点を強制的に前へ詰めることはしない）。
    """
    notes = [
        SynthNote(0.50, 0.45, 60),
        SynthNote(1.10, 0.45, 64),
        SynthNote(1.70, 0.45, 62),
        SynthNote(2.30, 0.45, 65),
    ]
    scooped = [SynthNote(n.start, n.dur, n.midi, shift_ms=80.0,
                         scoop_cents=-180.0, scoop_ms=100.0) for n in notes]
    guide_wav = tmp_path / "guide.wav"
    wav_in = tmp_path / "in.wav"
    wav_out = tmp_path / "out.wav"
    sf.write(guide_wav, render_voice(notes, 3.3), SR, subtype="FLOAT")
    vocal = render_voice(scooped, 3.3)
    sf.write(wav_in, vocal, SR, subtype="FLOAT")

    rc = main([
        "process", str(wav_in), str(guide_wav), "-o", str(wav_out),
        "--detector", "pyin", "--timing-only",
    ])
    assert rc == 0
    out, _ = sf.read(wav_out, dtype="float32")

    ref = GuideData(notes=guide_notes(notes), source="midi")
    from test_correction import measure_onset_residuals
    before = measure_onset_residuals(vocal, ref)
    after = measure_onset_residuals(out, ref)
    # 注: しゃくり信号は有声化検出自体が数十ms遅れるため、絶対残差ではなく
    # 「補正量が付与した80msのずれを打ち消したこと」を判定する
    corrected_by = np.median(before) - np.median(after)
    assert np.median(np.abs(before)) > 60.0, "テストデータのずれが想定より小さい"
    assert 55.0 < corrected_by < 105.0, f"補正量 {corrected_by:.1f}ms（期待 ~80ms）"
    assert np.median(np.abs(after)) < 45.0, f"補正後残差 {np.median(np.abs(after)):.1f}ms"
