"""CLIのエンドツーエンドテスト（MIDIファイル読み込み・レポート出力込み）。"""

from __future__ import annotations

import json

import numpy as np
import soundfile as sf

from synth import SR, SynthNote, render_voice, write_midi
from vat.cli import main

NOTES = [
    SynthNote(0.50, 0.50, 60, detune_cents=30.0),
    SynthNote(1.16, 0.50, 64, detune_cents=30.0),
    SynthNote(2.20, 0.50, 67, detune_cents=30.0),
]


def test_cli_process(tmp_path):
    wav_in = tmp_path / "in.wav"
    midi = tmp_path / "guide.mid"
    wav_out = tmp_path / "out.wav"
    report = tmp_path / "report.json"

    sf.write(wav_in, render_voice(NOTES, 3.5), SR, subtype="FLOAT")
    write_midi(NOTES, str(midi))

    rc = main([
        "process", str(wav_in), str(midi), "-o", str(wav_out),
        "--detector", "pyin", "--pitch-only", "--pitch-strength", "1.0",
        "--report", str(report),
    ])
    assert rc == 0
    assert wav_out.exists()
    out, sr = sf.read(wav_out)
    assert sr == SR and len(out) > 0

    rep = json.loads(report.read_text(encoding="utf-8"))
    assert rep["config"]["engine"] == "stretch"
    assert len(rep["notes"]) == len(NOTES)
    for n in rep["notes"]:
        assert "offset_cents_before" in n and "skip_reasons" in n


def test_cli_null_report(tmp_path):
    """ずれ無し入力: レポートにスキップ/微小補正が記録され、出力はほぼ入力。"""
    notes = [SynthNote(0.50, 0.50, 60), SynthNote(1.16, 0.50, 64)]
    wav_in = tmp_path / "in.wav"
    midi = tmp_path / "guide.mid"
    wav_out = tmp_path / "out.wav"

    audio = render_voice(notes, 2.3)
    sf.write(wav_in, audio, SR, subtype="FLOAT")
    write_midi(notes, str(midi))

    rc = main(["process", str(wav_in), str(midi), "-o", str(wav_out),
               "--detector", "pyin"])
    assert rc == 0
    out, _ = sf.read(wav_out, dtype="float32")
    assert np.max(np.abs(out - audio)) < 1e-6
