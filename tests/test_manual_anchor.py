"""残差レポートと手動アンカー（GUIのマーカードラッグ→再補正）のテスト。

ケース: ガイドはほぼレガート、ボーカルの後半の語が+160ms遅れ
（max-shiftガード120msを超えるため自動では補正されない）。
1. レポートに「まだ残っているズレ」(timing_residual_ms) が出ること
2. 手動アンカーを渡すとガード免除で補正されること
"""

from __future__ import annotations

import numpy as np
import soundfile as sf

from synth import SR, SynthNote, guide_notes, render_voice
from test_correction import measure_onset_residuals
from vat.config import Config
from vat.guide import GuideData
from vat.pipeline import process_file

WORD_A = [
    SynthNote(0.50, 0.30, 60),
    SynthNote(0.80, 0.30, 62),
    SynthNote(1.10, 0.32, 64),
]
WORD_B = [
    SynthNote(1.44, 0.30, 65),
    SynthNote(1.74, 0.36, 67),
]


def _prepare(tmp_path):
    guide_wav = tmp_path / "guide.wav"
    wav_in = tmp_path / "in.wav"
    sf.write(guide_wav, render_voice(WORD_A + WORD_B, 2.9), SR, subtype="FLOAT")
    vocal_notes = WORD_A + [
        SynthNote(n.start, n.dur, n.midi, shift_ms=160.0) for n in WORD_B
    ]
    sf.write(wav_in, render_voice(vocal_notes, 2.9), SR, subtype="FLOAT")
    return guide_wav, wav_in


def test_residual_reported_and_manual_anchor_fixes(tmp_path):
    guide_wav, wav_in = _prepare(tmp_path)
    out1 = tmp_path / "out1.wav"

    cfg = Config(detector="pyin", timing_only=True)
    report = process_file(str(wav_in), str(guide_wav), str(out1), cfg)

    # 1. B領域(1.4s以降)のノートに大きな残差が報告されること
    late = [n for n in report["notes"]
            if n["start_s"] > 1.4 and n["timing_residual_ms"] is not None]
    assert late, "残差が1件も報告されていない"
    worst = max(late, key=lambda n: abs(n["timing_residual_ms"]))
    assert abs(worst["timing_residual_ms"]) > 100.0, worst
    # lag_profile がレポートに含まれること（UIの表示データ）
    assert any("lag_profile" in ph for ph in report["phrases"])

    # 2. その残差ノートのアンカーを手動確定して再補正 → ガード免除で直る
    manual = [{
        "src_s": worst["anchor_src_s"],
        "dst_s": worst["anchor_dst_s"],
        "note_index": worst["index"],
    }]
    out2 = tmp_path / "out2.wav"
    cfg2 = Config(detector="pyin", timing_only=True)
    report2 = process_file(str(wav_in), str(guide_wav), str(out2), cfg2,
                           manual_anchors=manual)

    fixed = next(n for n in report2["notes"] if n["index"] == worst["index"])
    assert fixed["manual"] and fixed["timing_applied"]

    out_audio, _ = sf.read(out2, dtype="float32")
    ref_b = GuideData(notes=guide_notes(WORD_B), source="midi")
    after = measure_onset_residuals(out_audio, ref_b)
    assert np.median(np.abs(after)) < 60.0, f"手動アンカー後の残差 {np.median(np.abs(after)):.1f}ms"
