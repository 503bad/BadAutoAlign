"""サービスモード（vat serve）のプロトコルテスト。GUIは起動しない。"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import soundfile as sf

from synth import SR, SynthNote, render_voice

REPO = Path(__file__).parent.parent


def _talk(requests: list[dict], timeout: float = 300.0) -> list[dict]:
    proc = subprocess.Popen(
        ["uv", "--directory", str(REPO), "run", "vat", "serve"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, encoding="utf-8", errors="replace",  # サービスはUTF-8固定（Windowsのcp932に依存しない）
    )
    stdin = "".join(json.dumps(r) + "\n" for r in requests)
    out, err = proc.communicate(stdin, timeout=timeout)
    assert proc.returncode == 0, err
    lines = [ln for ln in out.splitlines() if ln.strip()]
    assert len(lines) == len(requests), f"応答数不一致: {out!r} / stderr: {err[-500:]}"
    return [json.loads(ln) for ln in lines]


def test_version_roundtrip():
    (resp,) = _talk([{"id": 1, "method": "version", "params": {}}])
    assert resp["id"] == 1 and resp["ok"]
    assert "version" in resp["result"]


def test_unknown_method_returns_error():
    (resp,) = _talk([{"id": 2, "method": "nope", "params": {}}])
    assert resp["id"] == 2 and not resp["ok"]
    assert "nope" in resp["error"]


def test_process_returns_report(tmp_path):
    notes = [SynthNote(0.50, 0.50, 60, detune_cents=30.0),
             SynthNote(1.16, 0.50, 64, detune_cents=30.0)]
    guide = [SynthNote(n.start, n.dur, n.midi) for n in notes]
    wav_in = tmp_path / "in.wav"
    wav_guide = tmp_path / "guide.wav"
    wav_out = tmp_path / "out.wav"
    sf.write(wav_in, render_voice(notes, 2.3), SR, subtype="FLOAT")
    sf.write(wav_guide, render_voice(guide, 2.3), SR, subtype="FLOAT")

    resps = _talk([
        {"id": 1, "method": "version", "params": {}},
        {"id": 2, "method": "process", "params": {
            "input": str(wav_in), "guide": str(wav_guide), "output": str(wav_out),
            "options": {"detector": "pyin", "pitch_only": True,
                        "pitch_strength": 1.0},
        }},
    ])
    assert resps[1]["ok"], resps[1]
    report = resps[1]["result"]
    assert wav_out.exists()
    assert report["notes"], "レポートにノートが無い"
    # stdoutがプロトコル専用であること（処理ログが混ざるとJSONにならない）は
    # _talk のパースが通った時点で保証される
