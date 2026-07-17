"""合成テスト用の擬似ボーカル生成（正解が既知のテストデータ）。

ノコギリ波＋フォルマントフィルタで倍音構造のある擬似歌声を作り、
既知のピッチずれ（セント）とタイミングずれ（ms）を人工的に加える。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import iirpeak, lfilter

from vat.guide import Note

SR = 44100


@dataclass
class SynthNote:
    start: float          # 正解（ガイド）の開始時刻 [s]
    dur: float
    midi: float
    detune_cents: float = 0.0   # 音声側に加えるピッチずれ
    shift_ms: float = 0.0       # 音声側に加えるタイミングずれ
    scoop_cents: float = 0.0    # しゃくり: 発声開始時の音高オフセット（例 -150）
    scoop_ms: float = 0.0       # しゃくりが目標音高に到達するまでの時間


def midi_to_hz(m: float) -> float:
    return 440.0 * 2 ** ((m - 69.0) / 12.0)


def render_voice(notes: list[SynthNote], total_s: float, sr: int = SR,
                 vibrato_cents: float = 8.0, vibrato_hz: float = 5.5) -> np.ndarray:
    """各ノートを個別にレンダリングしてミックスする（ずれ付与のため）。"""
    out = np.zeros(int(total_s * sr))
    rng = np.random.default_rng(42)
    for note in notes:
        n = int(note.dur * sr)
        t = np.arange(n) / sr
        f0 = midi_to_hz(note.midi) * 2 ** (note.detune_cents / 1200.0)
        vib = vibrato_cents * np.sin(2 * np.pi * vibrato_hz * t + rng.uniform(0, 6.28))
        scoop = np.zeros(n)
        if note.scoop_ms > 0 and note.scoop_cents != 0.0:
            k = int(note.scoop_ms / 1000.0 * sr)
            scoop[:k] = note.scoop_cents * (1.0 - np.arange(min(k, n)) / k)
        inst_f = f0 * 2 ** ((vib + scoop) / 1200.0)
        phase = 2 * np.pi * np.cumsum(inst_f) / sr
        # 帯域制限ノコギリ波（倍音10本）
        sig = np.zeros(n)
        for k in range(1, 11):
            sig += np.sin(k * phase) / k
        env = np.ones(n)
        a = int(0.02 * sr)
        r = int(0.04 * sr)
        env[:a] = np.linspace(0, 1, a)
        env[-r:] *= np.linspace(1, 0, r)
        start = note.start + note.shift_ms / 1000.0
        s = int(start * sr)
        e = min(len(out), s + n)
        if s >= 0 and e > s:
            out[s:e] += (sig * env)[: e - s]
    # フォルマントフィルタ（2共振: 700Hz / 1200Hz）
    for freq, q in ((700, 4.0), (1200, 6.0)):
        b, a_ = iirpeak(freq / (sr / 2), q)
        out = lfilter(b, a_, out)
    out = 0.5 * out / (np.max(np.abs(out)) + 1e-9)
    return out.astype(np.float32)


def guide_notes(notes: list[SynthNote]) -> list[Note]:
    return [Note(n.start, n.start + n.dur, n.midi) for n in notes]


def write_midi(notes: list[SynthNote], path: str) -> None:
    import pretty_midi

    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=0)
    for n in notes:
        inst.notes.append(pretty_midi.Note(
            velocity=100, pitch=int(round(n.midi)),
            start=n.start, end=n.start + n.dur,
        ))
    pm.instruments.append(inst)
    pm.write(path)
