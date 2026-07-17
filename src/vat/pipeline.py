"""処理パイプライン全体の統括。

処理順序（T3）: タイミング補正 → ピッチ再検出 → ピッチ補正。
非処理ルール: 「迷ったら触らない」。対応の取れないフレーズ・閾値未満の
補正量・ガード超過はすべて未処理スルー＋警告。
"""

from __future__ import annotations

import numpy as np

from .audio import load_wav, save_wav
from .config import Config
from .detect import detect_pitch
from .engines import get_engine
from .guide import GuideData, load_guide
from .pitch import NoteReport, build_correction_curve
from .segment import PhrasePair, match_phrases, split_audio_phrases, split_note_phrases
from .timing import build_warp_map


def process_file(input_wav: str, guide_path: str, output_wav: str | None,
                 cfg: Config, manual_anchors: list[dict] | None = None) -> dict:
    audio, sr = load_wav(input_wav)
    guide = load_guide(guide_path, cfg)
    for w in guide.warnings:
        cfg.warn(w)
    print(f"入力: {input_wav} ({len(audio) / sr:.2f}s, {sr}Hz) / "
          f"ガイド: {guide_path} ({guide.source}, {len(guide.notes)}ノート)")

    output, note_reports, phrase_logs = process_audio(
        audio, sr, guide, cfg, manual_anchors=manual_anchors)

    if output_wav:
        save_wav(output_wav, output, sr)
        print(f"出力: {output_wav}")

    report = {
        "version": "0.1.0",
        "input": input_wav,
        "guide": guide_path,
        "output": output_wav,
        "sample_rate": sr,
        "config": cfg.to_dict(),
        "phrases": phrase_logs,
        "notes": [r.to_dict() for r in note_reports],
        "warnings": list(cfg.warnings),
    }
    return report


def process_audio(audio: np.ndarray, sr: int, guide: GuideData,
                  cfg: Config, manual_anchors: list[dict] | None = None,
                  ) -> tuple[np.ndarray, list[NoteReport], list[dict]]:
    engine = get_engine(cfg)
    phrases = split_audio_phrases(audio, sr, cfg)
    note_groups = split_note_phrases(guide.notes, cfg)
    pairs, _ = match_phrases(phrases, note_groups, cfg)
    print(f"フレーズ: 音声 {len(phrases)} / MIDI {len(note_groups)} / 対応 {len(pairs)}")

    output = audio.copy()
    all_reports: list[NoteReport] = []
    phrase_logs: list[dict] = []
    matched_notes = {id(n) for p in pairs for n in p.notes}

    for i, note in enumerate(guide.notes):
        if id(note) not in matched_notes:
            rep = NoteReport(index=i, midi_pitch=note.pitch,
                             start=note.start, end=note.end)
            rep.skip_reasons.append("no_matching_phrase")
            all_reports.append(rep)

    note_index = {id(n): i for i, n in enumerate(guide.notes)}
    for pair in pairs:
        reports = [NoteReport(index=note_index[id(n)], midi_pitch=n.pitch,
                              start=n.start, end=n.end) for n in pair.notes]
        phrase_manual = [
            m for m in (manual_anchors or [])
            if pair.phrase.start <= float(m.get("src_s", -1)) < pair.phrase.end
        ]
        log = _process_phrase(output, sr, pair, engine, cfg, reports, guide,
                              manual_anchors=phrase_manual)
        all_reports.extend(reports)
        phrase_logs.append(log)
        print(f"  フレーズ {log['start_s']:.2f}-{log['end_s']:.2f}s: "
              f"timing={'適用' if log['timing_applied'] else 'なし'} "
              f"pitch={'適用' if log['pitch_applied'] else 'なし'} "
              f"({log['n_notes']}ノート)")

    all_reports.sort(key=lambda r: r.index)
    return output, all_reports, phrase_logs


def _process_phrase(output: np.ndarray, sr: int, pair: PhrasePair, engine,
                    cfg: Config, reports: list[NoteReport],
                    guide: GuideData,
                    manual_anchors: list[dict] | None = None) -> dict:
    ph = pair.phrase
    seg = output[ph.start_sample: ph.end_sample].copy()
    phrase_len_s = len(seg) / sr
    log = {
        "start_s": round(ph.start, 3),
        "end_s": round(ph.end, 3),
        "n_notes": len(pair.notes),
        "timing_applied": False,
        "pitch_applied": False,
    }

    track = detect_pitch(seg, sr, cfg)

    # --- タイミング補正（T2） ---
    if not cfg.pitch_only and cfg.timing_strength > 0:
        guide_ctx = _make_guide_ctx(seg, sr, pair, guide, cfg)
        warp, align_table, base_ms, lag_profile = build_warp_map(
            seg, sr, track, pair.notes, ph.start, phrase_len_s, cfg,
            reports, guide_ctx=guide_ctx, manual_anchors=manual_anchors)
        log["alignment"] = [e.to_dict() for e in align_table]
        log["base_shift_ms"] = round(base_ms, 1)
        log["lag_profile"] = lag_profile
        if not warp.is_identity(eps=0.002):
            seg = engine.time_warp(seg, sr, warp)
            log["timing_applied"] = True
            track = detect_pitch(seg, sr, cfg)  # T3: ワープ後に再検出
        else:
            for r in reports:
                r.timing_applied = False

    # --- ピッチ補正（P2/P3） ---
    if not cfg.timing_only and cfg.pitch_strength > 0:
        guide_curve = None
        if guide.source == "wav" and cfg.pitch_target == "curve":
            guide_curve = (guide.f0_times, guide.f0_midi)
        curve = build_correction_curve(track, pair.notes, ph.start, cfg,
                                       reports, guide_curve=guide_curve)
        if np.max(np.abs(curve)) * 100.0 >= cfg.min_correction_cents:
            seg = engine.pitch_shift(seg, sr, track, curve)
            log["pitch_applied"] = True
        else:
            for r in reports:
                if not r.skip_reasons and r.applied_cents is not None:
                    r.skip_reasons.append("correction_below_threshold")

    if log["timing_applied"] or log["pitch_applied"]:
        _splice(output, seg, ph.start_sample, sr)
    return log


def _make_guide_ctx(seg: np.ndarray, sr: int, pair: PhrasePair,
                    guide: GuideData, cfg: Config) -> dict | None:
    """WAVガイド時: 音響アライメント用に、このフレーズのノート範囲に対応する
    ガイド音声セグメント（±max_shiftの余白付き）を切り出す。"""
    if guide.audio is None or guide.f0_midi is None or len(guide.f0_times) < 2:
        return None
    dt_g = float(guide.f0_times[1] - guide.f0_times[0])
    pad = cfg.max_shift_ms / 1000.0 + 0.15
    ig0 = max(0, int((pair.notes[0].start - pad) / dt_g))
    ig1 = min(len(guide.f0_midi), int((pair.notes[-1].end + pad) / dt_g) + 1)
    if ig1 - ig0 < 4:
        return None
    a0 = ig0 * cfg.hop
    a1 = min(len(guide.audio), ig1 * cfg.hop)
    return {
        "guide_audio": guide.audio[a0:a1],
        "guide_sr": guide.sr,
        "guide_semis": guide.f0_midi[ig0:ig1],
        "guide_seg_start": ig0 * dt_g,
    }


def _splice(output: np.ndarray, seg: np.ndarray, start: int, sr: int,
            fade_ms: float = 10.0) -> None:
    """処理済みフレーズを出力へ書き戻す。境界は短いクロスフェード。"""
    n = len(seg)
    fade = min(int(fade_ms / 1000.0 * sr), n // 4)
    mixed = seg.copy()
    if fade > 0:
        ramp = np.linspace(0.0, 1.0, fade)
        orig = output[start: start + n]
        mixed[:fade] = orig[:fade] * (1 - ramp) + seg[:fade] * ramp
        mixed[-fade:] = orig[n - fade: n] * ramp[::-1] + seg[-fade:] * (1 - ramp[::-1])
    output[start: start + n] = mixed
