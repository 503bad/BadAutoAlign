"""ガイドアダプタ — MIDI / WAV ガイドを内部表現（ノート列＋任意でF0カーブ）へ正規化する。"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import Config


@dataclass
class Note:
    start: float        # [s]
    end: float          # [s]
    pitch: float        # MIDIノート番号（WAVガイドの擬似ノートは小数を許す）

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class GuideData:
    notes: list[Note]
    source: str                       # "midi" | "wav"
    f0_times: np.ndarray | None = None  # WAVガイドのF0カーブ（curveモード用）
    f0_midi: np.ndarray | None = None   # MIDIノート番号スケール。無声はnan
    audio: np.ndarray | None = None     # ガイド音声（WAVガイド時。音響アライメント用）
    sr: int | None = None
    warnings: list[str] = field(default_factory=list)


def load_guide(path: str, cfg: Config) -> GuideData:
    gtype = cfg.guide_type
    if gtype == "auto":
        gtype = "midi" if path.lower().endswith((".mid", ".midi")) else "wav"
        cfg.guide_type = gtype
    if gtype == "midi":
        return load_midi_guide(path)
    return load_wav_guide(path, cfg)


# ---------------------------------------------------------------- MIDI

def load_midi_guide(path: str) -> GuideData:
    """pretty_midi はテンポマップを解決して秒単位のノート時刻を返す（可変テンポ対応）。"""
    import pretty_midi

    pm = pretty_midi.PrettyMIDI(path)
    notes: list[Note] = []
    for inst in pm.instruments:
        if inst.is_drum:
            continue
        for n in inst.notes:
            notes.append(Note(float(n.start), float(n.end), float(n.pitch)))
    notes.sort(key=lambda n: n.start)
    # 重なり（レガート打ち込み）は前のノートを後のノート開始で切る
    for prev, cur in zip(notes, notes[1:]):
        if prev.end > cur.start:
            prev.end = cur.start
    notes = [n for n in notes if n.duration > 0.01]
    return GuideData(notes=notes, source="midi")


# ---------------------------------------------------------------- WAVガイド

def load_wav_guide(path: str, cfg: Config) -> GuideData:
    """ガイドWAVをF0検出し、安定区間を擬似ノート化する。

    非処理ルールを適用: 無音・低信頼度区間はノート化しない。
    F0解析はファイル・パラメータ単位でディスクキャッシュする（ガイドは
    繰り返し使われるため）。
    """
    from .audio import load_wav
    from .detect import detect_pitch

    audio, sr = load_wav(path)
    track = _cached_detect(path, audio, sr, cfg)
    dt = cfg.hop / sr
    semis = track.semitones()

    notes: list[Note] = []
    warnings: list[str] = []
    min_note_frames = max(2, int(0.05 / dt))   # 50ms未満の断片はノート化しない
    gap_frames = max(1, int(cfg.min_gap_ms / 1000.0 / dt))

    # 有声区間を切り出し、区間内をピッチの跳躍（>0.8半音）でさらに分割
    runs = _voiced_runs(track.voiced, max_gap=max(1, gap_frames // 4))
    for s, e in runs:
        seg = semis[s:e]
        splits = [0]
        med_win = max(3, int(0.03 / dt))
        for i in range(med_win, len(seg) - med_win):
            a = np.nanmedian(seg[max(0, i - med_win):i])
            b = np.nanmedian(seg[i:i + med_win])
            if np.isfinite(a) and np.isfinite(b) and abs(b - a) > 0.8 and i - splits[-1] >= min_note_frames:
                splits.append(i)
        splits.append(len(seg))
        for a_i, b_i in zip(splits, splits[1:]):
            if b_i - a_i < min_note_frames:
                continue
            pitch = float(np.nanmedian(seg[a_i:b_i]))
            if not np.isfinite(pitch):
                continue
            notes.append(Note(track.times[s + a_i], track.times[min(s + b_i, len(track.times) - 1)], pitch))

    if not notes:
        warnings.append("WAVガイドからノートを抽出できませんでした")
    return GuideData(
        notes=notes, source="wav",
        f0_times=track.times, f0_midi=semis,
        audio=audio, sr=sr, warnings=warnings,
    )


def _cached_detect(path: str, audio: np.ndarray, sr: int, cfg: Config):
    """ガイドWAVのピッチ解析キャッシュ。ファイルの実体と解析パラメータが
    一致する場合のみ再利用する。"""
    import hashlib
    import os

    from .detect import PitchTrack, detect_pitch

    st = os.stat(path)
    key = f"{os.path.abspath(path)}|{st.st_mtime_ns}|{st.st_size}|" \
          f"{cfg.detector}|{cfg.hop}|{cfg.frame_length}|{cfg.fmin}|{cfg.fmax}|" \
          f"{cfg.pyin_resolution}|{cfg.min_voiced_conf}|{cfg.silence_thresh_db}"
    digest = hashlib.sha256(key.encode()).hexdigest()[:24]
    try:
        import platformdirs

        cache_dir = os.path.join(platformdirs.user_cache_dir("vocal-align-tune"), "f0")
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, f"{digest}.npz")
        if os.path.exists(cache_path):
            z = np.load(cache_path)
            print(f"ガイドF0キャッシュを再利用: {cache_path}")
            return PitchTrack(z["times"], z["f0"], z["voiced"], z["conf"])
    except OSError:
        cache_path = None

    track = detect_pitch(audio, sr, cfg)
    if cache_path:
        np.savez(cache_path, times=track.times, f0=track.f0,
                 voiced=track.voiced, conf=track.conf)
    return track


def _voiced_runs(voiced: np.ndarray, max_gap: int) -> list[tuple[int, int]]:
    """有声フレームの連続区間。max_gap以下の短い無声を埋めて結合する。"""
    runs: list[tuple[int, int]] = []
    start = None
    gap = 0
    for i, v in enumerate(voiced):
        if v:
            if start is None:
                start = i
            gap = 0
        elif start is not None:
            gap += 1
            if gap > max_gap:
                runs.append((start, i - gap + 1))
                start, gap = None, 0
    if start is not None:
        runs.append((start, len(voiced) - gap))
    return runs
