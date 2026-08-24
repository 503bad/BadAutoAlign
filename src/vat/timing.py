"""T2: フレーズ内のノート単位アライメント。

ボーカルを音節（モーラ）単位に分割し、ガイドノート列と順序保存の
離散アライメント（Needleman-Wunsch）で対応付ける。対応した音節頭
（母音開始点）をノート開始時刻へ動かす区分線形ワープマップを構築する。
子音は直前アンカーとの間で連続的に伸縮されるため、母音と切り離されない。

さらにアンカー先のピッチ照合で掛け違い（隣の音節への誤対応）を棄却する。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .align import AlignmentEntry, align_notes_to_syllables
from .config import Config
from .detect import PitchTrack
from .features import phonetic_features
from .guide import Note
from .pitch import NoteReport
from .syllables import detect_syllables


@dataclass
class WarpMap:
    """区分線形の時刻写像。src→dst いずれも秒（フレーズ先頭基準）で単調増加。"""
    src: np.ndarray
    dst: np.ndarray

    def is_identity(self, eps: float = 1e-4) -> bool:
        return bool(np.all(np.abs(self.src - self.dst) < eps))

    def dst_to_src(self, t_dst: np.ndarray) -> np.ndarray:
        return np.interp(t_dst, self.dst, self.src)


def warp_frame_index(track: PitchTrack, warp: WarpMap) -> np.ndarray:
    """ワープ後グリッドの各フレームが参照するワープ前フレーム番号。"""
    n = track.n_frames
    dt = float(track.times[1] - track.times[0])
    src_t = warp.dst_to_src(track.times)
    return np.clip(np.round(src_t / dt).astype(int), 0, n - 1)


def warp_track(track: PitchTrack, warp: WarpMap) -> PitchTrack:
    """ワープ後の時間軸に載せ替えた検出結果（F0値は不変、時刻のみ写像）。

    ピッチ同期エンジンはワープでF0を変えないため、ワープ後の再検出と
    等価な結果を写像だけで得られる（1パス統合用）。
    """
    if track.n_frames < 2:
        return track
    idx = warp_frame_index(track, warp)
    return PitchTrack(
        times=track.times.copy(), f0=track.f0[idx], voiced=track.voiced[idx],
        conf=track.conf[idx],
        f0_raw=None if track.f0_raw is None else track.f0_raw[idx],
        voiced_raw=None if track.voiced_raw is None else track.voiced_raw[idx],
    )


def detect_voicing_onsets(track: PitchTrack, cfg: Config) -> list[float]:
    """有声化開始点（母音頭）。評価・デバッグ用の簡易オンセット。"""
    dt = float(track.times[1] - track.times[0]) if track.n_frames > 1 else 0.01
    min_run = max(2, int(0.03 / dt))
    min_sep = 0.06
    onsets: list[float] = []
    v = track.voiced
    i = 0
    while i < len(v):
        if v[i] and (i == 0 or not v[i - 1]):
            run = 0
            j = i
            while j < len(v) and v[j]:
                run += 1
                j += 1
            if run >= min_run:
                t = track.times[i]
                if not onsets or t - onsets[-1] >= min_sep:
                    onsets.append(float(t))
            i = j
        else:
            i += 1
    return onsets


def build_warp_map(
    audio: np.ndarray,
    sr: int,
    track: PitchTrack,
    notes: list[Note],
    phrase_start: float,
    phrase_len_s: float,
    cfg: Config,
    reports: list[NoteReport],
    guide_ctx: dict | None = None,
    manual_anchors: list[dict] | None = None,
) -> tuple[WarpMap, list[AlignmentEntry], float, dict]:
    """ワープマップ・対応表・基準シフト[ms]・ラグプロファイルを返す。

    manual_anchors: [{"src_s": 絶対時刻, "dst_s": 絶対時刻, "note_index": 任意}]
    ユーザーが指定した確定アンカー。各種ガードを免除し最優先で適用する
    （近傍の自動アンカーは取り除く）。
    """
    syllables = detect_syllables(audio, sr, cfg, track)
    guide_feats = _guide_note_feats(notes, cfg, guide_ctx) if guide_ctx else None
    matched, confident, table = align_notes_to_syllables(
        notes, syllables, phrase_start, cfg, guide_feats=guide_feats)

    for entry in table:
        if entry.decision == "note_skipped":
            reports[entry.note_index].skip_reasons.append("no_syllable_match")

    # P-center（芯）: ガイド音声がある場合は両側対称に芯を検出してアンカーにする。
    # 片側だけ有声化開始点を使うと、しゃくり・ガナリのぶん系統的に後ろへずれる。
    pc = _prepare_pcenter(audio, sr, track, cfg, guide_ctx)

    # 第1パス: 全対応ノートのシフトを計測し、信頼できるものを候補にする
    max_shift = cfg.max_shift_ms / 1000.0
    matched_entries = [e for e in table if e.syl_index is not None]
    semis = track.semitones()
    cands: list[dict] = []
    for entry, ok in zip(matched_entries, confident):
        note = notes[entry.note_index]
        rep = reports[entry.note_index]
        syl = syllables[entry.syl_index]

        if pc is not None:
            src_t = _vocal_core(pc, syl)
            dst_t = _guide_core(pc, note) - phrase_start
        else:
            src_t = syl.onset
            dst_t = note.start - phrase_start

        shift = dst_t - src_t
        rep.timing_shift_ms = shift * 1000.0
        rep.anchor_src_s = phrase_start + src_t
        rep.anchor_dst_s = phrase_start + dst_t
        if not ok:
            rep.skip_reasons.append("timing_low_confidence")
            continue
        if not _anchor_pitch_ok(track, semis, src_t, note):
            rep.skip_reasons.append("anchor_pitch_mismatch")
            continue
        if abs(shift) > max_shift:
            rep.skip_reasons.append("shift_exceeds_max")
            cfg.warn(
                f"移動量 {shift * 1000:.0f}ms が max-shift を超過（t={phrase_start + src_t:.2f}s）、未補正"
            )
            continue
        cands.append({"src": src_t, "shift": shift,
                      "cost": entry.cost if entry.cost is not None else 1.0,
                      "rep": rep})

    # ラグカーブの推定。歌声は連続体であり、隣接ノートのズレが交互に
    # ±100ms振れることは物理的にないため、個別アンカーをそのまま適用すると
    # 計測ジッタがそのまま「ガタつき」になる。第一候補は中帯域エネルギー
    # 包絡の局所相互相関（密・頑健。VocALign系の手法）で、ノートアンカーは
    # 包絡が使えない場合のフォールバック。緩やかなテンポの揺れはカーブが
    # 追従し、ノート単位のジッタは吸収される。
    xc = _xcorr_lag_samples(audio, sr, syllables, phrase_start, cfg, guide_ctx, pc)
    if xc is not None and len(xc[0]) >= 4:
        samples = xc
    else:
        samples = (np.array([c["src"] for c in cands]),
                   np.array([c["shift"] for c in cands]),
                   np.array([1.2 - min(c["cost"], 1.0) for c in cands]))

    auto: list[tuple[float, float]] = []
    base_ms = 0.0
    grid_t = lag = None
    if len(samples[0]) >= 3 and syllables:
        grid_t, lag = _smooth_lag_curve(samples, syllables, cfg, pc, phrase_start)
        if lag is not None:
            a = max(0.03, syllables[0].onset - 0.02)
            b = min(phrase_len_s - 0.03, syllables[-1].end + 0.02)
            head = float(np.clip(lag[0], -0.66 * a, 0.66 * (phrase_len_s - b)))
            tail = float(np.clip(lag[-1], -0.66 * a, 0.66 * (phrase_len_s - b)))
            auto.append((a, a + head))
            for t, s in zip(grid_t, lag):
                if a < t < b:
                    auto.append((t, t + s))
            auto.append((b, b + tail))
            base_ms = float(np.median(lag)) * 1000.0
            lag_at = lambda t: float(np.interp(t, grid_t, lag))  # noqa: E731
            for c in cands:
                if c["rep"] is not None:
                    c["rep"].timing_applied = (
                        abs(lag_at(c["src"])) >= cfg.min_shift_ms / 1000.0)
    elif cands:
        # サンプルが少ないフレーズは従来どおり個別アンカー（ガード付き）
        kept = [c for c in cands
                if abs(c["shift"]) >= cfg.min_shift_ms / 1000.0]
        kept = _drop_inconsistent_anchors(kept, cfg)
        for c in kept:
            auto.append((c["src"], c["src"] + c["shift"] * cfg.timing_strength))
            if c["rep"] is not None:
                c["rep"].timing_applied = True

    # 手動アンカー（ガード免除・最優先）: 近傍の自動アンカーを退けて採用する
    for m in (manual_anchors or []):
        s = float(m["src_s"]) - phrase_start
        d = float(m["dst_s"]) - phrase_start
        if not (0.02 < s < phrase_len_s - 0.02) or d <= 0.02:
            continue
        clear = max(0.3, 3.0 * abs(d - s))
        auto = [(as_, ad_) for (as_, ad_) in auto if abs(as_ - s) > clear]
        auto.append((s, d))
        idx = m.get("note_index")
        for rep in reports:
            if idx is not None and rep.index == idx:
                rep.manual = True
                rep.timing_applied = True
                rep.anchor_src_s = phrase_start + s
                rep.timing_shift_ms = (d - s) * 1000.0

    anchors = [(0.0, 0.0)] + sorted(auto) + [(phrase_len_s, phrase_len_s)]
    warp = _sanitize_anchors(anchors, reports, cfg)
    coarse = warp  # レポート・ラグプロファイル用（再配分前の折れ点列）

    # 伸縮の再配分: アンカー対応は保ったまま、区間内のレート配分を
    # 無音 > 母音持続 >> 子音 > アタック の重みで解き直す（音質改善 Phase 1）
    if cfg.elastic_warp and not warp.is_identity():
        warp = _elastic_redistribute(warp, audio, sr, track, syllables, cfg)

    # ---- レポート用: 実際に適用された移動量と、まだ残っているズレ ----
    def applied_at(t_local: float) -> float:
        return float(np.interp(t_local, warp.src, warp.dst)) - t_local

    used_xcorr = xc is not None and len(xc[0]) >= 4
    ts_m, ss_m = samples[0], samples[1]
    for rep in reports:
        if rep.anchor_src_s is None:
            continue
        src_local = rep.anchor_src_s - phrase_start
        applied = applied_at(src_local)
        rep.timing_applied_ms = applied * 1000.0
        untrusted = ("timing_low_confidence" in rep.skip_reasons
                     or "anchor_pitch_mismatch" in rep.skip_reasons)
        measured = None
        if used_xcorr and len(ts_m) and np.min(np.abs(ts_m - src_local)) <= 0.3:
            measured = float(np.interp(src_local, ts_m, ss_m))
        elif rep.timing_shift_ms is not None and not untrusted:
            measured = rep.timing_shift_ms / 1000.0
        if measured is not None:
            rep.timing_residual_ms = (measured - applied) * 1000.0

    lag_profile = {
        "measured_t_s": [round(phrase_start + float(t), 3) for t in ts_m],
        "measured_ms": [round(float(s) * 1000.0, 1) for s in ss_m],
        "measured_from_xcorr": bool(used_xcorr),
        "applied_t_s": [round(phrase_start + float(t), 3) for t in coarse.src],
        "applied_ms": [round(float(d - s) * 1000.0, 1)
                       for s, d in zip(coarse.src, coarse.dst)],
    }
    return warp, table, base_ms, lag_profile


def _env_corr(a: np.ndarray, b: np.ndarray) -> float:
    """平均除去した正規化相関。"""
    a = a - a.mean()
    b = b - b.mean()
    denom = (float(np.sqrt(np.dot(a, a))) + 1e-9) * (float(np.sqrt(np.dot(b, b))) + 1e-9)
    return float(np.dot(a, b)) / denom


def _xcorr_lag_samples(audio: np.ndarray, sr: int, syllables, phrase_start: float,
                       cfg: Config, guide_ctx: dict | None, pc: dict | None,
                       grid_step: float = 0.2, half: float = 0.4,
                       min_corr: float = 0.6, min_range_db: float = 6.0,
                       ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """中帯域エネルギー包絡の局所相互相関によるラグサンプル。

    各時刻tで、ボーカル包絡の窓 [t-half, t+half] をガイド包絡に対して
    ±max_shift の範囲でスライドさせ、相関が最大になるズレを求める。
    ノート単位の芯検出よりはるかにノイズが少なく、レガートでも密に取れる。
    """
    if guide_ctx is None or pc is None or not syllables:
        return None
    dt_v = pc["v_dt"]
    dt_g = pc["g_dt"]
    if abs(dt_v - dt_g) > 1e-9:
        return None  # サンプルレート不一致（別途リサンプル済みのはず）
    dt = dt_v
    ve, ge = pc["v_env"], pc["g_env"]
    g_off = phrase_start - pc["g_seg_start"]  # vocalローカル→guideセグメントローカル
    max_lag = int(cfg.max_shift_ms / 1000.0 / dt)
    h = int(half / dt)

    ts, ss, ws = [], [], []
    for t in np.arange(syllables[0].onset, syllables[-1].end, grid_step):
        i = int(t / dt)
        if i - h < 0 or i + h > len(ve):
            continue
        a = ve[i - h: i + h]
        if a.max() - a.min() < min_range_db:
            continue  # 無音・平坦な窓はラグが定義できない
        j0 = int((t + g_off) / dt)
        best_r, best_l = 0.0, None
        for lag_i in range(-max_lag, max_lag + 1):
            s0 = j0 - h + lag_i
            if s0 < 0 or s0 + len(a) > len(ge):
                continue
            r = _env_corr(a, ge[s0: s0 + len(a)])
            if r > best_r:
                best_r, best_l = r, lag_i
        if best_l is None or best_r < min_corr:
            continue
        ts.append(float(t))
        ss.append(best_l * dt)   # 正 = ガイド側が後ろ = ボーカルを後ろへ動かす
        ws.append(best_r)
    if not ts:
        return None
    return np.array(ts), np.array(ss), np.array(ws)


def _smooth_lag_curve(samples: tuple[np.ndarray, np.ndarray, np.ndarray],
                      syllables, cfg: Config,
                      pc: dict | None = None, phrase_start: float = 0.0,
                      grid_step: float = 0.25, window: float = 0.8,
                      outlier_s: float = 0.08, max_slope: float = 0.25,
                      ) -> tuple[np.ndarray | None, np.ndarray | None]:
    """ラグサンプルから頑健な平滑ラグカーブを推定する。

    加重メディアン（窓内、信頼度重み）→ 外れ値除去 → 再推定 →
    傾き制限（局所テンポ変化 max_slope 以下）。カーブ全体が min_shift 未満
    ならワープ不要として None を返す。
    """
    ts, ss, ws = samples
    ss = ss * cfg.timing_strength
    # 孤立スパイク除去: 両隣と大きく食い違うサンプルは計測ノイズとみなす
    if len(ss) >= 6:
        keep0 = np.ones(len(ss), dtype=bool)
        for i in range(1, len(ss) - 1):
            if (abs(ss[i] - ss[i - 1]) > 0.05 and abs(ss[i] - ss[i + 1]) > 0.05):
                keep0[i] = False
        ts, ss, ws = ts[keep0], ss[keep0], ws[keep0]

    t0, t1 = syllables[0].onset, syllables[-1].end
    grid = np.arange(t0, t1 + grid_step / 2, grid_step)

    def fit(ts_, ss_, ws_):
        out = np.full(len(grid), np.nan)
        for i, t in enumerate(grid):
            d = np.abs(ts_ - t)
            m = d < window
            if not m.any():
                continue
            w = ws_[m] * (1.0 - (d[m] / window) ** 2)
            out[i] = _weighted_median(ss_[m], w)
        if np.isnan(out).all():
            return None
        valid = ~np.isnan(out)
        return np.interp(grid, grid[valid], out[valid])

    lag = fit(ts, ss, ws)
    if lag is None:
        return None, None
    # 外れ値（カーブから大きく離れたサンプル）を捨てて再推定
    resid = ss - np.interp(ts, grid, lag)
    keep = np.abs(resid) <= outlier_s
    if keep.sum() >= 3 and not keep.all():
        lag = fit(ts[keep], ss[keep], ws[keep])
        if lag is None:
            return None, None

    lag = np.clip(lag, -cfg.max_shift_ms / 1000.0, cfg.max_shift_ms / 1000.0)
    # 傾き制限（隣接グリッド間のラグ変化 = 局所テンポ変化を抑える）
    for i in range(1, len(lag)):
        step = max_slope * grid_step
        lag[i] = float(np.clip(lag[i], lag[i - 1] - step, lag[i - 1] + step))

    # 不感帯: 十分合っている区間は恒等写像に落とす（合っている所は触らない）。
    dead = cfg.min_shift_ms / 1000.0 * 0.8
    lag = np.where(np.abs(lag) < dead, 0.0, lag)

    # 閉ループ検証: 各点で「lagだけ動かすと包絡相関が実際に改善する」ことを
    # 確認し、改善しない点はゼロに落とす。大きなミスマッチ（ガード対象）の
    # 近傍でカーブが汚染され、合っている区間を引きずるのを防ぐ最終防衛線。
    if pc is not None:
        lag = _verify_lag_curve(grid, lag, pc, phrase_start)

    # 遷移をなだらかに繋ぐ
    from scipy.ndimage import gaussian_filter1d

    lag = gaussian_filter1d(lag, sigma=max(1e-3, 0.15 / grid_step), mode="nearest")

    if np.max(np.abs(lag)) < cfg.min_shift_ms / 1000.0:
        return None, None  # 全体が十分合っている → 触らない
    return grid, lag


def _verify_lag_curve(grid: np.ndarray, lag: np.ndarray, pc: dict,
                      phrase_start: float, half: float = 0.4,
                      margin: float = 0.03) -> np.ndarray:
    """カーブの各点について、適用後の包絡相関が適用前より margin 以上
    良くなることを確認する。改善しない移動は行わない。"""
    ve, ge = pc["v_env"], pc["g_env"]
    dt = pc["v_dt"]
    g_off = phrase_start - pc["g_seg_start"]
    h = int(half / dt)
    out = lag.copy()
    for i, t in enumerate(grid):
        if out[i] == 0.0:
            continue
        iv = int(t / dt)
        if iv - h < 0 or iv + h > len(ve):
            out[i] = 0.0
            continue
        a = ve[iv - h: iv + h]
        j0 = int((t + g_off) / dt)
        li = int(round(out[i] / dt))
        s0, sL = j0 - h, j0 - h + li
        if s0 < 0 or s0 + len(a) > len(ge) or sL < 0 or sL + len(a) > len(ge):
            out[i] = 0.0
            continue
        r0 = _env_corr(a, ge[s0: s0 + len(a)])
        rL = _env_corr(a, ge[sL: sL + len(a)])
        if rL < r0 + margin:
            out[i] = 0.0
    return out


def _weighted_median(x: np.ndarray, w: np.ndarray) -> float:
    order = np.argsort(x)
    cw = np.cumsum(w[order])
    k = int(np.searchsorted(cw, cw[-1] / 2.0))
    return float(x[order][min(k, len(x) - 1)])


def _drop_inconsistent_anchors(cands: list[dict], cfg: Config,
                               max_jump: float = 0.12,
                               window: float = 0.8) -> list[dict]:
    """隣接アンカー間の一貫性ガード（掛け違い対策の最終防衛線）。

    近接する2アンカーのシフト量が大きく矛盾する場合（例: 0.4s間隔で
    +90ms→-75ms）、少なくとも一方は誤対応なので、アライメントコストが
    高い方を棄却する。安定するまで繰り返す。
    """
    cands = sorted(cands, key=lambda c: c["src"])
    for _ in range(len(cands)):
        worst = None
        for a, b in zip(cands, cands[1:]):
            if b["src"] - a["src"] < window and abs(b["shift"] - a["shift"]) > max_jump:
                drop = a if a["cost"] >= b["cost"] else b
                if worst is None or drop["cost"] > worst["cost"]:
                    worst = drop
        if worst is None:
            break
        cands.remove(worst)
        if worst["rep"] is not None:
            worst["rep"].skip_reasons.append("timing_inconsistent")
    return cands


def _prepare_pcenter(audio: np.ndarray, sr: int, track: PitchTrack,
                     cfg: Config, guide_ctx: dict | None) -> dict | None:
    """芯検出用の包絡・F0コンテキスト（ガイド音声がある場合のみ）。"""
    if guide_ctx is None:
        return None
    from .pcenter import midband_envelope

    g_semis = np.nan_to_num(guide_ctx["guide_semis"], nan=0.0)
    return {
        "v_env": midband_envelope(audio, sr, cfg),
        "v_semis": np.nan_to_num(track.semitones(), nan=0.0),
        "v_voiced": track.voiced,
        "v_dt": cfg.hop / sr,
        "g_env": midband_envelope(guide_ctx["guide_audio"],
                                  guide_ctx["guide_sr"], cfg),
        "g_semis": g_semis,
        "g_voiced": np.isfinite(guide_ctx["guide_semis"]),
        "g_dt": cfg.hop / guide_ctx["guide_sr"],
        "g_seg_start": guide_ctx["guide_seg_start"],
    }


def _vocal_core(pc: dict, syl) -> float:
    from .pcenter import core_time

    return core_time(pc["v_env"], pc["v_semis"], pc["v_voiced"],
                     syl.onset, syl.end, pc["v_dt"])


def _guide_core(pc: dict, note: Note) -> float:
    """ガイドノートの芯（絶対時刻）。"""
    from .pcenter import core_time

    t0 = note.start - pc["g_seg_start"]
    t1 = note.end - pc["g_seg_start"]
    core = core_time(pc["g_env"], pc["g_semis"], pc["g_voiced"],
                     t0, t1, pc["g_dt"])
    return core + pc["g_seg_start"]


def _anchor_pitch_ok(track: PitchTrack, semis: np.ndarray, src_t: float,
                     note: Note) -> bool:
    """アンカー直後のボーカルのピッチクラスがノート音高と一致するか
    （掛け違いの棄却。オクターブ差は許容）。検証不能なら触らない。"""
    win = min(0.15, max(note.duration, 0.05))
    m = (track.times >= src_t) & (track.times < src_t + win) & track.voiced
    if not m.any():
        return False
    med = float(np.nanmedian(semis[m]))
    fold = abs((med - note.pitch + 6.0) % 12.0 - 6.0)
    return fold <= 1.5


def _guide_note_feats(notes: list[Note], cfg: Config,
                      guide_ctx: dict) -> list[np.ndarray] | None:
    """ガイド音声からノートごとの母音特徴重心を計算する。"""
    g_audio = guide_ctx["guide_audio"]
    g_sr = guide_ctx["guide_sr"]
    g_semis = guide_ctx["guide_semis"]
    g_seg_start = guide_ctx["guide_seg_start"]
    g_voiced = np.isfinite(g_semis)
    feats = phonetic_features(g_audio, g_sr, cfg, active=g_voiced)
    n = min(len(feats), len(g_semis))
    dt_g = cfg.hop / g_sr

    out: list[np.ndarray] = []
    for note in notes:
        s = int(round((note.start - g_seg_start) / dt_g))
        e = int(round((note.end - g_seg_start) / dt_g))
        s, e = max(0, min(s, n - 1)), max(1, min(e, n))
        if e <= s:
            out.append(feats[:n].mean(axis=0))
            continue
        m = g_voiced[s:e]
        seg = feats[s:e][m] if m.any() else feats[s:e]
        out.append(seg.mean(axis=0))
    return out


def _elastic_redistribute(warp: WarpMap, audio: np.ndarray, sr: int,
                          track: PitchTrack, syllables, cfg: Config,
                          grid_step: float = 0.01) -> WarpMap:
    """区分線形ワープのレート配分を知覚重みで解き直す（05_改修方針 §1.1）。

    アンカー（折れ点）の対応は厳密に保ったまま、各アンカー区間内の伸縮を
    無音(重み1.0) > 母音持続(0.5) >> 子音(0.05) > アタック(0.02) の順で
    吸収させる。子音・アタック上のレートは 1.0 近傍に保たれ、
    トランジェント滲みとレート急変によるざらつき（バリバリ音）を避ける。
    出力は10msグリッドの細かい折れ点列で、レート遷移は重みの平滑化により
    なだらかになる。
    """
    total = float(warp.src[-1] - warp.src[0])
    if total < 4 * grid_step:
        return warp
    # グリッド: 一様格子 ∪ アンカー位置（区間積分をアンカーで正確に区切る）
    t = np.unique(np.concatenate(
        [np.arange(warp.src[0], warp.src[-1], grid_step), warp.src]))
    dts = np.diff(t)
    ok = dts > 1e-6
    t = np.concatenate([t[:1], t[1:][ok]])
    dts = np.diff(t)
    centers = (t[:-1] + t[1:]) / 2

    w, lo, hi = _elastic_profile(centers, audio, sr, track, syllables, cfg,
                                 grid_step)

    rate = np.ones(len(dts))
    for k in range(len(warp.src) - 1):
        s0, s1 = float(warp.src[k]), float(warp.src[k + 1])
        d0, d1 = float(warp.dst[k]), float(warp.dst[k + 1])
        m = (centers > s0) & (centers < s1)
        if not m.any():
            continue
        rate[m] = _solve_rates(dts[m], w[m], lo[m], hi[m], d1 - d0)

    dst = float(warp.dst[0]) + np.concatenate([[0.0], np.cumsum(rate * dts)])
    # 数値誤差の吸収: アンカー位置の到達値を厳密に合わせる（区間内で線形補正）
    anchor_dst = np.interp(warp.src, t, dst)
    dst += np.interp(t, warp.src, warp.dst - anchor_dst)
    return WarpMap(src=t, dst=dst)


def _elastic_profile(centers: np.ndarray, audio: np.ndarray, sr: int,
                     track: PitchTrack, syllables, cfg: Config,
                     grid_step: float,
                     ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """グリッドセルごとの伸縮許容重みとレート上下限。"""
    # 短窓RMS（±10ms）による無音判定
    e = np.concatenate([[0.0], np.cumsum(audio.astype(np.float64) ** 2)])
    half = max(1, int(0.01 * sr))
    i = np.clip((centers * sr).astype(int), 0, len(audio))
    a = np.clip(i - half, 0, len(audio))
    b = np.clip(i + half, 0, len(audio))
    rms = np.sqrt((e[b] - e[a]) / np.maximum(b - a, 1))
    silent = 20.0 * np.log10(rms + 1e-12) < cfg.silence_thresh_db

    voiced = np.interp(centers, track.times,
                       track.voiced.astype(float)) > 0.5

    w = np.where(silent, 1.0, np.where(voiced, 0.5, 0.05))
    lo = np.where(silent, 0.25, np.where(voiced, 1.0 / 1.5, 0.9))
    hi = np.where(silent, 4.0, np.where(voiced, 1.5, 1.1))

    # アタック保護: 音節頭 −20ms〜+40ms はほぼ伸縮させない
    for syl in syllables:
        m = (centers >= syl.onset - 0.02) & (centers <= syl.onset + 0.04)
        w[m] = np.minimum(w[m], 0.02)
        lo[m] = np.maximum(lo[m], 0.95)
        hi[m] = np.minimum(hi[m], 1.05)

    # 重みを平滑化してレート遷移をなだらかに（折れ点由来のざらつき防止）
    from scipy.ndimage import gaussian_filter1d

    w = gaussian_filter1d(w, sigma=max(1e-3, 0.03 / grid_step), mode="nearest")
    return np.maximum(w, 1e-3), lo, hi


def _solve_rates(dts: np.ndarray, w: np.ndarray, lo: np.ndarray,
                 hi: np.ndarray, target_dur: float) -> np.ndarray:
    """Σ rate·dt = target_dur を満たすレート列を、重みに比例した配分と
    上下限キャップの反復で解く。不足分が残る場合は一様配分で吸収する
    （上下限より到達精度を優先。旧・線形配分と同等以上の品質は保たれる）。"""
    r = np.ones(len(dts))
    free = np.ones(len(dts), dtype=bool)
    for _ in range(8):
        deficit = target_dur - float(np.sum(r * dts))
        if abs(deficit) < 1e-7:
            break
        denom = float(np.sum(w[free] * dts[free]))
        if denom <= 0:
            break
        r[free] += deficit * w[free] / denom
        clipped = np.clip(r, lo, hi)
        free = free & (np.abs(r - clipped) < 1e-12)
        r = clipped
    deficit = target_dur - float(np.sum(r * dts))
    if abs(deficit) > 1e-7:
        r += deficit / float(np.sum(dts))
    return np.maximum(r, 0.05)


def _sanitize_anchors(anchors: list[tuple[float, float]],
                      reports: list[NoteReport], cfg: Config) -> WarpMap:
    """単調増加・極端な伸縮率（<0.25 / >4.0）を除去。"""
    anchors = sorted(set(anchors))
    kept: list[tuple[float, float]] = [anchors[0]]
    for a in anchors[1:-1]:
        p = kept[-1]
        ds, dd = a[0] - p[0], a[1] - p[1]
        if ds <= 0.005 or dd <= 0.005:
            continue
        if not (0.25 <= dd / ds <= 4.0):
            cfg.warn(f"伸縮率が異常なアンカー（{dd / ds:.2f}倍）を除外しました")
            continue
        kept.append(a)
    last = anchors[-1]
    # 終端直前のアンカーが終端固定と矛盾する場合は取り除く
    while len(kept) > 1:
        p = kept[-1]
        ds, dd = last[0] - p[0], last[1] - p[1]
        if ds > 0.005 and dd > 0.005 and 0.25 <= dd / ds <= 4.0:
            break
        kept.pop()
    kept.append(last)
    src = np.array([a[0] for a in kept])
    dst = np.array([a[1] for a in kept])
    return WarpMap(src=src, dst=dst)
