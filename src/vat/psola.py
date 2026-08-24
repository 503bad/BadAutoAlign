"""ピッチ同期グレイン再合成（TD-PSOLA / Melodyne "Local Sound Synthesis" 系）による
1パスのタイムワープ＋ピッチシフト。

既存の検出結果（PitchTrack）・ワープマップ・補正カーブをそのまま入力とし、
「どれだけ動かすか」は変えずに「どう音を作るか」だけを置き換える。
位相ボコーダ（Signalsmith）と違い

- 母音持続部を伸縮しても倍音の位相関係が崩れない（ビリビリ／にじみが出ない）
- 声道包絡（フォルマント）はグレイン波形ごと保たれる（包絡推定が不要）
- 無変更区間は窓和正規化により入力とサンプル一致する（恒等再構成）

処理の流れ:
  1. 合成用の有声判定と周期の精密化（detector の生F0 ＋ 自己相関。
     オクターブ誤りは P/2・P・2P の相関比較で正す）
  2. 有声区間ごとに F0 軌跡から位相を積分し、基本波成分の複素包絡で位相を
     補正した「周期位相」Φ(t) を得る。エネルギー重心の位相 θ* を 1 周期の
     基準点とし、Φ = 2πk + θ* をピッチマークにする（ジッタの無い等間隔列）
  3. 出力時刻を進めながら、対応する入力時刻を挟む 2 つのピッチマークの
     2 周期 Hann グレインを補間（周期波形のモーフ）し、目標周期
     （元周期 ÷ ピッチ係数）間隔で重畳加算する。同一グレインの反復に
     よるフラッタを避ける。無声・無音区間は 10ms 固定グレインの OLA
     （ピッチ変更なし、雑音フレームの引き伸ばし時のみ読み出し位置をランダム化して反復による櫛形歪みを避ける）
  4. 窓和で正規化する
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import butter, sosfiltfilt

from .detect import PitchTrack

UNVOICED_HALF_S = 0.005     # 無声グレインの半長（窓長 10ms、ホップ 5ms）
MIN_RUN_FRAMES = 3          # これより短い有声区間は無声扱い
BRIDGE_FRAMES = 4           # 有声区間内のこの長さ以下の無声穴は補間で埋める
PERIODICITY_VOICED = 0.45   # 検出器が有声としたフレームの周期性しきい値
PERIODICITY_RECLAIM = 0.70  # 検出器が無声としたフレームを有声に拾うしきい値


@dataclass
class VoicedRun:
    marks: np.ndarray       # ピッチマーク（サンプル、昇順）
    spacing: np.ndarray     # 各マークから次のマークまでの周期（サンプル）
    lo: float               # この区間が有声とみなされる入力範囲 [lo, hi)
    hi: float


# ---------------------------------------------------------------- 有声判定・周期精密化

def _voiced_runs(voiced: np.ndarray, bridge: int, min_len: int) -> list[tuple[int, int]]:
    v = voiced.copy()
    n = len(v)
    i = 0
    while i < n:                      # 短い穴を埋める
        if not v[i]:
            j = i
            while j < n and not v[j]:
                j += 1
            if 0 < i and j < n and (j - i) <= bridge:
                v[i:j] = True
            i = j
        else:
            i += 1
    runs = []
    i = 0
    while i < n:
        if v[i]:
            j = i
            while j < n and v[j]:
                j += 1
            if j - i >= min_len:
                runs.append((i, j))
            i = j
        else:
            i += 1
    return runs


def _xcorr_curve(x: np.ndarray, center: int, sr: int, lmax: int, W: int) -> np.ndarray | None:
    """フレーム中心 center の正規化相互相関 r(L), L=0..lmax（FFT 実装）。"""
    n = len(x)
    a0 = center - W // 2
    if a0 < 0 or a0 + W + lmax > n:
        return None
    a = x[a0: a0 + W]
    b = x[a0: a0 + W + lmax]
    ea = float(np.dot(a, a))
    if ea <= 1e-10:
        return None
    nfft = 1 << int(np.ceil(np.log2(W + lmax)))
    fa = np.fft.rfft(a, nfft)
    fb = np.fft.rfft(b, nfft)
    xc = np.fft.irfft(fb * np.conj(fa), nfft)[: lmax + 1]  # Σ a(t) b(t+L)
    cs = np.concatenate([[0.0], np.cumsum(b * b)])
    eb = cs[W: W + lmax + 1] - cs[: lmax + 1]
    return xc / np.sqrt(ea * np.maximum(eb, 1e-12))


def _periodicity(x: np.ndarray, center: int, sr: int, cands: list[float]) -> list[float]:
    """候補周期ごとの最大正規化相関（±8% 探索）。"""
    lmax = int(np.ceil(max(cands) * 1.08)) + 1
    W = int(max(2.2 * max(cands), 0.02 * sr))
    r = _xcorr_curve(x, center, sr, lmax, W)
    if r is None:
        return [0.0] * len(cands)
    out = []
    for P in cands:
        l0 = max(1, int(np.floor(P * 0.92)))
        l1 = min(lmax, int(np.ceil(P * 1.08)))
        out.append(float(r[l0: l1 + 1].max()) if l1 >= l0 else 0.0)
    return out


def _generic_period(x: np.ndarray, center: int, sr: int,
                    fmin: float = 60.0, fmax: float = 1000.0) -> tuple[int, float]:
    """検出器の F0 に頼らない周期探索。相関曲線の局所ピークのうち、最大値近傍
    （−0.03）で最も短いラグを返す（ラグ/2 より明確に高いこと＝ピーク条件）。
    返り値 (ラグ[サンプル], 相関)。見つからなければ (0, rmax)。"""
    lmin = max(2, int(sr / fmax))
    lmax = int(sr / fmin) + 1
    W = int(max(2.2 * lmax, 0.02 * sr))
    r = _xcorr_curve(x, center, sr, lmax, W)
    if r is None:
        return 0, 0.0
    seg = r[lmin: lmax]
    rmax = float(seg.max())
    if rmax <= 0:
        return 0, rmax
    peaks = np.where((seg[1:-1] > seg[:-2]) & (seg[1:-1] >= seg[2:]))[0] + 1 + lmin
    for L in peaks:
        if r[L] < rmax - 0.03:
            continue
        h = L // 2
        if h >= lmin and r[h] > r[L] - 0.03:
            continue
        return int(L), float(r[L])
    return 0, rmax


def _voicing_and_ratio(audio: np.ndarray, sr: int, track: PitchTrack, hop: int,
                       silence_db: float = -50.0
                       ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """フレームごとの (合成用有声フラグ, F0, 周期補正比 ∈ {0.5,1,2}, 周期性, 汎用探索フラグ)。

    検出器の生 F0 を種に、自己相関で (a) 周期性の弱いフレームを落とし、
    (b) 検出器が取りこぼした周期的フレームを拾い、(c) P/2・P・2P の
    相関比較でオクターブ誤りを検出する。候補周期 L は「相関が最大近傍」かつ
    「L/2 の相関より明確に高い（相関のピークである）」ことを要求し、
    色付き雑音（摩擦音・ブレス）が短ラグの高相関で有声化するのを防ぐ。
    検出器自身の周期 P には強い優先を与える（ガナリ＝周期倍化の声で
    r(2P) がわずかに勝っても 2P に飛ばない。飛ぶとグレインが 4 周期になり
    シフト時に低調波が出る）。
    生 F0 が無い／誤っている（音高の急変・語尾の減衰など）フレームは、
    F0 に依存しない汎用周期探索で拾う（周期的な声を無声 OLA で処理すると
    雑音化するため）。補正量の判断（pitch.py）には影響しない。
    """
    x = audio.astype(np.float64)
    n = track.n_frames
    f0_raw = track.f0_raw if track.f0_raw is not None else track.f0
    v_raw = track.voiced_raw if track.voiced_raw is not None else track.voiced
    voiced = np.zeros(n, dtype=bool)
    generic = np.zeros(n, dtype=bool)
    f0_out = np.zeros(n)
    ratio = np.ones(n)
    per = np.zeros(n)
    # 短窓 RMS ゲート（無音部のハム等を有声化しない）
    e = np.concatenate([[0.0], np.cumsum(x * x)])
    half = int(0.01 * sr)
    tol = (0.03, 0.15, 0.0)  # P/2, P, 2P の許容（rmax からの差）
    for i in range(n):
        c = i * hop
        a, b = max(0, c - half), min(len(x), c + half)
        if b <= a or 10 * np.log10((e[b] - e[a]) / (b - a) + 1e-12) < silence_db:
            continue
        f = float(f0_raw[i])
        accepted = False
        if f > 0:
            P = sr / f
            r = _periodicity(x, c, sr, [P / 4, P / 2, P, 2 * P])
            rmax = max(r[1:])
            per[i] = rmax
            thr = PERIODICITY_VOICED if v_raw[i] else PERIODICITY_RECLAIM
            if rmax >= thr:
                for j in (1, 2, 3):
                    if r[j] >= rmax - tol[j - 1] and r[j] > r[j - 1] + 0.03:
                        ratio[i] = (0.5, 1.0, 2.0)[j - 1]
                        voiced[i] = True
                        f0_out[i] = f
                        accepted = True
                        break
        if not accepted:
            L, rb = _generic_period(x, c, sr)
            per[i] = max(per[i], rb)
            if L > 0 and rb >= PERIODICITY_RECLAIM:
                voiced[i] = True
                generic[i] = True
                f0_out[i] = sr / L
    return voiced, f0_out, ratio, per, generic


def synthesis_voicing(audio: np.ndarray, sr: int, track: PitchTrack, hop: int,
                      silence_db: float = -50.0) -> tuple[np.ndarray, np.ndarray]:
    """合成用の有声フラグと F0（フレームごと）。オクターブ判定は
    局所多数決（median 7）で安定化した値を返す。"""
    voiced, f0, ratio, _per, generic = _voicing_and_ratio(audio, sr, track, hop, silence_db)
    if voiced.any():
        lr_f = median_filter(np.log2(ratio), size=7, mode="nearest")
        f0 = np.where(voiced & ~generic, f0 / np.exp2(np.round(lr_f)), f0)
    return voiced, f0


# ---------------------------------------------------------------- ピッチマーク

def estimate_pitch_marks(audio: np.ndarray, sr: int, track: PitchTrack,
                         hop: int) -> list[VoicedRun]:
    """有声区間ごとのピッチマーク列を推定する。"""
    return analyze(audio, sr, track, hop)[0]


def analyze(audio: np.ndarray, sr: int, track: PitchTrack, hop: int
            ) -> tuple[list[VoicedRun], np.ndarray]:
    """(有声区間ごとのピッチマーク列, フレームごとの雑音フラグ) を返す。
    雑音フラグは「無声かつ周期性が低い」フレーム＝伸縮時に読み出し位置の
    ランダム化を許すフレーム。"""
    x = audio.astype(np.float64)
    n = len(x)
    voiced, f0_syn, ratio, per, generic = _voicing_and_ratio(audio, sr, track, hop)
    noise = ~voiced & (per < 0.5)
    runs_out: list[VoicedRun] = []
    frame_t = track.times
    for i0, i1 in _voiced_runs(voiced, BRIDGE_FRAMES, MIN_RUN_FRAMES):
        fv = np.arange(i0, i1)[voiced[i0:i1] & (f0_syn[i0:i1] > 0)]
        if len(fv) < 2:
            continue
        # オクターブ補正比は有声区間ごとに一定（区間途中の切替はグレイン長の
        # 不連続＝シフト時の低調波・ざらつきになる）
        fr = fv[~generic[fv]]
        if len(fr):
            run_ratio = float(np.exp2(np.round(np.median(np.log2(ratio[fr])))))
            f0_syn = f0_syn.copy()
            f0_syn[fr] = f0_syn[fr] / run_ratio
        s0 = max(0, int(round((i0 - 0.5) * hop)))
        s1 = min(n, int(round((i1 - 0.5) * hop)) + 1)
        if s1 - s0 < 2 * hop:
            continue
        t = np.arange(s0, s1) / sr
        # 対数F0をフレーム間で線形補間（区間内の無声穴も補間）
        f0 = np.exp(np.interp(t, frame_t[fv], np.log(f0_syn[fv])))
        phi = 2.0 * np.pi * np.cumsum(f0) / sr
        seg = x[s0:s1]
        # 基本波成分の複素包絡（F0で復調 → ゼロ位相ローパス）で位相を補正
        z = seg * np.exp(-1j * phi)
        fc = float(np.clip(0.45 * f0.min(), 15.0, 0.45 * sr / 2))
        sos = butter(2, fc, fs=sr, output="sos")
        try:
            zl = sosfiltfilt(sos, z.real) + 1j * sosfiltfilt(sos, z.imag)
            psi = np.unwrap(np.angle(zl + 1e-12))
            Phi = phi + psi
            if np.any(np.diff(Phi) <= 0):
                Phi = phi
        except ValueError:      # 区間が短すぎてフィルタ不能
            Phi = phi
        # 1周期内のエネルギー重心位相を基準点にする（声門パルス位置に相当）
        e = seg * seg
        if e.sum() <= 0:
            continue
        theta = float(np.angle(np.sum(e * np.exp(1j * Phi))))
        k0 = int(np.ceil((Phi[0] - theta) / (2 * np.pi)))
        k1 = int(np.floor((Phi[-1] - theta) / (2 * np.pi)))
        if k1 - k0 < 2:
            continue
        targets = 2 * np.pi * np.arange(k0, k1 + 1) + theta
        marks = np.interp(targets, Phi, np.arange(s0, s1, dtype=np.float64))
        marks = np.unique(np.round(marks).astype(np.int64))
        if len(marks) < 3:
            continue
        spacing = np.diff(marks).astype(np.float64)
        spacing = np.concatenate([spacing, spacing[-1:]])
        runs_out.append(VoicedRun(
            marks=marks, spacing=spacing,
            lo=float(marks[0] - 0.5 * spacing[0]),
            hi=float(marks[-1] + 0.5 * spacing[-1]),
        ))
    return runs_out, noise


# ---------------------------------------------------------------- 合成

def _hann(n: int) -> np.ndarray:
    """周期 Hann（長さ n、ホップ n/2 で窓和が 1）。"""
    k = np.arange(n)
    return 0.5 - 0.5 * np.cos(2.0 * np.pi * k / n)


def render(audio: np.ndarray, sr: int, runs: list[VoicedRun],
           src_samples: np.ndarray, dst_samples: np.ndarray,
           factor_at=None, n_out: int | None = None,
           morph: bool = True, trace: list | None = None,
           noise_frames: np.ndarray | None = None, hop: int = 512) -> np.ndarray:
    """PSOLA 再合成。

    src_samples/dst_samples: 区分線形ワープ（サンプル単位、単調増加）。
    factor_at(t_out_sample) -> ピッチ倍率（出力時間軸）。None で 1.0。
    morph: 隣接周期グレインの補間（True 推奨。False で最近傍グレイン）。
    noise_frames: フレーム（hop 間隔）ごとの雑音フラグ。True のフレームだけ
    伸縮時の読み出し位置ランダム化を行う（周期的なのに無声扱いになった
    フレームを雑音化しない）。None なら無声フレームすべてで行う。
    """
    x = audio.astype(np.float64)
    if n_out is None:
        n_out = int(round(dst_samples[-1]))
    max_half = int(sr / 40) + 8  # 最長グレイン半長（F0 ≥ 40Hz 相当）
    pad = max_half + 4
    xp = np.pad(x, (pad, pad))
    out = np.zeros(n_out + 2 * pad)
    wsum = np.zeros(n_out + 2 * pad)
    rng = np.random.default_rng(12345)

    run_lo = np.array([r.lo for r in runs]) if runs else np.zeros(0)
    run_hi = np.array([r.hi for r in runs]) if runs else np.zeros(0)
    uv_half = max(32, int(round(UNVOICED_HALF_S * sr)))
    jitter_max = uv_half // 2
    win_cache: dict[int, np.ndarray] = {}

    def window(half: int) -> np.ndarray:
        w = win_cache.get(half)
        if w is None:
            w = _hann(2 * half)
            win_cache[half] = w
        return w

    def place(center_out: float, grain: np.ndarray, w: np.ndarray, gain: float) -> None:
        half = len(w) // 2
        o = int(round(center_out)) - half + pad
        if o < 0:
            grain, w = grain[-o:], w[-o:]
            o = 0
        end = min(o + len(grain), len(out))
        if end <= o:
            return
        out[o:end] += (grain * w * gain)[: end - o]
        wsum[o:end] += w[: end - o]

    def grain_at(mark: int, half: int) -> np.ndarray:
        a = mark - half + pad
        return xp[a: a + 2 * half]

    seg_rate = np.diff(dst_samples) / np.maximum(np.diff(src_samples), 1e-9)
    t_out = 0.0
    in_voiced = False
    while t_out < n_out:
        t_src = float(np.interp(t_out, dst_samples, src_samples))
        r = -1
        if len(runs):
            k = int(np.searchsorted(run_lo, t_src, side="right")) - 1
            if k >= 0 and t_src < run_hi[k]:
                r = k
        if r >= 0:
            run = runs[r]
            marks = run.marks
            j = int(np.searchsorted(marks, t_src)) - 1  # marks[j] <= t_src < marks[j+1]
            j = min(max(j, 0), len(marks) - 1)
            if not in_voiced:
                # 区間の入口: 出力位置を最寄りマークの写像先に合わせる（恒等時に厳密再構成）
                jn = j if (j + 1 >= len(marks) or t_src - marks[j] <= marks[j + 1] - t_src) else j + 1
                t_m = float(np.interp(marks[jn], src_samples, dst_samples))
                if abs(t_m - t_out) < run.spacing[jn]:
                    t_out = max(t_m, 0.0)
                    t_src = float(np.interp(t_out, dst_samples, src_samples))
                    j = min(max(int(np.searchsorted(marks, t_src)) - 1, 0), len(marks) - 1)
                in_voiced = True
            f = 1.0 if factor_at is None else float(factor_at(t_out))
            f = min(max(f, 0.25), 4.0)
            if morph and j + 1 < len(marks):
                m0, m1 = int(marks[j]), int(marks[j + 1])
                alpha = (t_src - m0) / max(m1 - m0, 1)
                alpha = min(max(alpha, 0.0), 1.0)
                P = (1 - alpha) * run.spacing[j] + alpha * run.spacing[j + 1]
                half = int(round(min(P, max_half)))
                w = window(half)
                if alpha < 1e-3:
                    g = grain_at(m0, half)
                elif alpha > 1 - 1e-3:
                    g = grain_at(m1, half)
                else:
                    g = (1 - alpha) * grain_at(m0, half) + alpha * grain_at(m1, half)
            else:
                jn = j if (j + 1 >= len(marks) or t_src - marks[j] <= marks[j + 1] - t_src) else j + 1
                P = float(run.spacing[jn])
                half = int(round(min(P, max_half)))
                w = window(half)
                g = grain_at(int(marks[jn]), half)
            place(t_out, g, w, np.sqrt(f))
            if trace is not None:
                trace.append((t_out, t_src, "V", int(marks[j]), half, round(P, 1)))
            t_out += P / f
        else:
            in_voiced = False
            c = t_src
            # 伸縮中の無声部は読み出し位置を散らし、規則的反復による櫛形歪みを避ける
            kk = min(max(int(np.searchsorted(dst_samples, t_out, side="right")) - 1, 0),
                     len(seg_rate) - 1)
            noisy = True
            if noise_frames is not None:
                fi = min(max(int(t_src / hop + 0.5), 0), len(noise_frames) - 1)
                noisy = bool(noise_frames[fi])
            # ランダム化は引き伸ばし（同一内容の反復が起きる側）だけ。圧縮では不要
            if noisy and seg_rate[kk] > 1.02:
                c += rng.uniform(-jitter_max, jitter_max)
            place(t_out, grain_at(int(round(c)), uv_half), window(uv_half), 1.0)
            if trace is not None:
                trace.append((t_out, t_src, "U", int(round(c)), uv_half, float(seg_rate[kk])))
            t_out += uv_half

    out = out[pad: pad + n_out]
    wsum = wsum[pad: pad + n_out]
    norm = np.where(wsum > 1e-6, 1.0 / np.maximum(wsum, 0.25), 0.0)
    if trace is not None:
        trace.append(("wsum", wsum))
    return (out * norm).astype(np.float32)
