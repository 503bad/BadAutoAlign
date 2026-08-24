"""シフト/ストレッチエンジン（P1）。

- SignalsmithEngine: 周波数領域＋位相コヒーレンス＋フォルマント処理（MIT）。
  ストリーミングAPIでチャンクごとにレート/シフト量を変えられるため、
  可変レートワープと時間変化ピッチシフトの両方をこれ一つで行う。
- WorldEngine: F0/スペクトル包絡/非周期性に分解して再合成（修正BSD）。
  A/B比較用フォールバック。
- PsolaEngine（既定）: ピッチ同期グレイン再合成（TD-PSOLA、vat.psola）。
  ワープとピッチシフトを1パスで適用し、有声部は周期波形をそのまま
  並べ替えるため位相ボコーダ特有のにじみ・ビリビリが出ない。純Python。

librosa.effects.pitch_shift と単純リサンプリングは使用禁止（仕様）。
"""

from __future__ import annotations

import numpy as np

from .config import Config
from .detect import PitchTrack
from .timing import WarpMap


def get_engine(cfg: Config) -> "BaseEngine":
    if cfg.engine == "psola":
        return PsolaEngine(cfg)
    if cfg.engine == "stretch":
        from . import native

        if not native.is_available():
            cfg.warn(
                "C++コンパイラが無くstretchエンジンをビルドできないため"
                "worldエンジンにフォールバックします")
            return WorldEngine(cfg)
        return SignalsmithEngine(cfg)
    if cfg.engine == "world":
        return WorldEngine(cfg)
    raise ValueError(f"未知のエンジン: {cfg.engine}")


class BaseEngine:
    # True のエンジンは process() でワープ＋ピッチを同時に適用できる
    single_pass = False

    def __init__(self, cfg: Config):
        self.cfg = cfg

    def process(self, audio: np.ndarray, sr: int, track: PitchTrack,
                warp: WarpMap | None, curve_semitones: np.ndarray | None) -> np.ndarray:
        """ワープと補正カーブ（ワープ後時間軸のフレームごと半音）を1パスで適用。"""
        raise NotImplementedError

    def pitch_shift(self, audio: np.ndarray, sr: int, track: PitchTrack,
                    curve_semitones: np.ndarray) -> np.ndarray:
        """フレームごとの補正量カーブ（半音）でピッチシフト。長さは入力と同一。"""
        raise NotImplementedError

    def time_warp(self, audio: np.ndarray, sr: int, warp: WarpMap) -> np.ndarray:
        """区分線形ワープマップで可変レートストレッチ。長さは入力と同一
        （マップ両端が固定されている前提）。"""
        raise NotImplementedError


# ---------------------------------------------------------------- Signalsmith

class SignalsmithEngine(BaseEngine):
    """自作ラッパー（vat.native）によるストリーミング処理。

    SignalsmithStretch本体のストリーミングAPIは process(in, n_in, out, n_out) の
    サイズ比が伸縮率を決める。チャンクごとにトランスポーズ量を更新できるため、
    時間変化するピッチシフトと可変レートワープの両方をこれで行う。
    """

    def _new_stream(self, sr: int, track: PitchTrack | None):
        from .native import StretchStream

        s = StretchStream(float(sr))
        if self.cfg.formant_preserve:
            # 包絡を固定してシフト（声質保持）。基準F0で包絡推定を安定させる
            s.set_formant_factor(1.0, compensate=True)
            if track is not None and track.voiced.any():
                s.set_formant_base(float(np.median(track.f0[track.voiced])))
        return s

    def _tonality(self, sr: int) -> float:
        hz = self.cfg.tonality_limit_hz
        return hz / sr if hz > 0 else 0.0

    def pitch_shift(self, audio, sr, track, curve_semitones):
        hop = self.cfg.hop
        s = self._new_stream(sr, track)
        latency = s.input_latency + s.output_latency
        tonality = self._tonality(sr)
        x = audio.astype(np.float32)
        n = len(audio)
        chunks = []
        frame = 0
        for pos in range(0, n, hop):
            chunk = x[pos: pos + hop]
            semi = float(curve_semitones[min(frame, len(curve_semitones) - 1)])
            s.set_transpose_factor(2.0 ** (semi / 12.0), tonality)
            chunks.append(s.process(chunk, len(chunk)))
            frame += 1
        chunks.append(s.process(np.zeros(s.input_latency, dtype=np.float32),
                                s.input_latency))
        chunks.append(s.flush(s.output_latency))
        out = np.concatenate(chunks)
        return _fix_length(out[latency:], n)

    def time_warp(self, audio, sr, warp):
        s = self._new_stream(sr, None)
        latency = s.input_latency + s.output_latency
        x = audio.astype(np.float32)
        n = len(audio)
        # 出力サンプル数はフレーズ全体の累積写像から決める。区間ごとの
        # 丸めリセットを無くし、入力を欠落なく順番どおり送る（レートは
        # チャンク粒度で warp のレートカーブに追従する）
        src_s = warp.src * sr
        dst_s = warp.dst * sr
        chunk = 512
        chunks = []
        emitted = 0
        for pos in range(0, n, chunk):
            end = min(pos + chunk, n)
            target = float(np.interp(end, src_s, dst_s))
            want = max(int(round(target)) - emitted, 0)
            chunks.append(s.process(x[pos:end], want))
            emitted += want
        chunks.append(s.process(np.zeros(s.input_latency, dtype=np.float32),
                                s.input_latency))
        chunks.append(s.flush(s.output_latency))
        out = np.concatenate(chunks)
        return _fix_length(out[latency:], n)


# ---------------------------------------------------------------- PSOLA

class PsolaEngine(BaseEngine):
    """ピッチ同期グレイン再合成（vat.psola）。

    検出結果（PitchTrack）から合成用の周期解析を行い、ワープマップと
    補正カーブを1パスで適用する。無変更区間は入力と一致する。
    """

    single_pass = True

    def synthesis_voicing(self, audio, sr, track) -> np.ndarray:
        """合成側の有声判定（周期性ベース）。補正カーブの P2 マスクに使う。"""
        from . import psola

        voiced, _ = psola.synthesis_voicing(audio, sr, track, self.cfg.hop)
        return voiced

    def process(self, audio, sr, track, warp, curve_semitones):
        from . import psola

        n = len(audio)
        runs, noise = psola.analyze(audio, sr, track, self.cfg.hop)
        if warp is None or warp.is_identity():
            src = np.array([0.0, float(n)])
            dst = np.array([0.0, float(n)])
        else:
            src = np.asarray(warp.src, dtype=np.float64) * sr
            dst = np.asarray(warp.dst, dtype=np.float64) * sr
            src[0], dst[0] = 0.0, 0.0
            src[-1], dst[-1] = float(n), float(n)
        factor_at = None
        if curve_semitones is not None and np.any(curve_semitones != 0):
            dt_s = self.cfg.hop  # カーブはワープ後時間軸の解析グリッド（hop間隔）
            t_grid = np.arange(len(curve_semitones)) * dt_s
            fac = 2.0 ** (np.asarray(curve_semitones, dtype=np.float64) / 12.0)

            def factor_at(t_out: float) -> float:
                return float(np.interp(t_out, t_grid, fac))
        y = psola.render(audio, sr, runs, src, dst, factor_at=factor_at, n_out=n,
                         noise_frames=noise, hop=self.cfg.hop)
        return _fix_length(y, n)

    def pitch_shift(self, audio, sr, track, curve_semitones):
        return self.process(audio, sr, track, None, curve_semitones)

    def time_warp(self, audio, sr, warp, track: PitchTrack | None = None):
        if track is None:
            from .detect import detect_pitch

            track = detect_pitch(audio, sr, self.cfg)
        return self.process(audio, sr, track, warp, None)


# ---------------------------------------------------------------- WORLD

class WorldEngine(BaseEngine):
    FRAME_PERIOD_MS = 5.0

    def _analyze(self, audio: np.ndarray, sr: int):
        import pyworld as pw

        x = audio.astype(np.float64)
        f0, t = pw.harvest(x, sr, f0_floor=self.cfg.fmin,
                           f0_ceil=self.cfg.fmax * 1.2,
                           frame_period=self.FRAME_PERIOD_MS)
        f0 = pw.stonemask(x, f0, t, sr)
        sp = pw.cheaptrick(x, f0, t, sr)
        ap = pw.d4c(x, f0, t, sr)
        return f0, t, sp, ap

    def pitch_shift(self, audio, sr, track, curve_semitones):
        import pyworld as pw

        f0, t, sp, ap = self._analyze(audio, sr)
        curve = np.interp(t, track.times, curve_semitones)
        f0_new = np.where(f0 > 0, f0 * 2.0 ** (curve / 12.0), 0.0)
        y = pw.synthesize(f0_new, sp, ap, sr, self.FRAME_PERIOD_MS)
        return _fix_length(y.astype(np.float32), len(audio))

    def time_warp(self, audio, sr, warp):
        import pyworld as pw

        f0, t, sp, ap = self._analyze(audio, sr)
        src_t = warp.dst_to_src(t)  # 出力フレームごとの参照元時刻
        idx = src_t / (self.FRAME_PERIOD_MS / 1000.0)
        idx = np.clip(idx, 0, len(f0) - 1)
        i0 = np.floor(idx).astype(int)
        i1 = np.minimum(i0 + 1, len(f0) - 1)
        w = (idx - i0)[:, None]
        sp_w = np.ascontiguousarray(sp[i0] * (1 - w) + sp[i1] * w)
        ap_w = np.ascontiguousarray(ap[i0] * (1 - w) + ap[i1] * w)
        # F0は有声/無声の混在補間を避ける（最近傍で有声性を保ち、有声同士のみ対数補間）
        near = np.where(w[:, 0] < 0.5, i0, i1)
        f0_w = f0[near]
        both = (f0[i0] > 0) & (f0[i1] > 0)
        f0_w[both] = np.exp(
            np.log(f0[i0][both]) * (1 - w[both, 0]) + np.log(f0[i1][both]) * w[both, 0]
        )
        y = pw.synthesize(f0_w, sp_w, ap_w, sr, self.FRAME_PERIOD_MS)
        return _fix_length(y.astype(np.float32), len(audio))


def _fix_length(x: np.ndarray, n: int) -> np.ndarray:
    if len(x) >= n:
        return x[:n]
    return np.pad(x, (0, n - len(x)))
