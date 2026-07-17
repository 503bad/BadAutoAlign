"""ピッチ検出器（P4）。

- rmvpe : RMVPE の ONNX モデルによる推論（onnxruntime、要モデルファイル）
- crepe : torchcrepe（optional extra "crepe" のインストールが必要）
- pyin  : librosa.pyin（追加依存なしのフォールバック）

全検出器は共通の解析グリッド（hopサンプル間隔、フレーム数 1 + len//hop）に
リサンプリングした PitchTrack を返す。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .audio import frame_rms, frame_zcr, db_to_lin
from .config import Config


@dataclass
class PitchTrack:
    times: np.ndarray   # 各フレームの中心時刻 [s]
    f0: np.ndarray      # Hz。無声フレームは 0
    voiced: np.ndarray  # bool
    conf: np.ndarray    # 0〜1

    @property
    def n_frames(self) -> int:
        return len(self.times)

    def semitones(self, ref: float = 440.0) -> np.ndarray:
        """A4=69 基準のMIDIノート番号スケール。無声はnan。"""
        out = np.full_like(self.f0, np.nan)
        v = self.f0 > 0
        out[v] = 69.0 + 12.0 * np.log2(self.f0[v] / ref)
        return out


def _grid_times(n_samples: int, sr: int, hop: int) -> np.ndarray:
    n_frames = 1 + n_samples // hop
    return np.arange(n_frames) * (hop / sr)


def _regrid(times_src, f0_src, conf_src, grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """検出器ネイティブのフレームレートを共通グリッドへ最近傍で載せ替える。"""
    idx = np.searchsorted(times_src, grid)
    idx = np.clip(idx, 0, len(times_src) - 1)
    left = np.clip(idx - 1, 0, len(times_src) - 1)
    use_left = np.abs(times_src[left] - grid) < np.abs(times_src[idx] - grid)
    idx = np.where(use_left, left, idx)
    return f0_src[idx].copy(), conf_src[idx].copy()


def _apply_gates(track: PitchTrack, audio: np.ndarray, sr: int, cfg: Config) -> PitchTrack:
    """P2: 信頼度・RMS・ZCRによる有声判定の絞り込み。"""
    rms = frame_rms(audio, cfg.hop, cfg.frame_length)[: track.n_frames]
    zcr = frame_zcr(audio, cfg.hop, cfg.frame_length)[: track.n_frames]
    gate = rms >= db_to_lin(cfg.silence_thresh_db)
    # ZCRが極端に高いフレームは無声摩擦音とみなす（歌声の有声域では通常 <0.3）
    voiced = track.voiced & (track.conf >= cfg.min_voiced_conf) & gate & (zcr < 0.35)
    f0 = np.where(voiced, track.f0, 0.0)
    return PitchTrack(track.times, f0, voiced, track.conf)


def detect_pitch(audio: np.ndarray, sr: int, cfg: Config) -> PitchTrack:
    """cfg.detector に従いピッチ検出。auto は rmvpe → crepe → pyin の順で試す。"""
    name = cfg.detector
    if name == "auto":
        for cand in ("rmvpe", "crepe", "pyin"):
            try:
                track = _dispatch(cand, audio, sr, cfg)
                cfg.detector = cand  # 実際に使った検出器を記録（レポート用）
                return _apply_gates(track, audio, sr, cfg)
            except (ImportError, FileNotFoundError, ValueError):
                continue
        raise RuntimeError("利用可能なピッチ検出器がありません")
    return _apply_gates(_dispatch(name, audio, sr, cfg), audio, sr, cfg)


def _dispatch(name: str, audio: np.ndarray, sr: int, cfg: Config) -> PitchTrack:
    if name == "pyin":
        return _detect_pyin(audio, sr, cfg)
    if name == "crepe":
        return _detect_crepe(audio, sr, cfg)
    if name == "rmvpe":
        return _detect_rmvpe(audio, sr, cfg)
    raise ValueError(f"未知の検出器: {name}")


# ---------------------------------------------------------------- pyin

def _detect_pyin(audio: np.ndarray, sr: int, cfg: Config) -> PitchTrack:
    import librosa

    f0, voiced_flag, voiced_prob = librosa.pyin(
        audio.astype(np.float64),
        fmin=cfg.fmin,
        fmax=cfg.fmax,
        sr=sr,
        frame_length=cfg.frame_length,
        hop_length=cfg.hop,
        resolution=cfg.pyin_resolution,
        center=True,
    )
    grid = _grid_times(len(audio), sr, cfg.hop)
    n = min(len(grid), len(f0))
    f0 = np.nan_to_num(f0[:n], nan=0.0)
    conf = np.nan_to_num(voiced_prob[:n], nan=0.0)
    voiced = voiced_flag[:n].astype(bool) & (f0 > 0)
    return PitchTrack(grid[:n], f0, voiced, conf)


# ---------------------------------------------------------------- torchcrepe

def _detect_crepe(audio: np.ndarray, sr: int, cfg: Config) -> PitchTrack:
    import torch
    import torchcrepe

    device = "cpu"
    target_sr = 16000
    x = _resample(audio, sr, target_sr)
    hop_16k = 160  # 10ms
    t = torch.from_numpy(x).float().unsqueeze(0)
    f0, periodicity = torchcrepe.predict(
        t, target_sr, hop_16k,
        fmin=cfg.fmin, fmax=cfg.fmax,
        model="full", batch_size=512, device=device,
        return_periodicity=True,
    )
    f0 = f0.squeeze(0).numpy()
    conf = periodicity.squeeze(0).numpy()
    times = np.arange(len(f0)) * (hop_16k / target_sr)
    grid = _grid_times(len(audio), sr, cfg.hop)
    f0g, confg = _regrid(times, f0, conf, grid)
    voiced = confg >= cfg.min_voiced_conf
    return PitchTrack(grid, np.where(voiced, f0g, 0.0), voiced, confg)


# ---------------------------------------------------------------- RMVPE (ONNX)

# RMVPE標準の定数: 16kHz / hop160 / 出力360ビン（20セント刻み、基準 ~32.70Hz=C1）
_RMVPE_SR = 16000
_RMVPE_HOP = 160
_RMVPE_CENTS_BASE = 1997.3794084376191


def _detect_rmvpe(audio: np.ndarray, sr: int, cfg: Config) -> PitchTrack:
    if not cfg.rmvpe_model:
        raise FileNotFoundError(
            "RMVPEを使うには --rmvpe-model でONNXモデルのパスを指定してください"
        )
    import onnxruntime as ort  # 依存はMIT。未導入なら ImportError → autoは次候補へ

    x = _resample(audio, sr, _RMVPE_SR)
    mel = _rmvpe_mel(x)
    sess = ort.InferenceSession(cfg.rmvpe_model, providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0].name
    hidden = sess.run(None, {inp: mel[None].astype(np.float32)})[0][0]  # (T, 360)
    f0, conf = _rmvpe_decode(hidden)
    times = np.arange(len(f0)) * (_RMVPE_HOP / _RMVPE_SR)
    grid = _grid_times(len(audio), sr, cfg.hop)
    f0g, confg = _regrid(times, f0, conf, grid)
    voiced = (confg >= max(cfg.min_voiced_conf * 0.06, 0.03)) & (f0g > 0)
    return PitchTrack(grid, np.where(voiced, f0g, 0.0), voiced, np.clip(confg / 0.1, 0, 1))


def _rmvpe_mel(x: np.ndarray) -> np.ndarray:
    import librosa

    mel = librosa.feature.melspectrogram(
        y=x, sr=_RMVPE_SR, n_fft=1024, hop_length=_RMVPE_HOP,
        win_length=1024, n_mels=128, fmin=30, fmax=8000, power=1.0,
    )
    return np.log(np.clip(mel, 1e-5, None)).T  # (T, 128)


def _rmvpe_decode(hidden: np.ndarray, thresh: float = 0.03) -> tuple[np.ndarray, np.ndarray]:
    """360ビンのサリエンスからローカル加重平均でセントをデコード。"""
    cents_map = 20.0 * np.arange(360) + _RMVPE_CENTS_BASE
    center = hidden.argmax(axis=1)
    conf = hidden.max(axis=1)
    f0 = np.zeros(len(hidden))
    for i, c in enumerate(center):
        lo, hi = max(0, c - 4), min(360, c + 5)
        w = hidden[i, lo:hi]
        cents = float(np.sum(w * cents_map[lo:hi]) / (np.sum(w) + 1e-9))
        f0[i] = 10.0 * 2 ** (cents / 1200.0)
    f0[conf < thresh] = 0.0
    return f0, conf


def _resample(audio: np.ndarray, sr: int, target_sr: int) -> np.ndarray:
    if sr == target_sr:
        return audio.astype(np.float32)
    import soxr

    return soxr.resample(audio, sr, target_sr).astype(np.float32)
