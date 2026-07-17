"""処理パラメータ一式。CLI引数と1対1で対応する。"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict


@dataclass
class Config:
    # エンジン / 検出器
    engine: str = "stretch"          # stretch | world
    detector: str = "auto"           # auto | rmvpe | crepe | pyin
    rmvpe_model: str | None = None   # RMVPE ONNXモデルのパス（detector=rmvpe時に必要）

    # ピッチ補正
    pitch_strength: float = 0.85     # 0.0〜1.0（P3）
    attack_preserve_ms: float = 80.0 # ノート先頭のランプイン（しゃくり保持）
    pitch_smooth_ms: float = 40.0    # 補正カーブのガウシアン平滑 σ
    voicing_fade_ms: float = 10.0    # 有声⇔無声境界のクロスフェード（P2: 5〜20ms）
    min_correction_cents: float = 5.0  # これ未満の補正しか無いフレーズは触らない（ヌルテスト対応）
    max_correction_cents: float = 300.0  # ガード: これを超えるオフセットは誤検出とみなしスキップ

    # タイミング補正
    timing_strength: float = 1.0
    max_shift_ms: float = 120.0      # T2-4 最大移動量ガード
    min_shift_ms: float = 35.0       # これ未満のズレは補正しない（オンセット検出の量子化
                                     # ・遅延ジッタより十分大きく取る。ヌルテスト対応）
    min_gap_ms: float = 200.0        # T1 フレーズ区切りの最小無音長
    phrase_match_tol_s: float = 0.5  # フレーズ対応付けの開始時刻許容差

    # 非処理ルール
    silence_thresh_db: float = -45.0 # RMSゲート閾値（dBFS）

    # 解析
    hop: int = 512                   # 解析ホップ（サンプル）
    frame_length: int = 2048
    fmin: float = 65.40639           # C2。pYINのビングリッドを平均律に整列させる
    fmax: float = 1000.0
    pyin_resolution: float = 0.1     # pYIN分解能（半音）。10セント。グリッドが平均律に
                                     # 整列しているため中央値オフセット補正には十分。
                                     # 細かくすると計算量がビン数の2乗で増える
    min_voiced_conf: float = 0.5     # ピッチ信頼度がこれ未満のフレームは補正しない

    # モード
    pitch_only: bool = False
    timing_only: bool = False

    # ガイド
    guide_type: str = "auto"         # auto | midi | wav
    pitch_target: str = "note"       # note | curve（WAVガイド時のみ有効）

    # 出力
    report_path: str | None = None

    warnings: list[str] = field(default_factory=list)

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)
        print(f"[warn] {msg}")

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("warnings", None)
        return d
