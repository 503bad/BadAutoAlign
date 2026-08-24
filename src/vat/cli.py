"""CLIエントリポイント。

vat process input.wav guide.mid -o output.wav [options]
"""

from __future__ import annotations

import argparse
import sys

from .config import Config


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="vat",
        description="VocalAlignTune — MIDI/WAVガイドによるボーカルのタイミング・ピッチ一括補正",
    )
    sub = p.add_subparsers(dest="command", required=True)

    proc = sub.add_parser("process", help="WAVを補正して出力する")
    proc.add_argument("input", help="入力ボーカルWAV")
    proc.add_argument("guide", help="ガイド（.mid/.midi または .wav）")
    proc.add_argument("-o", "--output", required=True, help="出力WAVパス")

    proc.add_argument("--engine", choices=["psola", "stretch", "world"], default="psola",
                      help="合成エンジン: psola=ピッチ同期グレイン再合成（既定・高品質）, "
                           "stretch=Signalsmith Stretch, world=WORLDボコーダ")
    proc.add_argument("--detector", choices=["auto", "rmvpe", "crepe", "pyin"],
                      default="auto")
    proc.add_argument("--rmvpe-model", default=None,
                      help="RMVPE ONNXモデルのパス（--detector rmvpe時に必要）")

    proc.add_argument("--pitch-strength", type=float, default=0.85,
                      help="ピッチ補正強度 0.0〜1.0（デフォルト0.85）")
    proc.add_argument("--timing-strength", type=float, default=1.0)
    proc.add_argument("--max-shift-ms", type=float, default=120.0,
                      help="タイミング最大移動量ガード")
    proc.add_argument("--silence-thresh-db", type=float, default=-45.0,
                      help="非処理RMS閾値（dBFS）")
    proc.add_argument("--min-gap-ms", type=float, default=200.0,
                      help="フレーズ区切りとみなす最小無音長")
    proc.add_argument("--attack-preserve-ms", type=float, default=80.0,
                      help="ノート先頭の補正ランプイン（しゃくり保持）")

    proc.add_argument("--no-formant-preserve", action="store_true",
                      help="フォルマント保持を無効化（旧動作）")
    proc.add_argument("--tonality-limit-hz", type=float, default=8000.0,
                      help="非調和高域を1:1に保つコーナー周波数。0で無効")
    proc.add_argument("--no-elastic-warp", action="store_true",
                      help="タイミング補正の伸縮再配分を無効化（旧動作）")

    proc.add_argument("--report", default=None, help="レポートJSONの出力先")

    mode = proc.add_mutually_exclusive_group()
    mode.add_argument("--pitch-only", action="store_true")
    mode.add_argument("--timing-only", action="store_true")

    proc.add_argument("--guide-type", choices=["auto", "midi", "wav"], default="auto")
    proc.add_argument("--pitch-target", choices=["note", "curve"], default="note",
                      help="WAVガイド時のピッチターゲット")

    sub.add_parser("serve", help="GUI向けサービスモード（stdio上のJSON-RPC）")
    return p


def config_from_args(args: argparse.Namespace) -> Config:
    return Config(
        engine=args.engine,
        detector=args.detector,
        rmvpe_model=args.rmvpe_model,
        pitch_strength=args.pitch_strength,
        timing_strength=args.timing_strength,
        max_shift_ms=args.max_shift_ms,
        silence_thresh_db=args.silence_thresh_db,
        min_gap_ms=args.min_gap_ms,
        attack_preserve_ms=args.attack_preserve_ms,
        formant_preserve=not args.no_formant_preserve,
        tonality_limit_hz=args.tonality_limit_hz,
        elastic_warp=not args.no_elastic_warp,
        pitch_only=args.pitch_only,
        timing_only=args.timing_only,
        guide_type=args.guide_type,
        pitch_target=args.pitch_target,
        report_path=args.report,
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "serve":
        from .service import serve

        return serve()
    if args.command != "process":
        return 2
    cfg = config_from_args(args)

    from .pipeline import process_file
    from .report import write_report

    report = process_file(args.input, args.guide, args.output, cfg)
    if cfg.report_path:
        write_report(report, cfg.report_path, args.input, args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
