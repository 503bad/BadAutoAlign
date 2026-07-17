# VocalAlignTune 仕様書

MIDI/WAVガイドを基準に、ボーカルWAVのタイミングとピッチを一括自動補正する
オフライン処理ツールの仕様書群。実装(v0.1)と実素材検証を経て確定した内容を記す。

初版の要求仕様は [`../spec.md`](../spec.md)（開発開始時の文書、歴史的資料として保存）。

## ドキュメント構成

| 文書 | 内容 |
|---|---|
| [00-overview.md](00-overview.md) | システム概要・設計原則・全体構成 |
| [01-pipeline.md](01-pipeline.md) | 処理パイプライン（フレーズ分割・非処理ルール・処理順序） |
| [02-theory-timing.md](02-theory-timing.md) | **タイミング補正の理論**（本仕様書の中核） |
| [03-theory-pitch.md](03-theory-pitch.md) | ピッチ補正の理論 |
| [04-detection.md](04-detection.md) | 検出系（F0・音節・芯・包絡） |
| [05-implementation.md](05-implementation.md) | 実装仕様（モジュール・CLI・サービス・レポート・パラメータ） |
| [06-gui.md](06-gui.md) | スタンドアローンGUI版の仕様 |
| [07-verification.md](07-verification.md) | 検証方法と実測結果・既知の限界 |

## 読み方

- 補正結果の性質・限界を理解したい → 02 / 03 / 07
- パラメータを調整したい → 05 の設定一覧と 02/03 の該当節
- GUIの操作意味論（マーカーの色など） → 06
- 移植・拡張したい → 00 → 01 → 05
