# VocalAlignTune (v0.1 — CLIプロトタイプ)

MIDIノート（またはガイドWAV）をガイドとして、ボーカルWAVの**タイミング**と**ピッチ**を
一括自動補正するオフライン処理ツール。仕様は `docs/spec.md` を参照。

## セットアップ

```bash
uv sync            # Python 3.12 venv作成＋依存インストール
uv run vat --help
```

既定エンジン（psola）は純Python/NumPyで追加ビルド不要です。
`--engine stretch` 指定時のみ、初回実行時に Signalsmith Stretch のC++ラッパーを自動ビルドします
（システムのC++コンパイラが必要。macOSはXcode Command Line Toolsで可。
Windowsは g++/clang++（MSYS2, w64devkit等）のほか、Visual Studio /
Build Tools がインストールされていれば vswhere 経由で自動検出します）。
コンパイラの無い環境では自動的に `world` エンジンにフォールバックします
（`--engine world` 相当。レポートのwarningsに記録されます）。

## 使い方

```bash
uv run vat process input.wav guide.mid -o output.wav \
  [--engine psola|stretch|world] \
  [--detector auto|rmvpe|crepe|pyin] \
  [--pitch-strength 0.85] \
  [--timing-strength 1.0] \
  [--max-shift-ms 120] \
  [--silence-thresh-db -45] \
  [--min-gap-ms 200] \
  [--attack-preserve-ms 80] \
  [--report report.json] \
  [--pitch-only | --timing-only]

# WAVガイド（Synthesizer V等の合成ボーカル）
uv run vat process input.wav guide.wav -o output.wav --pitch-target note|curve
```

- `--detector auto`（デフォルト）: rmvpe → crepe → pyin の順で利用可能なものを選択
  - rmvpe: `--rmvpe-model model.onnx` でONNXモデルの指定が必要（onnxruntime導入時のみ）
  - crepe: `uv sync --extra crepe` でtorch/torchcrepeを導入した場合のみ
  - pyin: 追加依存なしで常に利用可（librosa）
- MIDIガイド（β）: `.mid/.midi` を渡すとノート列をガイドにする。ガイド音声が無いため
  タイミング補正はノートオンを芯とみなし、ボーカル側の芯（P-center）検出のみで合わせる
  （WAVガイドの包絡相互相関による密なラグ推定は使われない）。ピッチ補正はWAVガイドと同じ
- `--report`: ノートごとの補正前後F0・移動量・スキップ理由をJSON出力。
  matplotlib導入時は処理前後のスペクトログラム比較PNGも出力

## 実装の要点（仕様との対応）

| 仕様 | 実装 |
|---|---|
| P1 シフトエンジン | 既定は `vat/psola.py` — ピッチ同期グレイン再合成（TD-PSOLA、Melodyne 系の周期単位ローカル再合成）。ワープとピッチシフトを1パスで適用し、有声部の倍音位相・フォルマントをそのまま保つ。代替: Signalsmith Stretch (`--engine stretch`、`vat/native/`)、WORLD (`--engine world`) |
| P2 無声音素通し | pyin信頼度＋RMS＋ZCRで有声判定、無声はシフト比1.0、境界10msクロスフェード |
| P3 スナップしない補正 | ノートごとに検出F0中央値とのオフセットのみ補正、σ=40msガウシアン平滑、先頭80msランプイン、`--pitch-strength` |
| P4 検出器 | rmvpe(ONNX) / torchcrepe / pyin 切替式 |
| T1 フレーズ分割 | RMSゲート（無音≥200ms）で音声・MIDI双方を分割、開始時刻の貪欲マッチング |
| T2 タイミングアライメント | 中帯域エネルギー包絡の局所相互相関で密なラグサンプルを取得→頑健平滑カーブ（外れ値除去・傾き制限・不感帯・閉ループ検証）でワープ。音節単位の離散アライメント（Needleman-Wunsch）は対応表レポートとフォールバックに使用 |
| T3 処理順序 | タイミング補正 → ピッチ再検出 → ピッチ補正 |
| 非処理ルール | ノート無し/RMS閾値未満/対応不能/ガード超過/低信頼度 はすべて素通し（未処理区間はビット一致） |

### 注記: python-stretch を使わない理由

PyPIの `python-stretch` バインディングは呼び出しごとに内部で seek/flush を行う
ワンショットAPIであり、チャンクごとにパラメータを変える可変レートストレッチ・
時間変化ピッチシフトには使えないことを確認した（チャンク処理と全体処理の結果が
一致しない）。このため仕様P1の代替案どおり、MITライセンスのヘッダを
`src/vat/native/vendor/` にベンダリングし、薄いC ABIラッパーを自作している。

## テスト

```bash
uv run pytest tests/ -q
```

- 合成テスト: 擬似ボーカル（ノコギリ波＋フォルマントフィルタ）に既知の
  ±30セント／+60msのずれを付与 → 補正後残差 ピッチ<10セント・タイミング<20ms を検証
- ヌルテスト: ずれの無い入力で出力がビット一致することを検証
- 両エンジン（stretch/world）、max-shiftガード、DTW経路、CLIエンドツーエンドを網羅

## ライセンス

本体はプロプライエタリ（© 2026 503 bad gateway、`LICENSE`）。配布バイナリの使用条件は `EULA.txt`。

主な依存（全リストと義務は `THIRD_PARTY_NOTICES.md`）:

| 依存 | ライセンス |
|---|---|
| Signalsmith Stretch / signalsmith-linear（ベンダリング、`--engine stretch` 時のみ） | MIT |
| WORLD (pyworld) | 修正BSD |
| librosa（pYIN。pitch_shiftは使用禁止） | ISC |
| pretty_midi / mido | MIT |
| numpy / scipy / scikit-learn / numba | BSD系 |
| soundfile（バインディング） / libsndfile | BSD-3 / **LGPL-2.1+** |
| soxr（librosa依存） / libsoxr | **LGPL-2.1+** |
| torchcrepe（optional） | MIT |

強いコピーレフト（GPL）は不使用。LGPL の 3 件（libsndfile / libsoxr / Electron 同梱 FFmpeg）は
改変せず差し替え可能な独立ファイルとして同梱し、全文・表記・ソース入手先を `THIRD_PARTY_NOTICES.md` に記載。
