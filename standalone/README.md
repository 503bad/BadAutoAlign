# BadAutoAlign — スタンドアローン版（開発中）

開発: Igarashi Date（503 bad gateway）

Electron UI + Pythonバックエンド（`vat serve`、stdio上のJSON-RPC）構成。
UI/UX設計は `docs/soan.md` を参照。

## 現状の機能（v0骨格）

- ガイドボーカル / 補正対象ボーカル / オケ の3トラック＋補正結果レーン
- WAVのドラッグ&ドロップ、波形表示
- MIDIガイド（β）: ガイドレーンに `.mid` をドロップするとノート列をピアノロール風に表示し、
  そのままガイドとして補正できる（音は出ない）。初回ドロップ時にβ版ダイアログを表示、
  「次回から表示しない」は `userData/settings.json` の `midiBetaNoticeDismissed` に保存
- タイミング補正 / ピッチ補正 / 両方 の個別実行（CLIと同じパイプライン）
- 補正後、検出した発声ごとのマーカーとガイド⇔ボーカルの対応ラインを表示
  （緑=補正適用 / 黄=低信頼で未補正 / 赤=対応なし）
- ミュート切替つき同時再生、クリックでプレイヘッド移動

## 未実装（設計済み・docs/soan.md の順）

- マーカーのドラッグ移動 → 「再補正」（手動アンカーをガード免除で適用）
- 対応ラインの解除・付け替え（掛け違いの手動修正）
- フレーズ単位の部分再処理（再補正の高速化）
- Windowsパッケージング（Python同梱・ネイティブDLLプリビルド）

## 開発時の起動

```bash
cd standalone
npm install
npm start
```

バックエンドは `uv --directory <リポジトリルート> run vat serve` で自動起動される
（リポジトリルートでの `uv sync` 実行済みが前提）。
配布時は環境変数 `VAT_SERVE_CMD` で同梱Pythonのコマンドに差し替える。

## Windowsパッケージング方針（未実装）

1. `src/vat/native/wrapper.cpp` をCMakeでプリビルドし DLL を同梱
   （ローダーに「同梱バイナリ優先」の分岐を追加）
2. Python embeddable package + 依存wheelを同梱、または PyInstaller で
   `vat serve` を単一exe化
3. electron-builder でインストーラ作成
