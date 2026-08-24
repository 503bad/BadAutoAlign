# 09. Phase 1 実装記録（2026-08-04）

06_改修工程計画.md の Phase 1 を実装した際の変更点と検証結果。

## 変更内容

### 1-1. time_warp のストリーム駆動修正（engines.py）

- チャンク長 2048 → **512** サンプル（レート追従が ~10ms 粒度に）
- 区間ごとの丸めリセットを廃止し、**フレーズ全体の累積写像**
  `np.interp(入力位置, warp.src, warp.dst)` で出力サンプル数を決定
- 旧実装の `ls <= 0 or ld <= 0: continue`（入力サンプル欠落の温床）を除去。
  入力は常に順番どおり全量送られる

### 1-2. 伸縮再配分パス（timing.py: `_elastic_redistribute`）

`_sanitize_anchors` の後段に追加。アンカー対応（芯の到達位置）は厳密に保ったまま、
各アンカー区間内のレート配分を重み付きで解き直す:

| ラベル | 判定 | 重み | レート上下限 |
|---|---|---|---|
| 無音 | 短窓RMS < silence_thresh_db | 1.0 | 0.25〜4.0 |
| 母音持続 | 有声フレーム | 0.5 | 0.667〜1.5 |
| 子音 | 無声かつ非無音 | 0.05 | 0.9〜1.1 |
| アタック | 音節頭 −20ms〜+40ms | 0.02 | 0.95〜1.05 |

- ソルバー: 重み比例配分＋上下限キャップの反復（8回）。不足分は一様配分で吸収
  （到達精度を上下限より優先。旧・線形配分と同等以上を保証）
- 重みは σ=30ms のガウシアンで平滑化し、レート遷移をなだらかにする
- 出力は 10ms グリッド ∪ アンカー位置の細かい折れ点列
- レポートの `lag_profile.applied_*` は再配分前の粗い折れ点列を維持（JSON肥大防止）
- `Config.elastic_warp = True`（`--no-elastic-warp` で旧動作）

### 1-4. フォルマント保持（engines.py / native）

- `wrapper.cpp` に `vs_set_formant_base` を追加（ソースハッシュ変更 → DLL自動再ビルド）
- `StretchStream.set_formant_factor(1.0, compensate=True)` ＝
  トランスポーズを打ち消す方向に包絡補正（ヘッダ実装で確認: `formantCompensation` は
  ターゲット包絡を `mapFreq` 前の周波数で参照する）
- フレーズの有声中央値 F0 を `setFormantBase` に設定し包絡推定を安定化
- `Config.formant_preserve = True`（`--no-formant-preserve` で旧動作）

### 1-5. tonality limit（engines.py）

- `set_transpose_factor(factor, tonality_limit_hz / sr)`。
  ヘッダ側で √factor の入出力妥協が適用される（`setTransposeFactor` 実装で確認）
- `Config.tonality_limit_hz = 8000.0`（0 で無効）

### 1-6. 設定フラグ（config.py / cli.py）

`formant_preserve` / `tonality_limit_hz` / `elastic_warp` を追加。
サービスモード（service.py）は Config フィールドを自動透過するため変更不要。
GUI から使う場合は options に同名キーを渡す。

## 検証結果（合成信号によるスモーク／E2E）

- **再配分**: アンカー到達誤差 < 1e-6 s、dst 単調、
  アタック保護区間レート 1.0±0.12 以内、有声区間レート 0.964〜1.065
  （+80ms 移動時。旧実装では区間全体が一様 ~1.08 で子音・アタックにも同率がかかっていた）
- **time_warp**: 恒等マップ・実ワープとも長さ保存・全サンプル有限・クリックなし
- **ピッチシフト**: +3半音指定で出力 F0 261.7Hz（理論値 261.6Hz）
- **E2E**（合成ボーカル -30cent・+80ms遅れ vs WAVガイド）:
  補正後 -0.0 cent（3ノートとも）、急峻サンプル差分 0、旧動作フラグでも正常動作
- **既知の残課題**: タイミング適用量が保守的（ラグ推定のガードによる。Phase 1 の範囲外）

## ビルド

- `uv pip install pyinstaller` 後、
  `python -m PyInstaller build/vat-serve.spec --distpath dist --workpath build/pyi -y`
- ネイティブDLL: ソース変更後の初回実行で自動ビルドされたものを
  `dist/vat-serve/vatstretch.dll` に配置（frozen 時は exe と同じ場所を探す）
- インストーラ: `cd standalone && npm run dist` → `standalone/release/*.exe`
