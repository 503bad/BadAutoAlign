# 05. 実装仕様

## 1. モジュール構成（src/vat/）

| モジュール | 責務 |
|---|---|
| cli.py | `vat process` / `vat serve` のargparse |
| config.py | 全パラメータ（dataclass、CLI/serviceと1対1） |
| audio.py | WAV入出力（float32モノラル化）、フレームRMS/ZCR |
| guide.py | ガイドアダプタ（MIDI/WAV→ノート列）、F0キャッシュ |
| detect.py | ピッチ検出器（pyin/rmvpe/crepe）＋共通ゲート |
| segment.py | フレーズ分割・ノート⇔フレーズ対応付け |
| syllables.py | 音節分割（＋過分割マージ） |
| features.py | CMVN-MFCC音素特徴・DTWコスト |
| pcenter.py | 中帯域包絡・芯（ピッチ到達点/エネルギー上昇点） |
| align.py | ノート⇔音節のNeedleman-Wunsch・対応表 |
| timing.py | ラグ計測・平滑カーブ・ガード・手動アンカー・ワープマップ |
| pitch.py | 補正カーブ生成・NoteReport |
| engines.py | SignalsmithEngine / WorldEngine |
| native/ | Signalsmith Stretch 自作ラッパー（vendor/ にMITヘッダ同梱） |
| pipeline.py | フレーズごとの統括・スプライス・レポート組み立て |
| service.py | GUI向けJSON-RPC（stdio） |
| report.py | レポートJSON書き出し・スペクトログラム比較PNG（matplotlib任意） |

## 2. CLI仕様

```
vat process input.wav guide.(mid|wav) -o output.wav
  [--engine stretch|world]            既定 stretch
  [--detector auto|rmvpe|crepe|pyin]  既定 auto
  [--rmvpe-model model.onnx]
  [--pitch-strength 0.85] [--timing-strength 1.0]
  [--max-shift-ms 120] [--silence-thresh-db -45] [--min-gap-ms 200]
  [--attack-preserve-ms 80]
  [--report report.json]
  [--pitch-only | --timing-only]
  [--guide-type auto|midi|wav] [--pitch-target note|curve]

vat serve        # GUI向けサービスモード
```

## 3. サービスプロトコル（vat serve）

stdio上の行区切りJSON-RPC風。**stdoutはプロトコル専用**
（処理ログ・進捗printはすべてstderrへリダイレクト）。

```
→ {"id": 1, "method": "version", "params": {}}
← {"id": 1, "ok": true, "result": {"version": "0.1.0"}}

→ {"id": 2, "method": "process", "params": {
     "input": "...wav", "guide": "...wav", "output": "...wav",
     "options": {Configフィールドの部分集合},
     "manual_anchors": [{"src_s": 12.34, "dst_s": 12.20, "note_index": 41}]
   }}
← {"id": 2, "ok": true, "result": {レポート}}
← {"id": 2, "ok": false, "error": "..."}   # 例外時（サービスは落ちない）
```

進捗はstderrの行（`フレーズ: 音声 N / MIDI M / 対応 K`、
`  フレーズ a-bs: timing=... pitch=...`）をホスト側でパースする。

## 4. レポートJSONスキーマ（主要部）

```jsonc
{
  "version": "0.1.0",
  "input": "...", "guide": "...", "output": "...", "sample_rate": 44100,
  "config": { /* 使用した全パラメータ */ },
  "warnings": ["..."],
  "phrases": [{
    "start_s": 12.66, "end_s": 18.29, "n_notes": 16,
    "timing_applied": true, "pitch_applied": true,
    "base_shift_ms": 0.0,
    "alignment": [{           // ノート⇔音節の対応表（レビュー・AI裁定用）
      "note_index": 0, "note_start_s": 12.91, "note_pitch": 57.0,
      "syllable_onset_s": 0.31, "syllable_semitone": 57.0,
      "cost": 0.36, "decision": "matched|low_confidence|note_skipped"
    }],
    "lag_profile": {          // 計測と適用（残差=計測-適用）
      "measured_t_s": [...], "measured_ms": [...], "measured_from_xcorr": true,
      "applied_t_s": [...], "applied_ms": [...]
    }
  }],
  "notes": [{
    "index": 0, "midi_pitch": 57.0, "start_s": 12.91, "end_s": 13.20,
    "detected_median_hz": 220.1,
    "offset_cents_before": 31.0, "applied_cents": 26.4,
    "timing_shift_ms": -15.8,        // アンカー計測値（芯対芯）
    "timing_applied": true,
    "timing_applied_ms": -40.6,      // その位置で実際に適用した移動量
    "timing_residual_ms": 3.1,       // まだ残っているズレ（null=計測不能）
    "anchor_src_s": 12.92, "anchor_dst_s": 12.90,
    "manual": false,
    "skip_reasons": ["shift_exceeds_max", ...]
  }]
}
```

skip_reasons の語彙: no_matching_phrase / no_syllable_match /
timing_low_confidence / anchor_pitch_mismatch / shift_exceeds_max /
low_confidence / offset_exceeds_guard / note_outside_phrase /
correction_below_threshold。

## 5. 主要パラメータ（Config）

| パラメータ | 既定 | 説明 |
|---|---|---|
| engine / detector | stretch / auto | エンジン・検出器 |
| pitch_strength | 0.85 | ピッチ補正強度（1.0=完全一致） |
| pitch_smooth_ms / voicing_fade_ms | 40 / 10 | カーブ平滑σ・有声境界フェード |
| attack_preserve_ms | 80 | ノート頭のランプイン |
| min/max_correction_cents | 5 / 300 | ピッチ側の下限/ガード |
| timing_strength | 1.0 | タイミング補正強度 |
| min/max_shift_ms | 35 / 120 | タイミングの下限/ガード |
| min_gap_ms | 200 | フレーズ区切りの最小無音 |
| silence_thresh_db | -45 | 非処理RMSゲート |
| hop / frame_length | 512 / 2048 | 解析グリッド |
| fmin | 65.40639 (C2) | pYIN格子の平均律整列（変更非推奨） |
| pyin_resolution | 0.1 | pYIN分解能（半音）。細分は計算量がビン数²で増える |
| min_voiced_conf | 0.5 | 有声判定の信頼度下限 |

## 6. ネイティブラッパー（native/）

- vendor/: signalsmith-stretch.h + signalsmith-linear（いずれもMIT、LICENSE同梱）
- wrapper.cpp: モノラル・ストリーミングのC ABI（vs_create/process/flush等）。
  **固定シード**で生成（再現性。既定コンストラクタはrandom_deviceで毎回変わる）
- ローダー: 初回利用時にシステムの `c++` でビルドし、ソースハッシュをキーに
  ユーザーキャッシュへ保存。コンパイラの無い環境はエラー文言で --engine world を案内
- 配布時はCMakeでプリビルドしたDLL/dylibを同梱し、ローダーに同梱バイナリ優先の
  分岐を追加する（Windowsパッケージング方針 → 06 §5）

## 7. 決定性

同一入力・同一パラメータ・同一バージョンで出力はビット単位に再現される
（エンジン固定シード、解析は決定的）。GUIの「再補正」やA/B比較の前提。
