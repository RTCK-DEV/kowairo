# Kowairo 仕様書

リアルタイム AI ボイスチェンジャー — AivisSpeech (Style-Bert-VITS2) 音声モデル専用。

## 1. 概要

マイク入力の発話をほぼリアルタイムで対象 AI 音声モデルの声に変換して出力する
スタンドアロン GUI アプリケーション。対象モデルは TTS（テキスト→音声）モデルの
ため、音声→音声の直接変換ではなく **ASR→TTS カスケード** を採用する。

```
マイク → VAD(発話区間) → ASR(日本語認識) → 文節分割 → TTS(再合成) → 出力
```

変換元の「声質」ではなく「発話内容・話速・音量・ピッチ」を対象声で再現する方式のため、
どんな入力声でも同一の目標声質が得られる。一方で声そのものの音色は転写されない点が
RVC 系直接変換との本質的な差異。

## 2. 構成要素

| 要素 | 採用技術 | 役割 |
|---|---|---|
| キャプチャ/再生 | sounddevice (PortAudio) | 16 kHz mono 入力、48 kHz 出力 |
| VAD | Silero VAD v5 (sherpa-onnx) | 32 ms 窓、発話区間の切り出し |
| ASR | ReazonSpeech Zipformer int8 (sherpa-onnx) | CPU 完結の日本語認識 |
| ピッチ推定 | 自己相関法 (NumPy) | 入力 f0 → pitchScale 追従 |
| TTS | AivisSpeech Engine 1.2.0 (外部プロセス) | Style-Bert-VITS2 ONNX 推論、HTTP API |
| GPU | エンジンの `--use_gpu` | Windows: DirectML (NVIDIA/Radeon/Intel 共通)、CUDA EP 存在時は CUDA 優先。macOS/Linux は CPU |
| GUI | PySide6 (Qt Widgets, ダークテーマ) | デバイス選択・ボイス設定・ミキサー・ライブラリ |
| 凍結 | PyInstaller onedir | Python 不要の単体配布 |

## 3. リアルタイムパイプライン

### スレッド構成

- **キャプチャ**: PortAudio コールバック（32 ms フレーム）
- **ASR ワーカー**: 発話セグメント単位で認識
- **TTS ワーカー**: 文節単位でエンジンへ HTTP 合成
- **再生ワーカー**: 合成音声を再生リングバッファへ移送
- **出力**: PortAudio コールバック（20 ms ブロック）で drain

各段は `queue.Queue`（有界: 発話8/文節32/再生64）で接続し、処理追いつかずの際は
最古の再生待ち音声を破棄して遅延上限（既定 900 ms）を守る。

### 低遅延化

- 文分割（句点）→文節分割（読点/長さ48）で文の途中でも逐次合成開始
- ASR 完了と同時に最初の文節を合成投入（発話終了≠発話全体の待ち）
- VAD `min_silence_ms=450` で発話終端を早期確定
- 入力がネイティブ 16 kHz を受け付けない場合のみ soxr VHQ で再サンプル

### プロソディ転写

発話ごとに計測した値を AudioQuery パラメータへ写像:

| 計測 | 写像先 | 変換 |
|---|---|---|
| 文字数/秒 (cps) | speedScale | `cps/8` → clamp 0.7–1.6 |
| 入力 RMS (dB) | volumeScale | `10^((dB+20)/30)` → clamp 0.5–1.8 |
| 入力 f0 中央値 | pitchScale | 220 Hz 基準の半音差 ×0.0125 → clamp ±0.15 |

各追従は個別に ON/OFF 可能（話速/音量/ピッチ）。

## 4. 機能一覧

### コア
- マイク→対象声へのリアルタイム変換（VAD→ASR→TTS）
- パススルー（変換バイパス、入力そのまま再生）
- モニター出力（第2出力デバイスで自分の変換声を確認）
- 変換音声の録音（`recordings/kowairo-*.wav`）
- テキスト直接読み上げ（変換停止中は Qt Multimedia でプレビュー再生）

### ミキサー / エフェクト（競合機能対応）
- 入力ゲイン 0–200 %
- ノイズゲート（-80〜-20 dB、閾値未満は VAD/パススルー共に無音化）
- 出力ゲイン 0–200 % + tanh ソフトリミッター（クリップ防止）
- 出力エフェクト: エコー / リバーブ / ロボット（リング変調 30 Hz）+ 強さ
  - NumPy ベクトル化、チャンク境界で位相・履歴連続
- ミュート（入力遮断）
- 上記すべて変換実行中にライブ反映

### プリセット
モデル/スタイル・話速・ピッチ・抑揚・音量・追従フラグ・ゲイン・ゲート・
FX・パススルーを名前付きで保存/適用/削除（`settings.json` 内 `extra.presets`）。

### サウンドボード
8 パッド（wav/mp3/flac/ogg/aiff を soundfile でデコード→モノラル化）。
クリックで再生/未設定時は割り当て、右クリックで解除。変換中は出力へ
ミックス、停止中は Qt Multimedia で再生。

### モデル管理
- ローカル .aivmx / .zip（BOOTH 配布物そのまま）導入 → エンジン install API
- AivisHub UUID 直接指定
- オンラインライブラリ: AivisHub 検索・一覧（人気/いいね/新着、ページ送り）、
  詳細（作者/ライセンス/声質/サイズ/スタイル）、公式サンプル試聴、ワンクリック導入

### ショートカット
| キー | 動作 |
|---|---|
| Ctrl+Shift+C | 変換 開始/停止 |
| Ctrl+Shift+P | パススルー切替 |
| Ctrl+Shift+M | ミュート切替 |
| F1〜F8 | サウンドボード再生 |

### 仮想デバイス連携
「出力」に VB-CABLE 等の仮想デバイスを選べば Discord/ゲームのマイクとして
使用可能。デバイス欄のガイドボタンから導入ページを開く。

## 5. 設定 (`settings.json`)

保存場所: `%APPDATA%/Kowairo/settings.json`（Windows）等 platformdirs 準拠。

| キー | 既定 | 意味 |
|---|---|---|
| input_device / output_device / monitor_device | null | デバイス index |
| monitor_enabled | false | モニター出力 |
| style_id / model_uuid | null | 使用モデル |
| speed_scale / pitch_scale / intonation_scale / volume_scale | 1.0 / 0.0 / 1.0 / 1.0 | AudioQuery 基本値 |
| auto_speed / auto_volume / auto_pitch | true / true / false | 入力追従 |
| pitch_semitones | 0.0 | 固定半音オフセット |
| noise_gate_db | -60 | ノイズゲート閾値（-80 で実質無効） |
| input_gain / output_gain | 1.0 | ゲイン倍率 |
| limiter | true | tanh ソフトクリップ |
| fx_mode / fx_amount | off / 0.5 | エコー/リバーブ/ロボット |
| passthrough / muted | false | バイパス/入力遮断 |
| use_gpu | false | エンジン GPU 推論（再起動で反映） |
| vad_threshold / vad_min_silence_ms / vad_min_speech_ms | 0.5 / 450 / 160 | VAD パラメータ |
| max_speech_sec | 15 | 強制区切り |
| drop_when_behind_ms | 900 | 遅延上限（超過分は破棄） |
| asr_num_threads | 2 | ASR スレッド数 |
| record_output | false | 録音 |
| extra.pads / extra.presets | — | パッド割当・プリセット |

設定保存は tmp+replace の原子的書き込み。

## 6. 性能設計（低負荷化）

- VRAM 既定 0（エンジン CPU 推論; `--use_gpu` 時も DirectML 経由で専有抑制）
- ASR は int8 量子化・CPU・2 スレッド
- TTS 推論は外部エンジンプロセスに隔離（アプリ本体は音声 I/O+制御のみ）
- 出力コールバック内は deque drain のみ（推論・ネットワークはワーカー側）
- 想定遅延: 文節先頭の音出しまで ≈ VAD 終端 450 ms + ASR + 初回 TTS 合成

## 7. ディレクトリ構成

```
src/kowairo/
  app.py / __main__.py / cli.py      起動・CLI(ファイル変換モード)
  paths.py                            platformdirs 準拠パス
  settings.py                         JSON 設定
  audio/{pipeline,vad,pitch,fx,devices}.py
  asr/{base,sherpa}.py                ASR 抽象 + ReazonSpeech 実装
  engine/{manager,client,releases}.py エンジン DL/起動/HTTP
  models/{aivm,hub}.py                モデル導入 / AivisHub API
  ui/{run,main_window,workers,theme}.py
  util/{audio,net}.py
launcher.py                           凍結 exe エントリ (crash.log)
kowairo.spec                          PyInstaller spec
scripts/                              ビルド・スモーク
```

## 8. 既知の制約

- 入力言語は日本語（ReazonSpeech）
- 声の音色は転写されない（発話内容/リズムのみ転写）— RVC 方式が必要なら別バックエンド
- グローバルホットキーはウィンドウ非アクティブ時は効かない（QShortcut はアプリ内フォーカス）
- AivisSpeech Engine 1.2.0 は Windows/macOS/Linux x64。ARM Linux は未検証
- エンジン初回 DL 約 1.7 GB・各音声モデルは 0.3–1 GB 程度

## 9. ライセンス・注意

- 本体: MIT（pyproject 参照）
- 音声モデル（BOOTH/AivisHub 配布物）は各モデルのライセンスに従う。
  本アプリはモデルを同梱しない（ユーザーが導入）。
- AivisSpeech Engine はエンジン側の規約に従う。
