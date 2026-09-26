# Kowairo（声色）

**Kowairo** は、[Style-Bert-VITS2 / AivisSpeech 形式の AI 音声モデル](https://booth.pm/ja/items/8442719)（ダウナー系少女音声モデルほか任意の AIVMX モデル）を使った、**専用リアルタイム・ボイスチェンジャー**です。

話した内容をリアルタイムに音声認識（ASR）し、対象音声モデルで即座に再合成（TTS）して出力します。

```
マイク入力 → Silero VAD（発話区間検出）→ ReazonSpeech ASR（日本語認識）
          → 文節分割 → AivisSpeech Engine（Style-Bert-VITS2 ONNX）→ 音声出力
```

> **なぜ ASR→TTS 型か**: 本モデルは音声→音声の変換モデル（RVC 等）ではなく **テキスト→音声の TTS モデル**です。
> 声色変換のために、入力音声から一旦テキストを取り出し、対象の声で読み上げ直す方式を採っています。
> その代わりに、**声質はモデルそのまま（最大限の品質）**、**VRAM 使用量ほぼゼロでも動作**します。

## 特徴

- **高品質・高音質**: Style-Bert-VITS2 (JP-Extra) による感情豊かな 44.1/48 kHz 音声合成。出力は 48 kHz モノラル。
- **低 VRAM / 低負荷**: 既定は完全 CPU 動作（VRAM 0 MB）。ASR は int8 量子化 Zipformer（~160MB）、TTS は ONNX Runtime。
- **高リアルタイム性**: 発話終了から文節単位で合成を先回りして逐次再生。発話終了→発声開始まで実測 ~1 秒前後（CPU）。
- **GPU 対応（オプション）**: エンジンの `--use_gpu` で Windows は DirectML（NVIDIA/AMD Radeon/Intel すべて対象）、CUDA EP が存在する環境では CUDA を優先利用。macOS/Linux は CPU 実行（Style-Bert-VITS2 の ONNX は CPU でも実時間より十分高速です）。
- **スタンドアロン GUI**: PySide6 製デスクトップアプリ。PyInstaller で Windows/macOS/Linux の単一実行ファイルにパッケージ可能。
- **ASR 結果のプロソディ反映**: 入力の話速・音量・（任意で）ピッチを合成パラメータへ自動マッピング。

## 動作環境

| OS | TTS | ASR | 状態 |
|---|---|---|---|
| Windows 10/11 x64 | CPU / DirectML GPU | CPU | 本機開発・検証済み |
| macOS (arm64 / x64) | CPU（Metal 相当の高速化はエンジン側の将来対応） | CPU | ビルド提供 |
| Linux x64 / arm64 | CPU | CPU | ビルド提供 |

> NVIDIA CUDA を Linux で使いたい場合は、同梱エンジンをソース/Docker(nvidia) 版に差し替えるか、`faster-whisper` ASR バックエンドを選択してください（`pip install .[whisper]`）。

## セットアップ

```bash
# Python 3.10–3.13 が必要
pip install .
kowairo           # GUI 起動
```

初回起動時に以下を自動ダウンロードします（`~/.local/share/Kowairo` など OS 標準のデータ dir）:

1. **AivisSpeech Engine**（~200–340MB、OS 別）
2. **音声認識モデル** ReazonSpeech Zipformer int8（~160MB）+ Silero VAD（~2MB）
3. **音声モデル**: 「モデル管理」タブから以下のいずれかで導入
   - BOOTH で入手した `downer.zip`（または解凍済み `.aivmx`）を選択
   - AivisHub のモデル UUID を入力して直接ダウンロード

### BOOTH モデルの入手

1. https://booth.pm/ja/items/8442719 を開き、無料ダウンロード（要 Pixiv/BOOTH ログイン）
2. `downer.zip` をそのまま Kowairo の「モデル管理」→「ファイルから導入」で選択
3. zip 内の `.aivmx` が自動で取り出され、エンジンにインストールされます

## 使い方

1. 「入力 (マイク)」にマイク、「出力」に仮想オーディオデバイス（VB-CABLE / BlackHole 等）かスピーカーを選択
2. 「モデル / スタイル」でダウナー系少女モデルを選択
3. **▶ 変換開始** → マイクに話しかけると、対象の声で読み返されます
4. Discord 等で使うには、出力先を仮想デバイスにし、アプリ側の入力デバイスをその仮想デバイスに設定

遅延の目安: 発話終了（無音 ~0.45s 検出）→ ASR ~0.2–0.6s → TTS ~0.2–0.5s → 再生開始。

## CLI / テストモード

```bash
# テキスト読み上げ
kowairo --cli --text "こんにちは、世界" --output out.wav

# WAV → 変換 → WAV（音声認識→再合成の全経路を検証）
kowairo --cli --input in.wav --output converted.wav --model downer.aivmx

# AivisHub モデルで試す
kowairo --cli --hub-uuid <uuid> --text "テストです" --output t.wav
```

## 開発者向け

```bash
# 開発環境
pip install -e ".[dev]"
ruff check src/
python scripts/gui_smoke.py        # GUI ヘッドレス起動テスト

# スタンドアロン exe / app ビルド（出力: dist/Kowairo/）
scripts/build_windows.ps1          # Windows (PowerShell)
./scripts/build_posix.sh           # macOS / Linux
# または直接: pyinstaller kowairo.spec --noconfirm
```

## アーキテクチャ

| 層 | 役割 | 技術 |
|---|---|---|
| キャプチャ | 16 kHz マイク入力 | sounddevice (PortAudio) |
| VAD | 発話区間の切り出し | Silero VAD (ONNX, sherpa-onnx) |
| ASR | 日本語音声認識 | ReazonSpeech Zipformer int8 (sherpa-onnx, CPU) |
| 文節化 | 句点・読点で分割して先回り合成 | kowairo 内部 |
| TTS | テキスト→音声 | AivisSpeech Engine（同梱, ONNX Runtime, CPU/DirectML/CUDA） |
| 再生 | 48 kHz 出力 + 監視 + 録音 | sounddevice |
| GUI | デバイス選択・パラメータ・モデル管理 | PySide6 |

## ライセンス / クレジット

- Kowairo 本体: MIT
- AivisSpeech Engine: LGPL-3.0（同梱バイナリとして使用）
- 音声モデル（BOOTH「ダウナー系少女のAI音声モデル」、作者: 零音ほのか）: 商用可・クレジット不要・二次利用可（モデル同梱の README.md を参照）
- ReazonSpeech: Reazon Holdings — speech corpus/model license を参照
- Silero VAD: MIT

## 既知の制約

- ASR→TTS 型のため、**声の抑揚・感情は完全には引き継がれません**（話速・音量・ピッチは概ね反映）。
- 日本語専用の認識・合成です。
- 発話中のリアルタイム逐次変換ではなく、発話区間（数秒）ごとの変換です。
