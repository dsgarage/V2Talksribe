# talksribe_cli

Mac向けCUIの話者分離つき文字起こしツール。ローカル環境で完結する、プライバシーに配慮した音声文字起こしアプリケーションです。

## 概要

talksribeは、音声ファイルから話者を自動識別し、発言ごとに文字起こしを行うCLIツールです。外部APIを使用せず、すべての処理をローカルで実行するため、機密性の高い音声データも安全に処理できます。

## クイックスタート

3ステップで音声文字起こしを始められます：

### 1. インストール

```bash
# リポジトリをクローン
git clone https://github.com/dsgarage/V2Talksribe.git
cd V2Talksribe

# 仮想環境を作成して有効化
python -m venv .venv
source .venv/bin/activate

# 依存関係をインストール
pip install -r requirements.txt
pip install 'torch==2.0.1' 'torchaudio==2.0.2' 'numpy<2'
```

### 2. 実行

```bash
# 音声ファイルを文字起こし
python src/talksribe.py \
  --audio your_audio.wav \
  --model-size small \
  --language ja
```

### 3. 結果確認

出力ファイルは `./outputs` ディレクトリに生成されます：

- `*.json` - 完全な文字起こしデータ（話者情報、タイムスタンプ含む）
- `*.srt` - 動画字幕ファイル
- `*.vtt` - Web動画字幕ファイル

詳細な使い方は[Wiki](https://github.com/dsgarage/V2Talksribe/wiki)を参照してください。

### 主な機能

- **発話区間検出（VAD）**: WebRTC VADを使用して音声区間を自動検出
- **話者分離（Diarization）**: ECAPA-TDNNによる話者埋め込みとクラスタリングで話者を識別
- **自動文字起こし（ASR）**: faster-whisperによる高精度な音声認識（日本語対応）
- **複数フォーマット出力**: JSON、SRT、VTT形式での出力に対応
- **長時間録音対応**: 1時間程度の録音も堅牢に処理可能
- **話者数自動推定**: Silhouette法による最適な話者数の自動推定

## アーキテクチャ

talksribeは以下の4つのステージで音声を処理します：

```
音声入力 → VAD → 話者埋め込み → クラスタリング → ASR → 出力
```

### 処理フロー

1. **音声読み込み**: soundfileで任意の音声形式を読み込み、必要に応じてリサンプリング
2. **VAD（Voice Activity Detection）**: WebRTC VADで発話区間を検出
3. **話者埋め込み**: SpeechBrainのECAPA-TDNNモデルで各発話区間の話者特徴を抽出
4. **クラスタリング**: scikit-learnの階層的クラスタリングで話者をグループ化
5. **ASR（Automatic Speech Recognition）**: faster-whisperで各発話区間を文字起こし
6. **統合・出力**: 話者情報とテキストを統合してJSON/SRT/VTT形式で出力

## 使用ライブラリ

### コア機能

| ライブラリ | バージョン | 役割 |
|-----------|-----------|------|
| **numpy** | <2.0 | 数値計算の基盤。音声データの配列操作に使用 |
| **soundfile** | latest | 音声ファイルの読み書き。WAV、AIFF、M4Aなど多様な形式に対応 |
| **librosa** | latest | 音声信号処理。リサンプリングなどの前処理に使用 |

### VAD（発話区間検出）

| ライブラリ | バージョン | 役割 |
|-----------|-----------|------|
| **webrtcvad-wheels** | latest | WebRTC Voice Activity Detector。高精度な音声区間検出 |

### 話者分離

| ライブラリ | バージョン | 役割 |
|-----------|-----------|------|
| **speechbrain** | 1.0.3 | ECAPA-TDNNモデルによる話者埋め込みベクトル抽出 |
| **scikit-learn** | latest | 階層的クラスタリングとSilhouetteスコア計算 |
| **torch** | 2.0.1 | PyTorchディープラーニングフレームワーク |
| **torchaudio** | 2.0.2 | 音声処理用PyTorch拡張 |

### 音声認識（ASR）

| ライブラリ | バージョン | 役割 |
|-----------|-----------|------|
| **faster-whisper** | latest | OpenAI Whisperの高速実装。CTranslate2ベース |

### UI/UX

| ライブラリ | バージョン | 役割 |
|-----------|-----------|------|
| **typer** | latest | モダンなCLIインターフェース構築 |
| **rich** | latest | 美しいターミナル出力とテーブル表示 |
| **tqdm** | latest | 進捗バー表示 |

### データ管理

| ライブラリ | バージョン | 役割 |
|-----------|-----------|------|
| **pydantic** | 2.x | データバリデーションとモデル定義 |

## セットアップ

### 必要要件

- Python 3.11以降
- macOS（Apple Silicon対応）
- 2GB以上の空きメモリ（モデルサイズによる）

### インストール手順

```bash
# 仮想環境の作成
python -m venv .venv

# 仮想環境の有効化
source .venv/bin/activate

# 依存関係のインストール
pip install -r requirements.txt

# バージョン互換性のため以下も実行（推奨）
pip install 'torch==2.0.1' 'torchaudio==2.0.2' 'numpy<2'
```

**Note**: Apple Siliconで`torch/torchaudio`を最適化する場合は、[PyTorch公式](https://pytorch.org/)のインストール手順に従ってください。

## 使い方

### 基本的な使用例

```bash
python src/talksribe.py \
  --audio "your_audio.wav" \
  --model-size "medium" \
  --language "ja" \
  --outdir ./outputs
```

### オプション一覧

```
必須オプション:
  --audio TEXT              入力音声ファイルパス（WAV/AIFF/M4A等）

出力設定:
  --outdir TEXT            出力ディレクトリ [default: ./outputs]

ASRモデル設定:
  --model-size TEXT        Whisperモデルサイズ
                          tiny/base/small/medium/large-v2
                          [default: small]
  --language TEXT          言語コード（ja/en等）[default: ja]
  --beam-size INTEGER      ASRビーム幅 [default: 5]
  --compute-type TEXT      計算精度（int8/float16/float32）
                          [default: int8]

VAD設定:
  --vad-aggr INTEGER       VAD感度（0-3、大きいほど厳しい）
                          [default: 2]

話者分離設定:
  --max-speakers INTEGER   自動推定時の最大話者数 [default: 6]
  --num-speakers INTEGER   話者数を固定する場合に指定

デバイス設定:
  --device TEXT            使用デバイス（cpu推奨）[default: cpu]
```

### 使用例

#### 会議の文字起こし（話者数自動推定）

```bash
python src/talksribe.py \
  --audio meeting.wav \
  --model-size medium \
  --language ja
```

#### インタビューの文字起こし（話者数固定）

```bash
python src/talksribe.py \
  --audio interview.m4a \
  --model-size small \
  --language ja \
  --num-speakers 2
```

#### 英語音声の文字起こし

```bash
python src/talksribe.py \
  --audio podcast.wav \
  --model-size large-v2 \
  --language en \
  --max-speakers 4
```

## 出力フォーマット

### JSON形式

完全な文字起こし結果と話者情報を含む構造化データ：

```json
{
  "audio_path": "/path/to/audio.wav",
  "sample_rate": 48000,
  "duration": 125.5,
  "language": "ja",
  "speakers": ["SPK_1", "SPK_2"],
  "utterances": [
    {
      "start": 0.5,
      "end": 3.2,
      "text": "こんにちは",
      "speaker": "SPK_1",
      "words": [...]
    }
  ]
}
```

### SRT形式

動画字幕に適した形式：

```srt
1
00:00:00,500 --> 00:00:03,200
SPK_1: こんにちは

2
00:00:03,500 --> 00:00:06,100
SPK_2: よろしくお願いします
```

### VTT形式

Web動画字幕用：

```vtt
WEBVTT

00:00:00.500 --> 00:00:03.200
SPK_1: こんにちは

00:00:03.500 --> 00:00:06.100
SPK_2: よろしくお願いします
```

## モデルサイズと性能

| モデル | パラメータ数 | メモリ使用量 | 処理速度 | 精度 |
|--------|-------------|-------------|---------|------|
| tiny   | 39M         | ~1GB        | 最速    | 低   |
| base   | 74M         | ~1GB        | 速い    | 中   |
| small  | 244M        | ~2GB        | 普通    | 良   |
| medium | 769M        | ~5GB        | やや遅い | 高   |
| large-v2| 1550M      | ~10GB       | 遅い    | 最高 |

**推奨**: 一般的な用途には`small`または`medium`が最適です。

## トラブルシューティング

### 発話区間が検出されない

- `--vad-aggr` の値を下げてください（0に近づけるほど検出が緩くなります）
- 入力音声のサンプリングレートと品質を確認してください

### 話者分離の精度が低い

- `--num-speakers` で話者数を明示的に指定してください
- 音声品質が低い場合、話者分離の精度は低下します

### メモリ不足エラー

- より小さいモデルサイズを使用してください（`--model-size tiny` または `base`）
- `--compute-type int8` を使用してください（既定値）

## ライセンス

このプロジェクトは各種オープンソースライブラリを使用しています。
各ライブラリのライセンスについては、それぞれのドキュメントを参照してください。

## 謝辞

- [faster-whisper](https://github.com/guillaumekln/faster-whisper): 高速なWhisper実装
- [SpeechBrain](https://speechbrain.github.io/): 話者認識モデル
- [WebRTC VAD](https://webrtc.org/): 音声区間検出
