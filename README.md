# talksribe_cli

Mac向けCUIの話者分離つき文字起こしツール。

## セットアップ

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Apple Siliconで`torch/torchaudio`を最適化する場合は、PyTorch公式のインストール手順に従ってください。

## 使い方

```bash
python src/talksribe.py transcribe   --audio "your_audio.wav"   --model-size "medium"   --language "ja"   --max-speakers 6   --outdir ./outputs
```

- 入力は48kHz/32-bit float WAVに対応（内部で安全に変換）
- 出力：JSON / SRT / VTT
