#!/usr/bin/env bash
set -euo pipefail

if [ -z "${AUDIO:-}" ]; then
  echo "AUDIO が未設定です。例: AUDIO=\"/path/to/lecture.wav\""
  exit 1
fi

# venv & 依存（最小）
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip wheel --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host files.pythonhosted.org
pip install --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host files.pythonhosted.org faster-whisper==1.* numpy soundfile librosa tqdm typer rich pydantic==2.*

# 16k/mono へ正規化（ffmpeg 優先、無ければ Python で）
mkdir -p outputs
PREP="./outputs/lecture_16k.wav"
python verify_preprocess.py --in "$AUDIO" --out "$PREP"

# 速度優先パラメータ
DEVICE="${FW_DEVICE:-cpu}"           # Macはcpuが安定。Metalを試す場合は FW_DEVICE=metal
CTYPE="${FW_COMPUTE:-int8}"          # int8 が最速/低メモリ
MODEL="${FW_MODEL:-small}"           # small で十分検証可能
BEAM="${FW_BEAM:-3}"                 # 小さめで高速

python verify_faster_whisper.py \
  --audio "$PREP" \
  --outdir ./outputs \
  --model-size "$MODEL" \
  --language "ja" \
  --device "$DEVICE" \
  --compute-type "$CTYPE" \
  --beam-size "$BEAM"

# 成果物をZip
zip -r verify_artifacts.zip outputs
echo "Artifacts: verify_artifacts.zip"
