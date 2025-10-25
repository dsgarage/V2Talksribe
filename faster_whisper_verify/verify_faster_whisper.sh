#!/usr/bin/env bash
set -euo pipefail

AUDIO="${1:-}"
if [ -z "$AUDIO" ]; then
  echo "Usage: ./verify_faster_whisper.sh /path/to/lecture.wav"
  exit 1
fi

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip wheel
pip install faster-whisper==1.* numpy soundfile librosa tqdm typer rich pydantic==2.*

python verify_faster_whisper.py \
  --audio "$AUDIO" \
  --outdir ./outputs \
  --model-size small \
  --language ja \
  --device ${DEVICE:-cpu} \
  --compute-type ${COMPUTE_TYPE:-int8} \
  --beam-size 5
