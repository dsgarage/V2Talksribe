# Faster-Whisper Verification Kit

Minimal, robust way to *prove* Faster-Whisper works on a ~1h lecture.
- No diarization, no VAD — fewer moving parts.
- Writes **JSON / SRT / VTT** and prints speed metrics (RTF).
- **Language selection verification** - Demonstrates critical importance of `--language` parameter.

## Quick Start (macOS)
```bash
chmod +x verify_faster_whisper.sh
./verify_faster_whisper.sh /path/to/lecture.wav
# Optional: DEVICE=mps or cuda, COMPUTE_TYPE=int8_float16
# e.g., DEVICE=mps COMPUTE_TYPE=int8_float16 ./verify_faster_whisper.sh /path.wav
```

## Manual run
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip wheel
pip install faster-whisper==1.* numpy soundfile librosa tqdm typer rich pydantic==2.*

python verify_faster_whisper.py \
  --audio /path/to/lecture.wav \
  --outdir ./outputs \
  --model-size small \
  --language ja \
  --device cpu \
  --compute-type int8 \
  --beam-size 5
```

### Preprocess tip (optional, to avoid weird encodings)
```bash
ffmpeg -i lecture.wav -ar 16000 -ac 1 -acodec pcm_s16le lecture_16k.wav
```

### Interpreting the result
- Check the console lines for `Elapsed`, `Audio duration`, and `RTF`.
- Inspect `./outputs/*.srt` in a player (VLC/QuickTime + Subler) or `*.vtt` in the browser.

### Troubleshooting
- If model download is slow, prefetch to cache:
  - macOS: `~/.cache/ctranslate2`
- Memory: if you get OOM, try `--compute-type int8` (default) or smaller models (`tiny`, `base`).

## Language Selection - Critical Importance

**The `--language` parameter is the most critical setting for transcription quality.**

### Test Results (ZOOM0021.WAV - 37.4 min conference)

| Metric | Japanese Mode (Wrong) | English Mode (Correct) | Impact |
|--------|----------------------|------------------------|--------|
| Processing Time | 17.20 min | 6.20 min | **3x faster** |
| RTF | 0.46 | 0.17 | **-63%** |
| Quality | Unusable (garbled) | Production-ready | Critical |

### Quality Example

**English audio processed with wrong language setting:**
```
Japanese Mode: "本日もおりますが、このセベントの最後のキャンペーンの..."
English Mode:  "Thank you for joining us at the last panel of the..."
```

### Recommendation for TalkScribe

```python
# Required parameter - no default
--language {en,ja}  # Force user to specify

# Example usage
python verify_faster_whisper.py --audio conference.wav --language en --outdir ./outputs
```

See `language_comparison_report.md` for detailed analysis.

## Directory Structure

```
faster_whisper_verify/
├── verify_faster_whisper.py    # Main transcription script
├── verify_preprocess.py         # Audio preprocessing (16kHz conversion)
├── verify_run.sh                # All-in-one execution script
├── verify_faster_whisper.sh     # Simple wrapper script
├── README.md                    # This file
├── language_comparison_report.md # Language selection analysis
├── outputs/                     # Japanese mode results (example)
│   ├── lecture_16k.wav         # Preprocessed audio
│   ├── lecture_16k.json        # Transcription segments
│   ├── lecture_16k.srt         # Subtitle format
│   └── lecture_16k.vtt         # Web video text tracks
├── outputs_en/                  # English mode results (example)
│   ├── lecture_16k.json
│   ├── lecture_16k.srt
│   └── lecture_16k.vtt
└── .venv/                       # Python virtual environment
```

