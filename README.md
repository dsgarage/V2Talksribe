# Talksribe

macOS menubar real-time transcription app.

## Canonical Entry Points

- Development run: `bash scripts/run_dev.sh`
  - Runs the current source code in `src/menubar.py`
- Packaged app: `bash scripts/open_release.sh`
  - Opens `dist/Talksribe.app`
  - Use this only after rebuilding the package
- Source launcher app: `apps/source-launcher/Talksribe.app`
  - Thin wrapper that launches the same `src/menubar.py`
  - Useful only if you explicitly want a clickable app for the current source tree

## What Is Actually Running

- `src/menubar.py`
  - Thin menubar UI layer (rumps). Delegates all audio work to the engine.
- `src/engine.py`
  - `TranscriptionEngine`: owns all live-audio state in a single worker thread.
    UI communicates only via a command queue (in) and event queue (out).
- `src/transcriber.py`
  - Faster-Whisper wrapper with hallucination guards (used by live and file paths).
- `src/vad.py`
  - Unified VAD (webrtc / silero) with an RMS energy gate.
- `src/settings.py`
  - App settings + persistence to `~/Documents/Talksribe/config.json`.
- `src/file_transcribe.py`
  - File transcription (own transcriber, long-form friendly).
- `src/capture.py`
  - Child process for microphone capture.
- `talksribe.spec`
  - PyInstaller packaging definition for `dist/Talksribe.app`
- `scripts/install_source_launcher.sh`
  - Builds a local wrapper app that launches the source tree directly

## Repository Layout

```text
.
├── src/                        # Active application source
├── scripts/                    # Run/build helper scripts
├── dist/                       # PyInstaller release output
├── build/                      # PyInstaller working files
├── apps/source-launcher/       # Clickable app wrapper for current source
├── samples/zoom/               # Sample recordings and transcript outputs
├── experiments/                # Verification / one-off investigation work
├── legacy/py2app/              # Archived py2app setup, not used
├── talksribe.spec              # Release build spec
├── requirements.txt
└── README.md
```

## Common Commands

### Run current source

```bash
bash scripts/run_dev.sh
```

### Build packaged release

```bash
bash scripts/build_release.sh
```

### Open packaged release

```bash
bash scripts/open_release.sh
```

### Install source launcher into `/Applications`

```bash
bash scripts/install_source_launcher.sh
```

## Notes

- If behavior differs between source and app, check whether you launched `scripts/run_dev.sh` or `dist/Talksribe.app`.
- `dist/Talksribe.app` does not update automatically when you edit `src/`.
- `apps/source-launcher/Talksribe.app` always runs the current files under this repository.
- `legacy/py2app/` is kept only for reference.
- The menubar app now has an `音声前処理` menu with `ノーマライズ` and `ノイズ除去` toggles.
- File transcription uses ffmpeg-based preprocessing when enabled; live microphone input uses a lightweight realtime preprocessor.
