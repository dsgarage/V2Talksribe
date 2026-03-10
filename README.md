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
  - Main menubar app
- `src/audio_worker.py`
  - Child process for microphone capture
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
