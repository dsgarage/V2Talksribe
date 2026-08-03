#!/usr/bin/env python3
from __future__ import annotations

import shutil
import sys
import threading
from pathlib import Path

ROOT = Path("/Users/daisuketsukada/Documents/dsgarageBlog/TechBlog/stock/GDC2026")
SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from file_transcribe import FileTranscriber  # noqa: E402
from settings import AppSettings  # noqa: E402


def _emit(*event):
    action = event[0]
    if action in ("status", "detail"):
        print(f"  {event[1]}")


def find_pending_audio():
    pending: list[Path] = []
    for audio_path in sorted(ROOT.glob("*/source/audio/DJI_*.WAV")):
        transcript_dir = audio_path.parents[1] / "transcripts"
        transcript_dir.mkdir(parents=True, exist_ok=True)
        matches = list(transcript_dir.glob(f"*{audio_path.stem}*.txt"))
        if matches:
            continue
        pending.append(audio_path)
    return pending


def main() -> int:
    pending = find_pending_audio()
    print(f"pending={len(pending)}")
    for audio_path in pending:
        print(audio_path)

    if not pending:
        return 0

    settings = AppSettings()
    settings.language = "en"
    settings.model_size = "base"

    cancel = threading.Event()
    transcriber = FileTranscriber(settings, _emit, cancel)

    for index, audio_path in enumerate(pending, start=1):
        print(f"[{index}/{len(pending)}] transcribing {audio_path}")
        count, cancelled, out_path = transcriber.transcribe(str(audio_path))
        output_path = Path(out_path) if out_path else None
        if output_path is None or not output_path.exists():
            print(f"failed: output not found for {audio_path.name}", file=sys.stderr)
            continue

        target_dir = audio_path.parents[1] / "transcripts"
        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = target_dir / output_path.name
        if target_path.exists():
            print(f"exists: {target_path}")
            continue

        shutil.move(str(output_path), str(target_path))
        print(f"saved: {target_path} ({count}件)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
