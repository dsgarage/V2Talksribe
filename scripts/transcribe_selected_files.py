#!/usr/bin/env python3
from __future__ import annotations

import sys
import threading
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from file_transcribe import FileTranscriber  # noqa: E402
from settings import SAVE_DIR, AppSettings  # noqa: E402


def _emit(*event):
    action = event[0]
    if action in ("status", "detail"):
        print(f"  {event[1]}")


def main(argv: list[str]) -> int:
    if not argv:
        print("usage: transcribe_selected_files.py <media-file> [<media-file> ...]", file=sys.stderr)
        return 2

    files = [Path(arg).expanduser() for arg in argv]
    missing = [path for path in files if not path.exists()]
    if missing:
        for path in missing:
            print(f"missing: {path}", file=sys.stderr)
        return 1

    settings = AppSettings()
    settings.language = "en"
    settings.model_size = "base"

    cancel = threading.Event()
    transcriber = FileTranscriber(settings, _emit, cancel)

    print(f"save_dir={SAVE_DIR}")
    print(f"count={len(files)}")

    for index, path in enumerate(files, start=1):
        print(f"[{index}/{len(files)}] start {path}")
        count, cancelled, out_path = transcriber.transcribe(str(path))
        print(f"[{index}/{len(files)}] done  {path} → {out_path} ({count}件)")

    print("completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
