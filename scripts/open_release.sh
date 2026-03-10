#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
APP_PATH="$ROOT_DIR/dist/Talksribe.app"

if [ ! -d "$APP_PATH" ]; then
    echo "Missing packaged app: $APP_PATH" >&2
    echo "Run: bash scripts/build_release.sh" >&2
    exit 1
fi

open "$APP_PATH"
