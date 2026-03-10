#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="/Users/daisuketsukada/.pyenv/versions/3.11.0/bin/python3"

cd "$ROOT_DIR"
exec "$PYTHON" src/menubar.py
