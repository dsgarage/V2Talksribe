#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""音声/動画ファイルを 1 本文字起こしする CLI ラッパー。

ライブ経路(engine)とは独立した ``FileTranscriber`` を呼び出す。
結果は ``~/Documents/Talksribe/{時刻}_{名前}.txt`` に保存される。

使い方:
    python3 scripts/transcribe_file.py <ファイルパス> [--model base] [--lang ja]
"""
from __future__ import annotations

import argparse
import os
import sys
import threading

# src/ を import パスに追加（menubar.py と同じ流儀）
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from file_transcribe import FileTranscriber  # noqa: E402
from settings import AppSettings  # noqa: E402


def _emit(kind, *args):
    """UI コールバックの代わりに標準出力へ流す。"""
    if kind in ("status", "detail"):
        print(f"[{kind}] {args[0]}")
    elif kind == "notify":
        # ("Talksribe", title, message)
        print(f"[notify] {args[-1]}")


def main() -> int:
    ap = argparse.ArgumentParser(description="ファイル文字起こし")
    ap.add_argument("filepath", help="音声/動画ファイルのパス")
    ap.add_argument("--model", default=None, help="モデルサイズ (tiny/base/small/...)")
    ap.add_argument("--lang", default=None, help="言語コード (ja/en/...)。省略で自動検出")
    args = ap.parse_args()

    if not os.path.exists(args.filepath):
        print(f"ファイルが見つかりません: {args.filepath}", file=sys.stderr)
        return 1

    settings = AppSettings.load()
    if args.model:
        settings.model_size = args.model
    if args.lang:
        settings.language = args.lang

    ft = FileTranscriber(settings, _emit, threading.Event())
    count, cancelled, out_path = ft.transcribe(args.filepath)
    print(f"\n件数={count} 中断={cancelled}\n出力={out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
