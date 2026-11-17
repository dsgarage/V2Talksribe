#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""録音 WAV の逐次書き込みと、古い録音の整理。"""
from __future__ import annotations

import glob
import os
import time
from typing import Callable, Optional

import numpy as np
import soundfile as sf

import applog

FLUSH_SEC = 1.0


class WavRecorder:
    """16bit mono WAV へ逐次書く。定期的に flush し、強制終了でも読める状態を保つ。"""

    def __init__(self, path: str, sr: int):
        self.path = path
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self._f = sf.SoundFile(path, "w", samplerate=sr, channels=1,
                               format="WAV", subtype="PCM_16")
        self._last_flush = time.monotonic()

    def write(self, audio: np.ndarray) -> None:
        self._f.write(np.asarray(audio, dtype=np.float32))
        now = time.monotonic()
        if now - self._last_flush >= FLUSH_SEC:
            self._f.flush()
            self._last_flush = now

    def close(self) -> None:
        try:
            self._f.flush()
            self._f.close()
        except Exception as e:
            applog.log(f"WavRecorder.close failed: {e}")


def session_part_path(audio_dir: str, session_ts: str, part: int) -> str:
    return os.path.join(audio_dir, f"{session_ts}_{part:02d}.wav")


def move_to_trash(path: str) -> bool:
    """ファイルをゴミ箱へ移す。成功したら True。"""
    try:
        from Foundation import NSFileManager, NSURL

        ok, _url, err = NSFileManager.defaultManager().trashItemAtURL_resultingItemURL_error_(
            NSURL.fileURLWithPath_(path), None, None)
        if not ok:
            applog.log(f"move_to_trash failed: {path}: {err}")
        return bool(ok)
    except Exception as e:
        applog.log(f"move_to_trash failed: {path}: {e}")
        return False


def prune_recordings(audio_dir: str, *, max_age_days: float = 14,
                     max_total_bytes: int = 5 * 1024 ** 3,
                     protect_prefix: Optional[str] = None,
                     now: Optional[float] = None,
                     trash: Callable[[str], bool] = move_to_trash) -> list:
    """古い録音を、期限超過 → 容量超過(古い順)の順でゴミ箱へ移す。移したパスを返す。"""
    if not os.path.isdir(audio_dir):
        return []
    now = time.time() if now is None else now
    entries = []
    for p in glob.glob(os.path.join(audio_dir, "*.wav")):
        try:
            st = os.stat(p)
        except OSError:
            continue
        entries.append((st.st_mtime, st.st_size, p))
    entries.sort()

    def protected(p: str) -> bool:
        return bool(protect_prefix) and os.path.basename(p).startswith(protect_prefix)

    moved = []
    remaining = []
    cutoff = now - max_age_days * 86400
    for mtime, size, p in entries:
        if mtime < cutoff and not protected(p) and trash(p):
            moved.append(p)
        else:
            remaining.append((mtime, size, p))

    total = sum(size for _m, size, _p in remaining)
    for _mtime, size, p in remaining:
        if total <= max_total_bytes:
            break
        if protected(p):
            continue
        if trash(p):
            moved.append(p)
            total -= size
    return moved
