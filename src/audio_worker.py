#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""別プロセスでマイク音声をキャプチャし、メインプロセスに送る。

macOS 26 ではマイクを使用するプロセスのメニューバーアイコンが
マイク表示に置き換わるため、音声キャプチャを別プロセスに分離する。
"""
from __future__ import annotations

import multiprocessing as mp
import signal
import sys
import time

import numpy as np


def _capture_loop(
    audio_pipe: mp.Connection,
    stop_flag: mp.Event,
    sr: int,
    chunk_sec: float,
    device_id,
):
    """子プロセスで実行。sounddevice でマイク入力を取得し pipe で親へ送る。"""
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    import sounddevice as sd

    chunk_samples = int(sr * chunk_sec)
    buf = mp.Queue()

    def audio_cb(indata, frames, time_info, status):
        buf.put(indata[:, 0].copy().tobytes())

    stream = sd.InputStream(
        samplerate=sr, channels=1, dtype="float32",
        blocksize=chunk_samples, device=device_id,
        callback=audio_cb,
    )
    stream.start()

    try:
        while not stop_flag.is_set():
            try:
                raw = buf.get(timeout=0.5)
                audio_pipe.send_bytes(raw)
            except Exception:
                pass
    finally:
        stream.stop()
        stream.close()
        audio_pipe.close()


def start_capture(sr: int, chunk_sec: float, device_id=None):
    """キャプチャ子プロセスを開始。(pipe, stop_flag, process) を返す。"""
    parent_conn, child_conn = mp.Pipe(duplex=False)
    stop_flag = mp.Event()
    proc = mp.Process(
        target=_capture_loop,
        args=(child_conn, stop_flag, sr, chunk_sec, device_id),
        daemon=True,
    )
    proc.start()
    child_conn.close()
    return parent_conn, stop_flag, proc


def stop_capture(stop_flag: mp.Event, proc: mp.Process):
    """キャプチャ子プロセスを停止。"""
    stop_flag.set()
    proc.join(timeout=3)
    if proc.is_alive():
        proc.kill()
