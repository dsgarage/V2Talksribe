#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""別プロセスでマイク音声をキャプチャし、メインプロセスへ pipe で送る。

macOS 26 ではマイクを使うプロセスのメニューバーアイコンがマイク表示に
置き換わるため、キャプチャを別プロセスに分離している。

旧実装は「audio callback → mp.Queue → 送信ループ → pipe」と、同一プロセス内で
mp.Queue（pickle＋フィーダースレッド）を挟む二重バッファだった。ここでは
プロセス内のスレッドキュー(queue.Queue)に置き換え、余計な pickle と
フィーダースレッドを排して軽量化する。バウンド超過時は古いチャンクを捨てて
メモリ膨張と遅延の累積を防ぐ。
"""
from __future__ import annotations

import multiprocessing as mp
import os
import queue
import signal
import time

LOG_PATH = os.path.expanduser("~/Documents/Talksribe/debug.log")
# 送信待ちの上限。マイクは 1 チャンク数百ms なので 64 で十分な余裕。
_MAX_PENDING_CHUNKS = 64


def _log(msg: str):
    try:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(f"[{time.strftime('%H:%M:%S')}] {msg}\n")
    except OSError:
        pass


def _capture_loop(audio_pipe, stop_flag, sr: int, chunk_sec: float, device_id):
    """子プロセスで実行。sounddevice でマイク入力を取得し pipe で親へ送る。"""
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    import sounddevice as sd

    chunk_samples = int(sr * chunk_sec)
    pending: "queue.Queue" = queue.Queue(maxsize=_MAX_PENDING_CHUNKS)

    def audio_cb(indata, frames, time_info, status):
        if status:
            _log(f"capture audio_cb status: {status}")
        data = indata[:, 0].copy().tobytes()
        try:
            pending.put_nowait(data)
        except queue.Full:
            # 親の読み取りが詰まっている: 最古を捨てて最新を残す
            try:
                pending.get_nowait()
            except queue.Empty:
                pass
            try:
                pending.put_nowait(data)
            except queue.Full:
                pass

    stream = sd.InputStream(
        samplerate=sr, channels=1, dtype="float32",
        blocksize=chunk_samples, device=device_id,
        callback=audio_cb,
    )
    stream.start()
    _log(f"capture start: pid={os.getpid()} device={device_id} sr={sr} chunk={chunk_sec}")

    try:
        while not stop_flag.is_set():
            try:
                raw = pending.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                audio_pipe.send_bytes(raw)
            except (BrokenPipeError, EOFError, OSError) as e:
                _log(f"capture pipe error: {type(e).__name__}: {e}")
                break
    finally:
        try:
            stream.stop()
            stream.close()
        except Exception as e:
            _log(f"capture stream cleanup error: {e}")
        try:
            audio_pipe.close()
        except Exception as e:
            _log(f"capture pipe close error: {e}")
        _log(f"capture stop: pid={os.getpid()}")


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


def stop_capture(stop_flag, proc):
    """キャプチャ子プロセスを停止。"""
    stop_flag.set()
    proc.join(timeout=3)
    if proc.is_alive():
        proc.kill()
