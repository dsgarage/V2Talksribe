#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""別プロセスでマイク音声をキャプチャし、メインプロセスへ pipe で送る。

macOS 26 ではマイクを使うプロセスのメニューバーアイコンがマイク表示に
置き換わるため、キャプチャを別プロセスに分離している。

子プロセス内は 2 段のスレッドキューで構成する。
  audio callback → raw_q(RAW_QUEUE_MAX) → 主ループ(WAV 書き込み)
                                        → send_q(SEND_QUEUE_MAX) → 送信スレッド → 親
callback ではディスクにも pipe にも触らない。親が詰まって send_q が満杯に
なったときは古いチャンクを捨てる(文字起こしに回す分だけが欠ける。WAV には
raw_q に入った分がすべて書かれる)。捨てた数は stats で親へ知らせる。

親子の間は 2 本の pipe: audio(float32 の生バイト)と ctrl(状態の dict)。
"""
from __future__ import annotations

import multiprocessing as mp
import os
import queue
import signal
import threading
import time
import traceback
from dataclasses import dataclass
from multiprocessing.connection import Connection
from typing import Optional

import numpy as np

import applog
import audio_store
import devices

# コールバック → 子の主ループ(ファイル書き込み)
RAW_QUEUE_MAX = 600
# 子の主ループ → 送信用スレッド(親へ)
SEND_QUEUE_MAX = 64
STATS_INTERVAL_SEC = 2.0


def _log(msg: str):
    applog.log(msg)


@dataclass
class CaptureHandle:
    audio: Connection          # 親側(受信)。音声は recv_bytes() で float32 の生バイト
    ctrl: Connection           # 親側(受信)。状態は recv() で dict
    proc: "mp.Process"
    stop_flag: "mp.synchronize.Event"
    record_path: Optional[str]

    def is_alive(self) -> bool:
        return self.proc.is_alive()

    def stop(self, timeout: float = 3.0) -> None:
        """子プロセスを止めて pipe を閉じる。2 回呼んでも安全。"""
        try:
            self.stop_flag.set()
        except Exception as e:
            _log(f"capture handle stop_flag error: {e}")
        try:
            self.proc.join(timeout)
            if self.proc.is_alive():
                self.proc.kill()
                self.proc.join(1)
        except Exception as e:
            _log(f"capture handle join/kill error: {e}")
        for conn in (self.audio, self.ctrl):
            try:
                conn.close()
            except Exception as e:
                _log(f"capture handle close error: {e}")


def _send_ctrl(ctrl_conn, msg: dict, parent_gone: Optional[threading.Event] = None) -> bool:
    """ctrl pipe へ送る。失敗(親が消えた)は握って False を返す。"""
    try:
        ctrl_conn.send(msg)
        return True
    except Exception as e:
        _log(f"capture ctrl send error: {type(e).__name__}: {e}")
        if parent_gone is not None:
            parent_gone.set()
        return False


def _capture_loop(audio_conn, ctrl_conn, stop_flag, sr: int, chunk_sec: float,
                  device_name: Optional[str], record_path: Optional[str]):
    """子プロセスで実行。sounddevice でマイク入力を取得し pipe で親へ送る。"""
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    stream = None
    recorder = None
    sender_thread = None
    parent_gone = threading.Event()
    counters = {"dropped_live": 0, "dropped_record": 0, "status_flags": 0}
    try:
        import sounddevice as sd

        devs, default_index = devices.query_input_devices_inprocess()
        index, name, fell_back = devices.resolve_input_device(devs, device_name, default_index)
        if index is None:
            _send_ctrl(ctrl_conn, {"type": "error", "message": "使える入力デバイスがありません"})
            return

        if record_path:
            recorder = audio_store.WavRecorder(record_path, sr)

        raw_q: "queue.Queue" = queue.Queue(maxsize=RAW_QUEUE_MAX)
        send_q: "queue.Queue" = queue.Queue(maxsize=SEND_QUEUE_MAX)

        def cb(indata, frames, time_info, status):
            if status:
                counters["status_flags"] += 1
            try:
                raw_q.put_nowait(indata[:, 0].copy())
            except queue.Full:
                # 新しい方を捨てる(ディスク書き込みが詰まっている)
                counters["dropped_record"] += 1

        stream = sd.InputStream(
            samplerate=sr, channels=1, dtype="float32",
            blocksize=int(sr * chunk_sec), device=index, callback=cb,
        )
        stream.start()
        _send_ctrl(ctrl_conn, {
            "type": "started", "device_index": index, "device_name": name,
            "requested": device_name, "fell_back": fell_back, "record_path": record_path,
        }, parent_gone)
        _log(f"capture start: pid={os.getpid()} device={index} '{name}' "
             f"requested={device_name} fell_back={fell_back} sr={sr} "
             f"chunk={chunk_sec} record={record_path}")

        def sender():
            while not stop_flag.is_set() and not parent_gone.is_set():
                try:
                    x = send_q.get(timeout=0.5)
                except queue.Empty:
                    continue
                try:
                    audio_conn.send_bytes(x.tobytes())
                except (BrokenPipeError, EOFError, OSError) as e:
                    _log(f"capture pipe error: {type(e).__name__}: {e}")
                    parent_gone.set()
                    return

        sender_thread = threading.Thread(target=sender, daemon=True)
        sender_thread.start()

        last_stats_time = time.monotonic()
        last_sent = (0, 0, 0)
        while not stop_flag.is_set() and not parent_gone.is_set():
            try:
                x = raw_q.get(timeout=0.5)
            except queue.Empty:
                x = None
            if x is not None:
                if recorder is not None:
                    recorder.write(x)
                try:
                    send_q.put_nowait(x)
                except queue.Full:
                    # 親が詰まっている: 最古を捨てて最新を残す
                    try:
                        send_q.get_nowait()
                    except queue.Empty:
                        pass
                    counters["dropped_live"] += 1
                    try:
                        send_q.put_nowait(x)
                    except queue.Full:
                        pass
            now = time.monotonic()
            if now - last_stats_time >= STATS_INTERVAL_SEC:
                last_stats_time = now
                cur = (counters["dropped_live"], counters["dropped_record"],
                       counters["status_flags"])
                if cur != last_sent:
                    last_sent = cur
                    _send_ctrl(ctrl_conn, {
                        "type": "stats", "dropped_live_chunks": cur[0],
                        "dropped_record_chunks": cur[1], "status_flags": cur[2],
                    }, parent_gone)
    except Exception as e:
        _send_ctrl(ctrl_conn, {"type": "error", "message": f"{type(e).__name__}: {e}"})
        _log(f"capture loop error: {type(e).__name__}: {e}\n{traceback.format_exc()}")
    finally:
        if stream is not None:
            try:
                stream.stop()
                stream.close()
            except Exception as e:
                _log(f"capture stream cleanup error: {e}")
        if recorder is not None:
            recorder.close()
        if sender_thread is not None:
            sender_thread.join(timeout=1.0)
        for conn in (audio_conn, ctrl_conn):
            try:
                conn.close()
            except Exception as e:
                _log(f"capture pipe close error: {e}")
        _log(f"capture stop: pid={os.getpid()} dropped_live={counters['dropped_live']} "
             f"dropped_record={counters['dropped_record']} status_flags={counters['status_flags']}")


def _file_source_loop(audio_conn, ctrl_conn, stop_flag, sr: int, chunk_sec: float,
                      file_path: str, speed: float):
    """子プロセスで実行。音声ファイルをチャンクに切って親へ送る(再生・テスト用)。"""
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        from faster_whisper.audio import decode_audio

        audio = decode_audio(file_path, sampling_rate=sr)
        _send_ctrl(ctrl_conn, {
            "type": "started", "device_index": None,
            "device_name": "file:" + os.path.basename(file_path), "requested": None,
            "fell_back": False, "record_path": None,
        })
        n = int(sr * chunk_sec)
        for i in range(0, len(audio), n):
            if stop_flag.is_set():
                break
            try:
                audio_conn.send_bytes(audio[i:i + n].astype(np.float32).tobytes())
            except (BrokenPipeError, EOFError, OSError) as e:
                _log(f"file source pipe error: {type(e).__name__}: {e}")
                return
            if speed > 0:
                time.sleep(chunk_sec / speed)
        else:
            _send_ctrl(ctrl_conn, {"type": "eof"})
        # stop は親が必ず呼ぶ
        while not stop_flag.wait(0.5):
            pass
    except Exception as e:
        _send_ctrl(ctrl_conn, {"type": "error", "message": f"{type(e).__name__}: {e}"})
        _log(f"file source error: {type(e).__name__}: {e}\n{traceback.format_exc()}")
    finally:
        for conn in (audio_conn, ctrl_conn):
            try:
                conn.close()
            except Exception as e:
                _log(f"file source pipe close error: {e}")


def _spawn(target, args_tail: tuple, record_path: Optional[str]) -> CaptureHandle:
    audio_parent, audio_child = mp.Pipe(duplex=False)
    ctrl_parent, ctrl_child = mp.Pipe(duplex=False)
    stop_flag = mp.Event()
    proc = mp.Process(
        target=target,
        args=(audio_child, ctrl_child, stop_flag) + args_tail,
        daemon=True,
    )
    proc.start()
    audio_child.close()
    ctrl_child.close()
    return CaptureHandle(audio=audio_parent, ctrl=ctrl_parent, proc=proc,
                         stop_flag=stop_flag, record_path=record_path)


def start_capture(sr: int, chunk_sec: float, device_name: Optional[str] = None,
                  record_path: Optional[str] = None) -> CaptureHandle:
    """キャプチャ子プロセスを開始する。"""
    return _spawn(_capture_loop, (sr, chunk_sec, device_name, record_path), record_path)


def start_file_capture(sr: int, chunk_sec: float, file_path: str,
                       speed: float = 1.0) -> CaptureHandle:
    """音声ファイルをキャプチャと同じ形で流す子プロセスを開始する。"""
    return _spawn(_file_source_loop, (sr, chunk_sec, file_path, speed), None)
