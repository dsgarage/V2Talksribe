# -*- coding: utf-8 -*-
"""エンジンのテスト用の偽物(マイクも本物のモデルも使わない)。"""
import multiprocessing as mp
import queue
import threading
import time

import numpy as np


class FakeCapture:
    def __init__(self, record_path=None):
        self.audio, self._audio_w = mp.Pipe(duplex=False)
        self.ctrl, self._ctrl_w = mp.Pipe(duplex=False)
        self.record_path = record_path
        self.alive = True
        self.stopped = False

    def feed(self, x: np.ndarray):
        self._audio_w.send_bytes(x.astype(np.float32).tobytes())

    def send_ctrl(self, msg: dict):
        self._ctrl_w.send(msg)

    def started(self, name="Fake Mic", requested=None, fell_back=False):
        self.send_ctrl({"type": "started", "device_index": 0, "device_name": name,
                        "requested": requested, "fell_back": fell_back,
                        "record_path": self.record_path})

    def die(self):
        """子プロセスが落ちたのと同じ状態にする。"""
        self.alive = False
        for c in (self._audio_w, self._ctrl_w):
            try:
                c.close()
            except Exception:
                pass

    def is_alive(self):
        return self.alive

    def stop(self, timeout=3.0):
        self.stopped = True
        self.die()
        for c in (self.audio, self.ctrl):
            try:
                c.close()
            except Exception:
                pass


class FakeCaptureFactory:
    """behaviors: "ok" / "ok_nostart" / Exception インスタンス のリスト。尽きたら "ok"。"""

    def __init__(self, behaviors=None):
        self.behaviors = list(behaviors or [])
        self.calls = []
        self.captures = []
        self.lock = threading.Lock()

    def __call__(self, sr, chunk_sec, device_name, record_path):
        with self.lock:
            self.calls.append({"sr": sr, "chunk_sec": chunk_sec, "device_name": device_name,
                               "record_path": record_path, "t": time.monotonic()})
            b = self.behaviors.pop(0) if self.behaviors else "ok"
        if isinstance(b, Exception):
            raise b
        cap = FakeCapture(record_path)
        if b == "ok":
            cap.started(name=device_name or "Default Mic")
        self.captures.append(cap)
        return cap

    @property
    def last(self):
        return self.captures[-1]


class FakeTranscriber:
    def __init__(self, model_size="base", language=None, beam_size=1, *, block=None,
                 texts=None, raise_on=None, cached=True):
        self.model_size = model_size
        self.language = language
        self.beam_size = beam_size
        self.block = block
        self.texts = texts if texts is not None else ["テスト"]
        self.raise_on = raise_on
        self.cached = cached
        self.loaded = False
        self.calls = 0

    def cached_path(self):
        return "/fake/path" if self.cached else None

    def download(self):
        return "/fake/path"

    def load(self, model_path=None):
        if self.block is not None:
            self.block.wait()
        self.loaded = True

    def collect_texts(self, audio, emit_until_sec=None, **kw):
        self.calls += 1
        if self.raise_on is not None and self.calls in self.raise_on:
            raise RuntimeError("fake transcribe error")
        return list(self.texts)


class FakeVAD:
    def __init__(self, backend="webrtc", sr=16000, aggressiveness=2, *, raise_once=False,
                 load_error=False):
        self.backend = backend
        self.raise_once = raise_once
        self.load_error = load_error

    def load(self):
        if self.load_error:
            raise RuntimeError("fake vad load error")

    def reset(self):
        pass

    def is_speech(self, audio):
        if self.raise_once:
            self.raise_once = False
            raise RuntimeError("fake vad error")
        return float(np.sqrt(np.mean(np.square(audio)))) > 0.01


def collect_events(engine, until, timeout=5.0) -> list:
    """engine.events を読み、until(event) が真になるか timeout まで集めて返す。"""
    out = []
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        try:
            ev = engine.events.get(timeout=0.05)
        except queue.Empty:
            continue
        out.append(ev)
        if until(ev):
            break
    return out
