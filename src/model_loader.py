#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""モデル読み込みを別スレッドで行う LoadJob。

エンジンのワーカーは読み込み中も止まらず、コマンド(停止・取り消し)を処理し続ける。
LoadJob は音声状態(speech_buf / capture / vad の状態など)に触らず、
出来上がった transcriber と vad を渡すだけ。取り消しはフラグだけで、
スレッドは殺せないので終わるまで裏で走らせる(結果は捨てる、または再利用する)。
"""
from __future__ import annotations

import os
import threading
import time
import traceback
from typing import Optional

import applog


def _hf_repo(model_size: str) -> str:
    try:
        from faster_whisper.utils import _MODELS

        return _MODELS.get(model_size, model_size)
    except Exception:
        return model_size


def _cache_dirs(model_size: str) -> list:
    import huggingface_hub.constants as hc

    repo = _hf_repo(model_size)
    return [
        os.path.join(hc.HF_HUB_CACHE, "models--" + repo.replace("/", "--")),
        os.path.join(os.path.dirname(hc.HF_HUB_CACHE), "xet"),
    ]


def hf_cache_bytes(model_size: str) -> int:
    """モデルの HF キャッシュ(と xet)の合計バイト数。ダウンロードの進み具合の目安。"""
    total = 0
    try:
        dirs = _cache_dirs(model_size)
    except Exception:
        return 0
    for d in dirs:
        try:
            for root, _dirs, files in os.walk(d):
                for name in files:
                    try:
                        total += os.stat(os.path.join(root, name)).st_size
                    except OSError:
                        pass
        except Exception:
            pass
    return total


class LoadJob:
    """Whisper と VAD の読み込みを別スレッドで行う。"""

    def __init__(self, *, transcriber_factory, vad_factory, model_size: str, language,
                 beam_size: int, vad_backend: str, vad_aggressiveness: int, sr: int):
        self.transcriber_factory = transcriber_factory
        self.vad_factory = vad_factory
        self.model_size = model_size
        self.language = language
        self.beam_size = beam_size
        self.vad_backend = vad_backend
        self.vad_aggressiveness = vad_aggressiveness
        self.sr = sr

        self.phase = "check"          # "check" | "download" | "whisper" | "vad" | "done"
        now = time.monotonic()
        self.phase_started_at = now
        self.started_at = now
        self.transcriber = None
        self.vad = None
        self.vad_fell_back = False
        self.error: Optional[str] = None
        self.warnings: list = []
        self.cancelled = False
        self.done = threading.Event()
        self.download_dir: Optional[str] = None

    @property
    def key(self) -> tuple:
        return (self.model_size, self.beam_size, self.vad_backend)

    def _set_phase(self, phase: str):
        self.phase = phase
        self.phase_started_at = time.monotonic()

    def start(self) -> None:
        threading.Thread(target=self._run, daemon=True).start()

    def cancel(self) -> None:
        self.cancelled = True

    def _run(self):
        try:
            self._set_phase("check")
            t = self.transcriber_factory(self.model_size, self.language, self.beam_size)
            p = t.cached_path()
            if p is None:
                self._set_phase("download")
                try:
                    self.download_dir = _cache_dirs(self.model_size)[0]
                except Exception:
                    self.download_dir = None
                p = t.download()
            self._set_phase("whisper")
            t.load(p)
            self.transcriber = t

            self._set_phase("vad")
            v = self.vad_factory(self.vad_backend, self.sr, self.vad_aggressiveness)
            try:
                v.load()
            except Exception as e:
                if self.vad_backend == "webrtc":
                    raise
                applog.log(f"vad load failed ({self.vad_backend}): {e}")
                self.warnings.append(f"Silero VAD を読み込めないため webrtc に切り替えました: {e}")
                self.vad_fell_back = True
                v = self.vad_factory("webrtc", self.sr, self.vad_aggressiveness)
                v.load()
            self.vad = v
            self._set_phase("done")
        except Exception as e:
            self.error = f"{type(e).__name__}: {e}"
            applog.log(f"LoadJob error: {self.error}")
            applog.log(traceback.format_exc())
        finally:
            self.done.set()
