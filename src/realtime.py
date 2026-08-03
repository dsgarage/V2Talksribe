#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""リアルタイム文字起こし（CLI/ターミナル表示版）。

VAD 判定と Whisper 呼び出しは共通モジュール(vad.py / transcriber.py)へ委譲し、
menubar 経路との二重実装を解消する。本モジュールは rich による
ターミナル表示ループのみを担う。
"""
from __future__ import annotations

import threading
import time
from collections import deque
from typing import Optional

import numpy as np
from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.text import Text

from transcriber import Transcriber
from vad import VoiceActivityDetector

console = Console()


class AudioBuffer:
    """スレッドセーフなリングバッファ。マイク入力を蓄積する。"""

    def __init__(self, sr: int = 16000, max_sec: float = 30.0):
        self.sr = sr
        self.lock = threading.Lock()
        self.buf = np.zeros(0, dtype=np.float32)
        self.max_samples = int(sr * max_sec)

    def append(self, chunk: np.ndarray):
        with self.lock:
            self.buf = np.concatenate([self.buf, chunk.astype(np.float32)])
            if len(self.buf) > self.max_samples:
                self.buf = self.buf[-self.max_samples:]

    def get_last(self, sec: float) -> np.ndarray:
        n = int(self.sr * sec)
        with self.lock:
            if len(self.buf) < n:
                return self.buf.copy()
            return self.buf[-n:].copy()

    def clear(self):
        with self.lock:
            self.buf = np.zeros(0, dtype=np.float32)


def realtime_pipeline(
    model_size: str,
    language: Optional[str],
    device: str,
    compute_type: str,
    beam_size: int,
    chunk_sec: float,
    vad_threshold: float,
    input_device: Optional[int],
):
    import sounddevice as sd

    sr = 16000
    audio_buf = AudioBuffer(sr=sr, max_sec=60.0)
    stop_event = threading.Event()

    console.print(f"[bold]Whisper モデル読み込み中... ({model_size})[/bold]")
    transcriber = Transcriber(
        model_size, device=device, compute_type=compute_type,
        language=language, beam_size=beam_size, condition_on_previous_text=False,
    )
    transcriber.load()
    console.print("[green]モデル準備完了[/green]")

    console.print("[bold]Silero VAD 読み込み中...[/bold]")
    vad = VoiceActivityDetector("silero", sr, silero_threshold=vad_threshold)
    vad.load()
    console.print("[green]VAD 準備完了[/green]")

    transcript_lines: deque = deque(maxlen=50)
    current_partial = ""
    speech_buf = np.zeros(0, dtype=np.float32)
    silence_count = 0
    silence_limit = max(1, int(0.8 / chunk_sec))

    def audio_callback(indata, frames, time_info, status):
        if status:
            console.print(f"[yellow]Audio: {status}[/yellow]")
        audio_buf.append(indata[:, 0].copy())

    def build_display() -> Table:
        table = Table(
            show_header=True, header_style="bold cyan", expand=True,
            title="[bold]リアルタイム文字起こし[/bold]", title_style="bold white",
        )
        table.add_column("時刻", style="dim", width=10)
        table.add_column("テキスト", ratio=1)
        for ts, text in transcript_lines:
            table.add_row(ts, text)
        if current_partial:
            table.add_row(time.strftime("%H:%M:%S"), Text(current_partial, style="italic yellow"))
        return table

    if input_device is not None:
        dev_info = sd.query_devices(input_device)
        console.print(f"[bold]入力デバイス: {dev_info['name']}[/bold]")
    else:
        dev_info = sd.query_devices(kind="input")
        console.print(f"[bold]入力デバイス: {dev_info['name']}（デフォルト）[/bold]")

    console.print("[dim]Ctrl+C で終了[/dim]\n")
    chunk_samples = int(sr * chunk_sec)

    try:
        with sd.InputStream(
            samplerate=sr, channels=1, dtype="float32",
            blocksize=chunk_samples, device=input_device, callback=audio_callback,
        ):
            with Live(build_display(), console=console, refresh_per_second=4) as live:
                while not stop_event.is_set():
                    time.sleep(chunk_sec)
                    chunk = audio_buf.get_last(chunk_sec)
                    if len(chunk) < 512:
                        continue

                    if vad.is_speech(chunk):
                        silence_count = 0
                        speech_buf = np.concatenate([speech_buf, chunk])
                        if len(speech_buf) > sr * 1.0:
                            texts = transcriber.collect_texts(speech_buf)
                            current_partial = " ".join(texts)
                            live.update(build_display())
                    else:
                        silence_count += 1
                        if silence_count >= silence_limit and len(speech_buf) > sr * 0.3:
                            texts = transcriber.collect_texts(speech_buf)
                            if texts:
                                ts = time.strftime("%H:%M:%S")
                                transcript_lines.append((ts, " ".join(texts)))
                            current_partial = ""
                            speech_buf = np.zeros(0, dtype=np.float32)
                            live.update(build_display())
                        elif silence_count >= silence_limit:
                            speech_buf = np.zeros(0, dtype=np.float32)
                            current_partial = ""
    except KeyboardInterrupt:
        pass

    return list(transcript_lines)


def list_devices():
    import sounddevice as sd

    devices = sd.query_devices()
    table = Table(title="オーディオデバイス一覧", show_header=True, header_style="bold")
    table.add_column("ID", width=4)
    table.add_column("名前", ratio=1)
    table.add_column("入力ch", width=6)
    table.add_column("出力ch", width=6)
    table.add_column("SR", width=8)
    for i, d in enumerate(devices):
        if d["max_input_channels"] > 0:
            table.add_row(
                str(i), d["name"], str(d["max_input_channels"]),
                str(d["max_output_channels"]), str(int(d["default_samplerate"])),
            )
    console.print(table)


def save_transcript(lines, outpath: str):
    import io

    with io.open(outpath, "w", encoding="utf-8") as f:
        for ts, text in lines:
            f.write(f"[{ts}] {text}\n")
    console.print(f"[green]保存完了: {outpath}[/green]")
