#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""リアルタイム文字起こしモジュール

マイク入力 → Silero VAD → Faster-Whisper 逐次認識 → ターミナル表示
"""
from __future__ import annotations

import threading
import time
from collections import deque
from typing import Optional

import numpy as np
import typer
from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.text import Text

console = Console()

# --- 音声バッファ管理 ---

class AudioBuffer:
    """スレッドセーフなリングバッファ。マイク入力を蓄積し、VAD+ASR に渡す。"""

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


# --- VAD (Silero) ---

def load_silero_vad():
    import torch
    model, utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        verbose=False,
    )
    return model, utils


def is_speech_silero(model, audio: np.ndarray, sr: int = 16000, threshold: float = 0.3) -> bool:
    """512サンプルずつ Silero VAD に渡し、平均確率で判定する。"""
    import torch
    if len(audio) < 512:
        return False
    probs = []
    for i in range(0, len(audio) - 512 + 1, 512):
        tensor = torch.from_numpy(audio[i:i + 512]).float()
        probs.append(model(tensor, sr).item())
    return (sum(probs) / len(probs)) > threshold if probs else False


# --- リアルタイムパイプライン ---

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
    from faster_whisper import WhisperModel

    sr = 16000
    audio_buf = AudioBuffer(sr=sr, max_sec=60.0)
    stop_event = threading.Event()

    # Whisper モデルロード
    console.print(f"[bold]Whisper モデル読み込み中... ({model_size})[/bold]")
    whisper = WhisperModel(model_size, device=device, compute_type=compute_type)
    console.print("[green]モデル準備完了[/green]")

    # Silero VAD ロード
    console.print("[bold]Silero VAD 読み込み中...[/bold]")
    vad_model, _ = load_silero_vad()
    console.print("[green]VAD 準備完了[/green]")

    # 結果保持
    transcript_lines: deque = deque(maxlen=50)
    current_partial = ""
    speech_buf = np.zeros(0, dtype=np.float32)
    silence_count = 0
    # 無音が何チャンク続いたら確定とするか
    silence_limit = max(1, int(0.8 / chunk_sec))

    def audio_callback(indata, frames, time_info, status):
        if status:
            console.print(f"[yellow]Audio: {status}[/yellow]")
        mono = indata[:, 0].copy()
        audio_buf.append(mono)

    def build_display() -> Table:
        table = Table(
            show_header=True,
            header_style="bold cyan",
            expand=True,
            title="[bold]リアルタイム文字起こし[/bold]",
            title_style="bold white",
        )
        table.add_column("時刻", style="dim", width=10)
        table.add_column("テキスト", ratio=1)

        for ts, text in transcript_lines:
            table.add_row(ts, text)

        if current_partial:
            table.add_row(
                time.strftime("%H:%M:%S"),
                Text(current_partial, style="italic yellow"),
            )

        return table

    # マイクデバイス情報
    if input_device is not None:
        dev_info = sd.query_devices(input_device)
        console.print(f"[bold]入力デバイス: {dev_info['name']}[/bold]")
    else:
        dev_info = sd.query_devices(kind="input")
        console.print(f"[bold]入力デバイス: {dev_info['name']}（デフォルト）[/bold]")

    console.print(f"[dim]Ctrl+C で終了[/dim]\n")

    chunk_samples = int(sr * chunk_sec)

    try:
        with sd.InputStream(
            samplerate=sr,
            channels=1,
            dtype="float32",
            blocksize=chunk_samples,
            device=input_device,
            callback=audio_callback,
        ):
            with Live(build_display(), console=console, refresh_per_second=4) as live:
                while not stop_event.is_set():
                    time.sleep(chunk_sec)

                    chunk = audio_buf.get_last(chunk_sec)
                    if len(chunk) < 512:
                        continue

                    is_speech = is_speech_silero(
                        vad_model, chunk, sr=sr, threshold=vad_threshold
                    )

                    if is_speech:
                        silence_count = 0
                        speech_buf = np.concatenate([speech_buf, chunk])

                        # 途中経過: 蓄積が一定以上なら部分認識
                        if len(speech_buf) > sr * 1.0:
                            segments, _ = whisper.transcribe(
                                speech_buf,
                                language=language,
                                beam_size=beam_size,
                                vad_filter=False,
                            )
                            texts = [s.text.strip() for s in segments if s.text.strip()]
                            current_partial = " ".join(texts)
                            live.update(build_display())
                    else:
                        silence_count += 1

                        if silence_count >= silence_limit and len(speech_buf) > sr * 0.3:
                            # 発話終了 → 確定認識
                            segments, _ = whisper.transcribe(
                                speech_buf,
                                language=language,
                                beam_size=beam_size,
                                vad_filter=False,
                            )
                            texts = [s.text.strip() for s in segments if s.text.strip()]
                            if texts:
                                ts = time.strftime("%H:%M:%S")
                                transcript_lines.append((ts, " ".join(texts)))
                            current_partial = ""
                            speech_buf = np.zeros(0, dtype=np.float32)
                            live.update(build_display())

                        elif silence_count >= silence_limit:
                            # 短すぎる音声は破棄
                            speech_buf = np.zeros(0, dtype=np.float32)
                            current_partial = ""

    except KeyboardInterrupt:
        pass

    return list(transcript_lines)


def list_devices():
    """利用可能なオーディオデバイスを一覧表示する。"""
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
                str(i),
                d["name"],
                str(d["max_input_channels"]),
                str(d["max_output_channels"]),
                str(int(d["default_samplerate"])),
            )

    console.print(table)


def save_transcript(lines, outpath: str):
    """確定した文字起こし結果をテキストファイルに保存する。"""
    import io

    with io.open(outpath, "w", encoding="utf-8") as f:
        for ts, text in lines:
            f.write(f"[{ts}] {text}\n")
    console.print(f"[green]保存完了: {outpath}[/green]")
