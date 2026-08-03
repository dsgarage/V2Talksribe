#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import math
import os
import subprocess
import tempfile
from dataclasses import dataclass, field

import numpy as np


EPSILON = 1e-6


@dataclass
class AudioPreprocessSettings:
    normalize_enabled: bool = False
    denoise_enabled: bool = False
    target_peak: float = 0.9
    max_gain: float = 4.0
    noise_floor_db: float = -38.0

    def enabled(self) -> bool:
        return self.normalize_enabled or self.denoise_enabled

    def label(self) -> str:
        parts = []
        if self.denoise_enabled:
            parts.append("ノイズ除去")
        if self.normalize_enabled:
            parts.append("ノーマライズ")
        return " / ".join(parts) if parts else "なし"


@dataclass
class FilePreprocessResult:
    path: str
    cleanup_paths: list[str] = field(default_factory=list)
    description: str = "なし"


class LiveAudioPreprocessor:
    """リアルタイム経路向けの軽量前処理。"""

    def __init__(self, settings: AudioPreprocessSettings, sample_rate: int):
        self.settings = settings
        self.sample_rate = sample_rate
        self._prev_x = 0.0
        self._prev_y = 0.0
        self._noise_floor_rms = max(10 ** (settings.noise_floor_db / 20.0), 0.0015)

    def process(self, audio: np.ndarray) -> np.ndarray:
        out = np.asarray(audio, dtype=np.float32).copy()
        if not len(out):
            return out
        if self.settings.denoise_enabled:
            out = self._apply_denoise(out)
        if self.settings.normalize_enabled:
            out = self._apply_normalize(out)
        return np.clip(out, -1.0, 1.0)

    def _apply_denoise(self, audio: np.ndarray) -> np.ndarray:
        filtered = self._highpass(audio, cutoff_hz=80.0)
        rms = float(np.sqrt(np.mean(filtered * filtered) + EPSILON))
        if rms < self._noise_floor_rms * 1.8:
            self._noise_floor_rms = (self._noise_floor_rms * 0.97) + (rms * 0.03)

        threshold = max(self._noise_floor_rms * 2.5, 0.002)
        envelope = np.abs(filtered)
        gate = np.clip((envelope - threshold) / max(threshold * 2.0, EPSILON), 0.0, 1.0)
        return filtered * (0.12 + 0.88 * gate.astype(np.float32))

    def _apply_normalize(self, audio: np.ndarray) -> np.ndarray:
        peak = float(np.max(np.abs(audio)))
        if peak < 0.01:
            return audio
        gain = min(self.settings.target_peak / peak, self.settings.max_gain)
        if gain <= 1.01:
            return audio
        return audio * gain

    def _highpass(self, audio: np.ndarray, cutoff_hz: float) -> np.ndarray:
        rc = 1.0 / (2.0 * math.pi * cutoff_hz)
        dt = 1.0 / float(self.sample_rate)
        alpha = rc / (rc + dt)
        out = np.empty_like(audio)
        prev_x = self._prev_x
        prev_y = self._prev_y
        for i, x in enumerate(audio):
            prev_y = alpha * (prev_y + float(x) - prev_x)
            prev_x = float(x)
            out[i] = prev_y
        self._prev_x = prev_x
        self._prev_y = prev_y
        return out


def preprocess_audio_file(filepath: str, settings: AudioPreprocessSettings) -> FilePreprocessResult:
    """ファイル経路向けに ffmpeg ベースの前処理 WAV を生成する。"""
    if not settings.enabled():
        return FilePreprocessResult(path=filepath, description=settings.label())

    fd, out_path = tempfile.mkstemp(prefix="talksribe_pre_", suffix=".wav")
    os.close(fd)

    filters = []
    if settings.denoise_enabled:
        filters.extend(
            [
                "highpass=f=80",
                "afftdn=nf=-32:tn=1",
            ]
        )
    if settings.normalize_enabled:
        filters.append("loudnorm=I=-16:LRA=7:TP=-1.5")

    cmd = ["ffmpeg", "-y", "-i", filepath]
    if filters:
        cmd.extend(["-af", ",".join(filters)])
    cmd.extend(["-ar", "16000", "-ac", "1", "-f", "wav", out_path])

    result = subprocess.run(cmd, capture_output=True, timeout=600)
    if result.returncode != 0:
        try:
            os.unlink(out_path)
        except OSError:
            pass
        raise RuntimeError(f"ffmpeg 前処理失敗: {result.stderr.decode(errors='ignore')[:200]}")

    return FilePreprocessResult(
        path=out_path,
        cleanup_paths=[out_path],
        description=settings.label(),
    )
