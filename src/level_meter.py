# -*- coding: utf-8 -*-
"""入力レベルのメーター(メニューバー用)。numpy だけの純関数とその状態。"""
from __future__ import annotations

import math

import numpy as np

WINDOW_SEC = 0.25
HISTORY = 5
FLOOR_DB = -60.0
TOP_DB = -20.0
CLIP_PEAK = 0.99
SILENT_PEAK = 1e-4
CLIP_HOLD_SEC = 2.0
BARS = "\u2581\u2582\u2583\u2584\u2585\u2586\u2587\u2588"
QUIET = "\u00b7"


def window_levels(audio: np.ndarray, sr: int, window_sec: float = WINDOW_SEC) -> list:
    """音声を window_sec ごとに区切り、[(rms, peak), ...] を返す。
    端の余り(window の半分以上)も 1 区間にする。空なら []。"""
    n = len(audio)
    win = max(1, int(round(sr * window_sec)))
    if n == 0:
        return []
    out = []
    pos = 0
    while pos < n:
        seg = audio[pos:pos + win]
        if len(seg) < win and len(seg) * 2 < win:
            break
        seg = seg.astype(np.float64, copy=False)
        rms = float(np.sqrt(np.mean(seg * seg)))
        peak = float(np.max(np.abs(seg)))
        out.append((rms, peak))
        pos += win
    return out


def level_char(rms: float, peak: float) -> str:
    """1 区間の (rms, peak) を 1 文字にする。"""
    if peak < SILENT_PEAK or rms <= 0.0:
        return QUIET
    db = 20.0 * math.log10(rms)
    if db < FLOOR_DB:
        return QUIET
    idx = int((db - FLOOR_DB) / (TOP_DB - FLOOR_DB) * len(BARS))
    return BARS[min(idx, len(BARS) - 1)]


class LevelMeter:
    """直近の区間の文字を持ち、メニューバー用の文字列を作る。"""

    def __init__(self, history: int = HISTORY, clip_hold_sec: float = CLIP_HOLD_SEC):
        self.history = history
        self.clip_hold_sec = clip_hold_sec
        self.reset()

    def reset(self) -> None:
        self._chars = []
        self._last_clip_at = None

    def push(self, levels: list, now: float) -> None:
        for rms, peak in levels:
            self._chars.append(level_char(rms, peak))
            if peak >= CLIP_PEAK:
                self._last_clip_at = now
        if len(self._chars) > self.history:
            self._chars = self._chars[-self.history:]

    def render(self, now: float) -> str:
        chars = QUIET * (self.history - len(self._chars)) + "".join(self._chars)
        if self._last_clip_at is not None and now - self._last_clip_at < self.clip_hold_sec:
            chars += "!"
        return chars
