#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ライブ経路の入力レベルの扱い(純粋な計算。numpy だけ)。

- 床に合わせて動く関門(雑音の床 + マージン以上のフレームだけ VAD に回す)
- VAD に渡す音へ掛けるゆっくり動く利得(AGC)
- Whisper に渡す直前の、区間ごとの RMS 正規化と柔らかい制限
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

FRAME_SAMPLES = 4096          # 0.256 秒(16kHz)。silero の 512 の倍数
ZERO_PEAK = 1e-4              # これ未満のフレームは「完全な無音」(床の計算に入れない)
ABS_MIN_DB = -65.0            # これ未満は必ず関門で落とす
_ACTIVE_WARMUP_SEC = 1.0      # 通過が この秒数ぶんたまるまで active_db は None


def frame_rms_db(x: np.ndarray) -> float:
    if x.size == 0:
        return -240.0
    ms = float(np.mean(np.square(x, dtype=np.float64)))
    return 20.0 * float(np.log10(np.sqrt(ms) + 1e-12))


def db_to_gain(db: float) -> float:
    return 10.0 ** (db / 20.0)


def soft_limit(x: np.ndarray, knee: float = 0.9) -> np.ndarray:
    """|x| <= knee はそのまま。超えた分は knee + (1-knee) * tanh((|x|-knee)/(1-knee)) に押し込む。

    符号は保つ。出力は必ず |y| < 1.0(float32)。
    """
    a = np.asarray(x, dtype=np.float32)
    mag = np.abs(a)
    over = mag > knee
    if not np.any(over):
        return a.copy()
    out = a.copy()
    room = 1.0 - knee
    squeezed = knee + room * np.tanh((mag[over] - knee) / room)
    # float32 に丸めても 1.0 に届かないようにする
    squeezed = np.minimum(squeezed, np.float32(1.0) - np.float32(1e-6))
    out[over] = np.sign(a[over]) * squeezed
    return out.astype(np.float32)


@dataclass
class FrameLevel:
    rms_db: float
    peak: float
    zero: bool
    gated: bool      # rms_db >= floor_db + margin かつ rms_db >= ABS_MIN_DB かつ not zero
    floor_db: float  # このフレームを見たあとの床


class LevelTracker:
    def __init__(self, sr: int, tuning):
        self.sr = sr
        self.tuning = tuning
        self.reset()

    def reset(self) -> None:
        self.floor_db: float = self.tuning.floor_init_db
        self.active_db: Optional[float] = None
        self.gain_db: float = 0.0
        self._warm_sum = 0.0
        self._warm_sec = 0.0

    def update(self, frame: np.ndarray) -> FrameLevel:
        t = self.tuning
        n = len(frame)
        dt = n / self.sr
        peak = float(np.max(np.abs(frame))) if n else 0.0
        zero = peak < ZERO_PEAK
        rms_db = frame_rms_db(frame)

        floor_before = self.floor_db
        if not zero:
            if rms_db < self.floor_db:
                self.floor_db = rms_db
            else:
                self.floor_db += min(t.floor_rise_db_per_sec * dt, rms_db - self.floor_db)

        gated = (not zero) and rms_db >= ABS_MIN_DB and rms_db >= floor_before + t.gate_margin_db

        if t.agc_enabled:
            if gated:
                if self.active_db is None:
                    self._warm_sum += rms_db * dt
                    self._warm_sec += dt
                    if self._warm_sec >= _ACTIVE_WARMUP_SEC:
                        self.active_db = self._warm_sum / self._warm_sec
                else:
                    self.active_db += (rms_db - self.active_db) * min(1.0, dt / t.agc_tau_sec)
            if self.active_db is None:
                target = 0.0
            else:
                target = float(np.clip(t.agc_target_db - self.active_db, 0.0, t.agc_max_gain_db))
            step = t.agc_slew_db_per_sec * dt
            if target > self.gain_db:
                self.gain_db = min(target, self.gain_db + step)
            else:
                self.gain_db = max(target, self.gain_db - step)
            self.gain_db = max(0.0, self.gain_db)   # 大きい声を小さくはしない
        return FrameLevel(rms_db, peak, zero, gated, self.floor_db)

    def vad_input(self, frame: np.ndarray) -> np.ndarray:
        if self.tuning.vad_on_agc and self.tuning.agc_enabled:
            return soft_limit(frame * np.float32(db_to_gain(self.gain_db)))
        return frame


def normalize_for_asr(audio: np.ndarray, voiced_rms_db: list, tuning) -> np.ndarray:
    """voiced_rms_db(この区間で発話と判定したフレームの RMS dB)の中央値を基準に、
    gain_db = clip(asr_target_db - median, 0, asr_max_gain_db) を掛け、soft_limit を通す。
    tuning.asr_normalize が偽、または voiced_rms_db が空なら audio をそのまま返す(コピーしない)。"""
    if not tuning.asr_normalize or not voiced_rms_db:
        return audio
    med = float(np.median(voiced_rms_db))
    gain_db = float(np.clip(tuning.asr_target_db - med, 0.0, tuning.asr_max_gain_db))
    return soft_limit(audio * np.float32(db_to_gain(gain_db)))
