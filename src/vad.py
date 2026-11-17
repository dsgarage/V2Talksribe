#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""音声区間検出 (VAD) の統一ラッパ。

webrtc / silero の 2 バックエンドを 1 つのインターフェースにまとめ、
menubar / realtime に散らばっていた判定ロジックを集約する。

VAD 判定の前に **RMS エネルギーゲート** を必ず通す。これがないと
webrtc VAD が環境ノイズ/無音を発話と誤検出し、バッファが張り付いて
Whisper が幻聴を出し続ける負のループに陥る（本アプリの主要な不安定要因）。
"""
from __future__ import annotations

from typing import Optional

import numpy as np

EPSILON = 1e-9

# デフォルトのエネルギー下限 (RMS)。約 -46dBFS。
# これ未満は明確な無音/ノイズ床とみなし発話ではないと判定する。
DEFAULT_ENERGY_FLOOR_RMS = 0.005
# webrtc: フレーム投票率がこの値以上なら発話。0.2 は緩すぎたため引き上げ。
DEFAULT_WEBRTC_VOTE_RATIO = 0.6
# silero: 平均発話確率がこの値を超えたら発話。
DEFAULT_SILERO_THRESHOLD = 0.5


def rms(audio: np.ndarray) -> float:
    if audio.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(audio, dtype=np.float64)) + EPSILON))


class VoiceActivityDetector:
    """バックエンド非依存の VAD。

    使い方:
        vad = VoiceActivityDetector("webrtc", sr=16000)
        vad.load()
        vad.is_speech(chunk)  # -> bool
    """

    def __init__(
        self,
        backend: str,
        sr: int = 16000,
        *,
        aggressiveness: int = 2,
        silero_threshold: float = DEFAULT_SILERO_THRESHOLD,
        webrtc_vote_ratio: float = DEFAULT_WEBRTC_VOTE_RATIO,
        energy_floor_rms: float = DEFAULT_ENERGY_FLOOR_RMS,
    ):
        self.backend = backend
        self.sr = sr
        self.aggressiveness = aggressiveness
        self.silero_threshold = silero_threshold
        self.webrtc_vote_ratio = webrtc_vote_ratio
        self.energy_floor_rms = energy_floor_rms

        self._webrtc = None
        self._silero = None

    # ------------------------------------------------------------------
    # ロード / リセット
    # ------------------------------------------------------------------

    def load(self) -> None:
        if self.backend == "webrtc":
            import webrtcvad

            self._webrtc = webrtcvad.Vad(self.aggressiveness)
            self._silero = None
        elif self.backend == "silero":
            # faster-whisper 同梱の ONNX 版を使う(ネットワークに出ない)
            from faster_whisper.vad import get_vad_model

            self._silero = get_vad_model()
            self._webrtc = None
        else:
            raise ValueError(f"未知の VAD backend: {self.backend}")

    def reset(self) -> None:
        # silero(ONNX 版)は呼び出しごとに状態を作り直すので、何もしない
        return None

    # ------------------------------------------------------------------
    # 判定
    # ------------------------------------------------------------------

    def is_speech(self, audio: np.ndarray) -> bool:
        # まずエネルギーゲート。ノイズ床未満は問答無用で無音扱い。
        if rms(audio) < self.energy_floor_rms:
            return False

        if self.backend == "webrtc":
            return self._is_speech_webrtc(audio)
        return self._is_speech_silero(audio)

    def _is_speech_webrtc(self, audio: np.ndarray) -> bool:
        if self._webrtc is None:
            return False
        frame_len = int(self.sr * 0.03)  # 30ms
        if len(audio) < frame_len:
            return False
        clipped = np.clip(audio, -1.0, 1.0)
        pcm16 = (clipped * 32767.0).astype(np.int16)
        votes = 0
        total = 0
        for i in range(0, len(pcm16) - frame_len + 1, frame_len):
            frame = pcm16[i:i + frame_len]
            total += 1
            if self._webrtc.is_speech(frame.tobytes(), self.sr):
                votes += 1
        if total == 0:
            return False
        return (votes / total) >= self.webrtc_vote_ratio

    def _is_speech_silero(self, audio: np.ndarray) -> bool:
        if self._silero is None:
            return False
        if len(audio) < 512:
            return False
        # 入力長は 512 の倍数でなければならない
        padded = np.pad(audio, (0, (-len(audio)) % 512)).astype(np.float32)
        probs = self._silero(padded)
        if len(probs) == 0:
            return False
        return float(np.mean(probs)) > self.silero_threshold
