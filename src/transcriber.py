#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Faster-Whisper ラッパ（幻聴フィルタを一箇所に集約）。

ライブ経路・ファイル経路の双方がここを通す。無音やノイズに対して
Whisper が幻聴（実在しない発話）を出す問題を、以下で抑止する。

- ``condition_on_previous_text=False``: 直前テキストへの引きずられ暴走を防止
- ``no_speech_threshold`` / ``log_prob_threshold``: 無音セグメントの skip
- ``compression_ratio_threshold``: 反復ハルシネーションの skip
- 後段フィルタ: avg_logprob が低すぎる/no_speech_prob が高すぎる区間を追加で除外
"""
from __future__ import annotations

from typing import Iterable, Iterator, List, Optional

import numpy as np

# 後段フィルタの既定しきい値。faster-whisper 内部の skip をすり抜けた
# 低信頼セグメントを追加で落とす保険。
DEFAULT_NO_SPEECH_THRESHOLD = 0.6
DEFAULT_LOG_PROB_THRESHOLD = -1.0
DEFAULT_COMPRESSION_RATIO_THRESHOLD = 2.4


class Transcriber:
    """WhisperModel を保持し、幻聴抑止つきで文字起こしする。"""

    def __init__(
        self,
        model_size: str,
        *,
        device: str = "cpu",
        compute_type: str = "int8",
        cpu_threads: int = 0,
        language: Optional[str] = None,
        beam_size: int = 1,
        condition_on_previous_text: bool = False,
        no_speech_threshold: float = DEFAULT_NO_SPEECH_THRESHOLD,
        log_prob_threshold: float = DEFAULT_LOG_PROB_THRESHOLD,
        compression_ratio_threshold: float = DEFAULT_COMPRESSION_RATIO_THRESHOLD,
        hotwords: Optional[str] = None,
    ):
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.cpu_threads = cpu_threads
        self.language = language
        self.beam_size = beam_size
        # ライブは False（直前文脈への暴走を防ぐ）、ファイルは True（長尺の連続性）
        self.condition_on_previous_text = condition_on_previous_text
        self.no_speech_threshold = no_speech_threshold
        self.log_prob_threshold = log_prob_threshold
        self.compression_ratio_threshold = compression_ratio_threshold
        self.hotwords = hotwords
        self._model = None

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def cached_path(self) -> Optional[str]:
        """キャッシュ済みならローカルのパス。無ければ None(ネットワークに出ない)。"""
        try:
            from faster_whisper.utils import download_model

            return download_model(self.model_size, local_files_only=True)
        except Exception:
            return None

    def download(self) -> str:
        from faster_whisper.utils import download_model

        return download_model(self.model_size)

    def load(self, model_path: Optional[str] = None) -> None:
        from faster_whisper import WhisperModel

        # モデル名で渡すとキャッシュ済みでも HF Hub に問い合わせるので、
        # ローカルのパスがあればそれで読む
        path = model_path or self.cached_path() or self.model_size
        self._model = WhisperModel(
            path, device=self.device, compute_type=self.compute_type,
            cpu_threads=self.cpu_threads,
        )

    def unload(self) -> None:
        self._model = None

    # ------------------------------------------------------------------
    # セグメントの信頼度判定（幻聴フィルタ本体）
    # ------------------------------------------------------------------

    def is_confident(self, seg, *, strict: bool = True) -> bool:
        """幻聴/低信頼セグメントを弾く。

        strict=True (ライブ): 対数尤度が低い区間を無条件で除外し暴走を防ぐ。
        strict=False (ファイル): Whisper 標準の判定に留め、静かな有効発話を
        取りこぼさない（長尺の連続性を優先）。
        """
        text = (seg.text or "").strip()
        if not text:
            return False
        # 反復ハルシネーション（同一トークン連打）は圧縮率が跳ね上がる
        comp = getattr(seg, "compression_ratio", 0.0) or 0.0
        if comp > self.compression_ratio_threshold:
            return False
        no_speech = getattr(seg, "no_speech_prob", 0.0) or 0.0
        avg_logprob = getattr(seg, "avg_logprob", 0.0) or 0.0
        # 無音確率が高く、かつ対数尤度が低い区間は幻聴とみなす（Whisper 標準）
        if no_speech > self.no_speech_threshold and avg_logprob < self.log_prob_threshold:
            return False
        # ライブのみ: 対数尤度が極端に低い（＝でたらめ）区間も除外
        if strict and avg_logprob < self.log_prob_threshold:
            return False
        return True

    # ------------------------------------------------------------------
    # 低レベル transcribe（共通パラメータを一元化）
    # ------------------------------------------------------------------

    def _transcribe(self, source, *, vad_filter: bool, initial_prompt: Optional[str] = None,
                    beam_size: Optional[int] = None, temperature=None,
                    max_new_tokens: Optional[int] = None, without_timestamps: bool = False):
        if self._model is None:
            raise RuntimeError("Transcriber.load() が未実行です")
        kwargs = {}
        if temperature is not None:
            kwargs["temperature"] = list(temperature)
        if max_new_tokens is not None:
            kwargs["max_new_tokens"] = max_new_tokens
        return self._model.transcribe(
            source,
            language=self.language,
            beam_size=beam_size or self.beam_size,
            vad_filter=vad_filter,
            condition_on_previous_text=self.condition_on_previous_text,
            no_speech_threshold=self.no_speech_threshold,
            log_prob_threshold=self.log_prob_threshold,
            compression_ratio_threshold=self.compression_ratio_threshold,
            hotwords=self.hotwords,
            initial_prompt=initial_prompt,
            without_timestamps=without_timestamps,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # ファイル経路: 信頼できるセグメントを逐次 yield
    # ------------------------------------------------------------------

    def iter_segments(self, source, *, vad_filter: bool, strict: bool = True, **opts):
        """(info, generator) を返す。generator は信頼度フィルタ済みセグメント。"""
        segments, info = self._transcribe(source, vad_filter=vad_filter, **opts)

        def gen() -> Iterator:
            for seg in segments:
                if self.is_confident(seg, strict=strict):
                    yield seg

        return info, gen()

    # ------------------------------------------------------------------
    # ライブ経路: numpy 音声 → 確定テキスト列
    # ------------------------------------------------------------------

    def collect_segments(self, audio: np.ndarray, initial_prompt: Optional[str] = None, **opts) -> list:
        """音声配列を文字起こしし、信頼できるセグメントを SegInfo の列で返す。"""
        from segmenter import SegInfo

        _, segs = self.iter_segments(audio, vad_filter=False, initial_prompt=initial_prompt, **opts)
        return [SegInfo(float(s.start), float(s.end), s.text.strip(),
                        float(getattr(s, "no_speech_prob", 0.0) or 0.0),
                        float(getattr(s, "avg_logprob", 0.0) or 0.0))
                for s in segs if s.text.strip()]

    def collect_texts(
        self, audio: np.ndarray, emit_until_sec: Optional[float] = None, **opts
    ) -> List[str]:
        """音声配列を文字起こしし、信頼できるテキストのみ返す。

        ``emit_until_sec`` を与えると、その時刻までに終わる区間のみ確定し、
        跨いでいる区間は暫定 1 件だけ返す（長尺発話の分割確定に使う）。
        """
        _, segs = self.iter_segments(audio, vad_filter=False, **opts)

        texts: List[str] = []
        fallback: List[str] = []
        for seg in segs:
            text = seg.text.strip()
            if emit_until_sec is None:
                texts.append(text)
                continue
            if seg.end <= emit_until_sec:
                texts.append(text)
            elif seg.start < emit_until_sec:
                fallback.append(text)

        return texts or fallback[:1]
