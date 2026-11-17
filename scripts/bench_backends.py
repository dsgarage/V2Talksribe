#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""計測用の別バックエンド(開発用。アプリからは使わない)。

MlxTranscriber は mlx-audio の Whisper を、Transcriber のライブ用インターフェース
(collect_texts / collect_segments)に合わせたもの。replay_live.py の `--backend mlx` から使う。

注意: ライブで使う hotwords は mlx には無いので、hotwords が設定されていて
initial_prompt が None のときに限り、initial_prompt として渡す(近似)。
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

# transcriber.py と同じしきい値
NO_SPEECH_THRESHOLD = 0.6
LOG_PROB_THRESHOLD = -1.0
COMPRESSION_RATIO_THRESHOLD = 2.4


class MlxTranscriber:
    def __init__(self, repo: str, *, language="ja", beam_size=1):
        self.repo = repo
        self.model_size = f"mlx:{repo}"
        self.language = language
        self.beam_size = beam_size
        self.hotwords = None
        self._model = None

    def cached_path(self):
        return None

    def download(self):
        return self.repo

    def load(self, model_path=None):
        from mlx_audio.stt.utils import load

        self._model = load(self.repo)

    def _generate(self, audio, initial_prompt=None):
        if self._model is None:
            raise RuntimeError("MlxTranscriber.load() が未実行です")
        if initial_prompt is None and self.hotwords:
            initial_prompt = self.hotwords
        return self._model.generate(np.asarray(audio, dtype=np.float32), language=self.language,
                                    condition_on_previous_text=False, initial_prompt=initial_prompt)

    def _confident(self, seg: dict) -> bool:
        """Transcriber.is_confident(strict=True) と同じ判定を dict に対して行う。"""
        text = (seg.get("text") or "").strip()
        if not text:
            return False
        if (seg.get("compression_ratio", 0.0) or 0.0) > COMPRESSION_RATIO_THRESHOLD:
            return False
        no_speech = seg.get("no_speech_prob", 0.0) or 0.0
        avg_logprob = seg.get("avg_logprob", 0.0) or 0.0
        if no_speech > NO_SPEECH_THRESHOLD and avg_logprob < LOG_PROB_THRESHOLD:
            return False
        if avg_logprob < LOG_PROB_THRESHOLD:
            return False
        return True

    def collect_segments(self, audio, initial_prompt=None) -> List[Tuple[float, float, str]]:
        out = self._generate(audio, initial_prompt)
        return [(float(s["start"]), float(s["end"]), s["text"].strip())
                for s in (out.segments or []) if self._confident(s)]

    def collect_texts(self, audio, emit_until_sec: Optional[float] = None, initial_prompt=None):
        segs = self.collect_segments(audio, initial_prompt)
        if emit_until_sec is None:
            return [t for _, _, t in segs]
        texts = [t for s, e, t in segs if e <= emit_until_sec]
        fallback = [t for s, e, t in segs if e > emit_until_sec and s < emit_until_sec]
        return texts or fallback[:1]
