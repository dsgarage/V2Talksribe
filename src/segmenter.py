#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""長い発話の窓の確定点の決め方と、確定の重複除去(純関数だけ)。"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SegInfo:
    start: float
    end: float
    text: str
    no_speech_prob: float = 0.0
    avg_logprob: float = 0.0


@dataclass
class CommitPlan:
    segs: list          # 確定する SegInfo
    cut_sec: float

    @property
    def texts(self) -> list:
        return [s.text for s in self.segs if s.text.strip()]


def plan_window_commit(segs: list, buf_sec: float, *, force: bool, tail_guard_sec: float = 1.0,
                       cut_margin_sec: float = 0.2, empty_keep_sec: float = 2.0) -> CommitPlan:
    """segs は時刻順、start/end はバッファの先頭からの秒。"""
    if not segs:
        return CommitPlan([], max(0.0, buf_sec - empty_keep_sec))
    if force:
        return CommitPlan(list(segs), buf_sec)
    last = segs[-1]
    if last.end <= buf_sec - tail_guard_sec:
        committed = list(segs)
    else:
        committed = list(segs[:-1])
    if len(committed) == len(segs):
        cut = min(buf_sec, last.end)
    elif not committed:
        cut = 0.0
    else:
        first_unc = segs[len(committed)]
        cut = max(committed[-1].end, first_unc.start - cut_margin_sec, 0.0)
    cut = min(max(cut, 0.0), buf_sec)
    return CommitPlan(committed, cut)


def _ascii_alnum(ch: str) -> bool:
    return ch.isascii() and ch.isalnum()


def dedupe_prefix(prev_tail: str, text: str, min_len: int = 4, max_len: int = 20) -> str:
    """prev_tail の末尾と text の先頭が一致する最長の部分(長さ min_len〜max_len)を text から取り除く。

    比べるときは両方とも空白を除いた文字列で比べ、取り除く量は text の元の位置に戻して切る。
    切れ目が英単語の途中になる(一致の最後の文字と text の次の文字がどちらも ASCII の英数字)
    長さは採用しない。
    """
    if not prev_tail or not text:
        return text
    a = "".join(ch for ch in prev_tail if not ch.isspace())
    idx_map = [i for i, ch in enumerate(text) if not ch.isspace()]
    b = "".join(text[i] for i in idx_map)
    for n in range(min(max_len, len(a), len(b)), min_len - 1, -1):
        if n < min_len:
            break
        if a[-n:] == b[:n]:
            cut_at = idx_map[n - 1] + 1
            if cut_at < len(text) and _ascii_alnum(text[cut_at]) and _ascii_alnum(text[cut_at - 1]):
                continue
            return text[cut_at:].strip()
    return text
