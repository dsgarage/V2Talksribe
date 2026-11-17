#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""文字起こしの評価用の指標(文字誤り率など)。開発用で、アプリ本体からは使わない。

純 Python で書く(新しい依存を足さない)。
"""
from __future__ import annotations

import re
import unicodedata
from typing import List

_TIME_MARK = re.compile(r"\[\d{1,2}:\d{2}:\d{2}\]")


def normalize_ja(text: str) -> str:
    """NFKC にそろえ、空白・句読点・記号を除く。

    `[HH:MM:SS]` の時刻の印と `#` で始まる行は先に除く。
    落とす文字は unicodedata.category が P* / Z* / S* のもの。改行などの制御文字
    (C*)も、行をつないだときに残らないよう落とす。
    """
    lines = []
    for line in (text or "").splitlines():
        if line.lstrip().startswith("#"):
            continue
        lines.append(_TIME_MARK.sub("", line))
    s = unicodedata.normalize("NFKC", "".join(lines))
    return "".join(ch for ch in s if unicodedata.category(ch)[0] not in ("P", "Z", "S", "C"))


# 操作の種類(表に 1 バイトで持つ)
_MATCH, _SUB, _DEL, _INS = 0, 1, 2, 3


def char_errors(ref: str, hyp: str) -> dict:
    """normalize_ja したあとの文字列で Levenshtein を取り、内訳を返す。

    D は ref にあって hyp に無い文字(抜け)、I は hyp にだけある文字(余り)。
    longest_deletion は連続した削除の最長(抜けの目安)。
    """
    r = normalize_ja(ref)
    h = normalize_ja(hyp)
    n, m = len(r), len(h)
    ops: List[bytearray] = [bytearray(m + 1) for _ in range(n + 1)]
    prev = list(range(m + 1))
    for j in range(1, m + 1):
        ops[0][j] = _INS
    for i in range(1, n + 1):
        cur = [i] + [0] * m
        ops[i][0] = _DEL
        ri = r[i - 1]
        row = ops[i]
        for j in range(1, m + 1):
            if ri == h[j - 1]:
                best, op = prev[j - 1], _MATCH
            else:
                best, op = prev[j - 1] + 1, _SUB
            d = prev[j] + 1
            if d < best:
                best, op = d, _DEL
            ins = cur[j - 1] + 1
            if ins < best:
                best, op = ins, _INS
            cur[j] = best
            row[j] = op
        prev = cur

    s = d = ins = 0
    longest = run = 0
    i, j = n, m
    while i > 0 or j > 0:
        op = ops[i][j]
        if op == _DEL:
            d += 1
            run += 1
            longest = max(longest, run)
            i -= 1
            continue
        run = 0
        if op == _INS:
            ins += 1
            j -= 1
        else:
            if op == _SUB:
                s += 1
            i -= 1
            j -= 1
    return {"N": n, "S": s, "D": d, "I": ins, "cer": (s + d + ins) / max(n, 1),
            "longest_deletion": longest}


def percentile(values: List[float], p: float) -> float:
    """線形補間のパーセンタイル(numpy.percentile と同じ値)。空なら 0.0。"""
    if not values:
        return 0.0
    xs = sorted(float(v) for v in values)
    k = (len(xs) - 1) * (p / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)
