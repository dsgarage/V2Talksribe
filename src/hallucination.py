#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Whisper の幻聴(決まり文句・同じ語句の繰り返し)の判定。

一覧はここに置く(設定ファイルにしない。直すときはコードの変更で、テストを伴う)。
「おめでとうございます」は実際の会話でもよく出るので入れない。
"""
from __future__ import annotations

import re
import unicodedata
from typing import Optional

STOCK_PHRASES = {
    "ja": ["ご視聴ありがとうございました", "ご清聴ありがとうございました", "ありがとうございました",
           "チャンネル登録よろしくお願いします", "チャンネル登録お願いします", "おやすみなさい",
           "字幕は自動生成されています", "最後までご視聴いただきありがとうございました"],
    "en": ["thank you", "thanks", "thank you very much", "thanks for watching", "thank you for watching",
           "please subscribe", "bye", "goodbye", "see you", "see you next time", "you", "peace", "adieu",
           "so", "okay"],
}

_SENT_SPLIT = re.compile(r"[.!?。!?、,]+")


def norm_phrase(text: str) -> str:
    """NFKC、小文字、句読点・記号・空白(unicodedata の category が P*/S*/Z*/C*)を除く。"""
    s = unicodedata.normalize("NFKC", text or "").lower()
    return "".join(ch for ch in s if unicodedata.category(ch)[0] not in ("P", "S", "Z", "C"))


def _units(text: str) -> list:
    return [u for u in _SENT_SPLIT.split(text) if u.strip()]


def collapse_repeats(text: str, min_repeats: int = 3) -> tuple:
    """同じ語句が min_repeats 回以上続く部分を 1 回にまとめる。(新しい文字列, まとめたか)。"""
    changed = False
    # 1) 文の区切りで分けた単位の列で、norm_phrase が同じ単位が続く部分を 1 つにする
    pieces = [(m.group(0), m.start(), m.end()) for m in re.finditer(r"[^.!?。!?、,\s]+[.!?。!?、,]*\s*", text)]
    if pieces:
        keep = []
        i = 0
        while i < len(pieces):
            key = norm_phrase(pieces[i][0])
            j = i
            while j + 1 < len(pieces) and key and norm_phrase(pieces[j + 1][0]) == key:
                j += 1
            run = j - i + 1
            if key and run >= min_repeats:
                keep.append(pieces[i][0])
                changed = True
            else:
                keep.extend(p[0] for p in pieces[i:j + 1])
            i = j + 1
        new = "".join(keep)
        if changed:
            text = new.strip()
    # 2) 区切りの無い日本語のために、同じ部分文字列(2〜12 文字)の連続を 1 回にする。
    #    1 文字の連続('www'、'ははは')と、数字を含む単位('2222')は対象外
    pat = re.compile(r"(.{2,12}?)\1{%d,}" % (min_repeats - 1), re.S)

    def _sub(m):
        unit = m.group(1)
        if any(unicodedata.category(ch)[0] == "N" for ch in unit):
            return m.group(0)
        return unit

    new2 = pat.sub(_sub, text)
    if new2 != text:
        changed = True
        text = new2
    return text, changed


def is_stock_phrase(text: str, lang: Optional[str]) -> bool:
    """collapse_repeats したあとの文を、文の区切りで分けた単位すべてが STOCK_PHRASES[lang] のどれか
    (norm_phrase で比べる)なら True。lang が None か未知なら全言語の一覧で判定。空文字は True。"""
    collapsed, _ = collapse_repeats(text)
    units = _units(collapsed)
    if not units:
        return True
    if lang in STOCK_PHRASES:
        pool = STOCK_PHRASES[lang]
    else:
        pool = [p for v in STOCK_PHRASES.values() for p in v]
    stock = {norm_phrase(p) for p in pool}
    return all(norm_phrase(u) in stock for u in units)
