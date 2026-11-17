#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""replay_live.py の結果フォルダを、BASE を仮の基準にして並べて比べる。

使い方:
    compare_runs.py BASE_DIR OTHER_DIR [OTHER_DIR ...] [--lang ja|en] [--out FILE.md]

各フォルダの transcript.txt(`NNNN 本文`)と metrics.json を読む。目的は、同じ録音について
「ファイルとして一括で文字起こしした結果(--mode batch)」を基準に、ライブの経路の結果が
どれだけ離れているかを見ること。差の大きさは「切り方で失っている分」の目安になる
(モデルの限界は batch の側にも出る)。
"""
from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import eval_metrics  # noqa: E402

HEADER = ["フォルダ", "一致 CER", "一致 WER", "行数", "1 行の中央値", "短い行", "決まり文句", "繰り返し",
          "RTF", "呼び出し p95", "呼び出し最大", "落とした区間", "幻聴で捨てた", "部分更新"]


def read_lines(d):
    lines = []
    with open(os.path.join(d, "transcript.txt"), encoding="utf-8") as f:
        for raw in f.read().splitlines():
            parts = raw.split(" ", 1)
            lines.append(parts[1] if len(parts) == 2 else "")
    return lines


def read_metrics(d):
    try:
        with open(os.path.join(d, "metrics.json"), encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def _f(v, fmt="{:.3f}"):
    return "-" if v is None else fmt.format(v)


def make_row(name, base_lines, lines, metrics, lang, is_base=False):
    if is_base:
        cer = wer = None
    else:
        ag = eval_metrics.agreement(base_lines, lines, lang)
        cer = ag["cer"]["cer"]
        wer = ag["wer"]["wer"] if "wer" in ag else None
    ls = metrics.get("line_stats") or eval_metrics.line_stats(lines, lang)
    st = metrics.get("live_stats") or {}
    return [name, _f(cer), _f(wer), str(ls["lines"]), _f(ls["units_median"], "{:.1f}"),
            _f(ls["short_ratio"], "{:.2f}"), str(ls["stock_lines"]), str(ls["repeat_lines"]),
            _f(metrics.get("rtf"), "{:.2f}"), _f(metrics.get("call_sec_p95"), "{:.2f}"),
            _f(metrics.get("call_sec_max"), "{:.2f}"),
            str(st["utt_skipped"]) if "utt_skipped" in st else "-",
            str(st["halluc_dropped"]) if "halluc_dropped" in st else "-",
            str(st["partials"]) if "partials" in st else "-"]


def build_table(base, others, lang):
    base_lines = read_lines(base)
    rows = [make_row(os.path.basename(os.path.normpath(base)) + "(基準)", base_lines, base_lines,
                     read_metrics(base), lang, is_base=True)]
    for d in others:
        rows.append(make_row(os.path.basename(os.path.normpath(d)), base_lines, read_lines(d),
                             read_metrics(d), lang))
    out = ["| " + " | ".join(HEADER) + " |", "|" + "|".join(["---"] * len(HEADER)) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description="replay_live の結果を BASE を基準に比べる")
    ap.add_argument("base")
    ap.add_argument("others", nargs="+")
    ap.add_argument("--lang", choices=["ja", "en"], default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    for d in [args.base] + args.others:
        if not os.path.exists(os.path.join(d, "transcript.txt")):
            print(f"transcript.txt が見つかりません: {d}", file=sys.stderr)
            return 1
    table = build_table(args.base, args.others, args.lang)
    print(table)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(table + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
