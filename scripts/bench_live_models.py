#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""複数の候補(モデル・設定)を複数のクリップで replay_live.py に流し、比較表を作る(開発用)。

使い方:
    python3 scripts/bench_live_models.py --clips CLIPS.json --candidates CANDS.json
        [--mode offline|realtime] [--out DIR]

CLIPS.json(ref は無くてもよい):
    [{"name": "zoom_a", "audio": "/abs/a.wav", "ref": "/abs/a.ref.txt"}]

CANDS.json の例(backend は faster か mlx。省略した項目は replay_live.py の既定):
    [{"name": "base_int8_b1", "backend": "faster", "model": "base", "compute_type": "int8",
      "beam": 1, "profile": "balanced", "cpu_threads": 0, "context_chars": 0, "vocab": null},
     {"name": "small_int8_b2", "backend": "faster", "model": "small", "compute_type": "int8",
      "beam": 2, "profile": "accuracy"},
     {"name": "mlx_turbo", "backend": "mlx", "mlx_model": "<HF のリポジトリ名>", "profile": "accuracy"}]

各 clip x candidate を 1 本ずつ別プロセスで実行する(モデルのメモリを持ち越さない)。
出力: DIR/<candidate>/<clip>/(transcript.txt, metrics.json)、DIR/summary.json、DIR/summary.md
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def build_cmd(clip, cand, mode, out_dir):
    cmd = [sys.executable, os.path.join(ROOT, "scripts", "replay_live.py"), clip["audio"],
           "--mode", mode, "--out", out_dir]
    opts = [("profile", "--profile"), ("backend", "--backend"), ("model", "--model"),
            ("mlx_model", "--mlx-model"), ("compute_type", "--compute-type"), ("beam", "--beam"),
            ("cpu_threads", "--cpu-threads"), ("lang", "--lang"), ("vocab", "--vocab"),
            ("context_chars", "--context-chars"), ("speed", "--speed")]
    for key, flag in opts:
        v = cand.get(key)
        if v is not None:
            cmd += [flag, str(v)]
    if clip.get("ref"):
        cmd += ["--ref", clip["ref"]]
    return cmd


def run_all(clips, cands, mode, out):
    results = []
    for cand in cands:
        for clip in clips:
            d = os.path.join(out, cand["name"], clip["name"])
            os.makedirs(d, exist_ok=True)
            row = {"candidate": cand["name"], "clip": clip["name"]}
            try:
                p = subprocess.run(build_cmd(clip, cand, mode, d), cwd=ROOT,
                                   capture_output=True, text=True)
                if p.returncode != 0:
                    row["error"] = (p.stderr or p.stdout)[-500:]
                else:
                    with open(os.path.join(d, "metrics.json"), encoding="utf-8") as f:
                        row.update(json.load(f))
            except Exception as e:
                row["error"] = str(e)
            print(f"{cand['name']} / {clip['name']}: " + (row.get("error", "ok")[:80]))
            results.append(row)
    return results


def _f(v, fmt):
    return "-" if v is None else format(v, fmt)


def write_summary(results, out):
    with open(os.path.join(out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    lines = ["| 候補 | clip | CER | S/D/I | 最長の削除 | RTF | 呼び出し p95(秒) | 呼び出し RTF p95 "
             "| 取りこぼし(秒) | CPU 平均 | RSS ピーク(MB) |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        if "error" in r:
            lines.append(f"| {r['candidate']} | {r['clip']} | error: {r['error'][:60]!r} |" + " |" * 8)
            continue
        e = r.get("errors")
        cer = _f(e["cer"], ".3f") if e else "-"
        sdi = f"{e['S']}/{e['D']}/{e['I']}" if e else "-"
        longest = str(e["longest_deletion"]) if e else "-"
        lines.append(
            f"| {r['candidate']} | {r['clip']} | {cer} | {sdi} | {longest} | {r['rtf']:.3f} "
            f"| {r['call_sec_p95']:.2f} | {r['call_rtf_p95']:.2f} | {_f(r.get('dropped_sec'), '.0f')} "
            f"| {r['cpu_avg']:.0f} | {r['rss_peak_mb']:.0f} |")
    lines += ["", "| 候補 | CER 平均(N で重み付け) | RTF 最大 |", "|---|---|---|"]
    for name in dict.fromkeys(r["candidate"] for r in results):
        rows = [r for r in results if r["candidate"] == name and "error" not in r]
        withref = [r for r in rows if r.get("errors")]
        n = sum(r["errors"]["N"] for r in withref)
        cer = sum(r["errors"]["cer"] * r["errors"]["N"] for r in withref) / n if n else None
        rtf = max((r["rtf"] for r in rows), default=None)
        lines.append(f"| {name} | {_f(cer, '.3f')} | {_f(rtf, '.3f')} |")
    with open(os.path.join(out, "summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description="候補 x クリップの計測")
    ap.add_argument("--clips", required=True)
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--mode", choices=["offline", "realtime"], default="offline")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = args.out or os.path.expanduser(f"~/Documents/Talksribe/bench/{datetime.now():%Y%m%d_%H%M%S}")
    os.makedirs(out, exist_ok=True)
    with open(args.clips, encoding="utf-8") as f:
        clips = json.load(f)
    with open(args.candidates, encoding="utf-8") as f:
        cands = json.load(f)
    results = run_all(clips, cands, args.mode, out)
    write_summary(results, out)
    print(f"summary: {os.path.join(out, 'summary.md')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
