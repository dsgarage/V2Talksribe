#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Zoom 会議のライブ文字起こし＋定期スクリーンショット収録セッション。

BlackHole 経由で会議音声を連続収録し、一定長のセグメントごとに文字起こしして
追記する（途中クラッシュしても直前セグメントまで残る）。同時に全画面スクショを
一定間隔で保存する。開始時にシステム音声出力を MultiOutput へ切り替え、終了時に
元へ戻す。

使い方:
    python3 scripts/zoom_session.py \
        --start-at 12:00 --minutes 90 --shot-interval 60 \
        --device 1 --model base --lang ja --segment 300
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import queue
import subprocess
import sys
import threading
import time

import numpy as np
import sounddevice as sd
import soundfile as sf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from transcriber import Transcriber  # noqa: E402

stop_event = threading.Event()


def log(msg: str):
    ts = dt.datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# 音声出力デバイスの切替（SwitchAudioSource）
# ---------------------------------------------------------------------------
def get_output_device() -> str:
    try:
        return subprocess.check_output(
            ["SwitchAudioSource", "-c", "-t", "output"], text=True
        ).strip()
    except Exception as e:  # noqa: BLE001
        log(f"出力デバイス取得失敗: {e}")
        return ""


def set_output_device(name: str) -> bool:
    try:
        subprocess.run(
            ["SwitchAudioSource", "-t", "output", "-s", name],
            check=True, capture_output=True, text=True,
        )
        log(f"システム音声出力を切替: {name}")
        return True
    except Exception as e:  # noqa: BLE001
        log(f"出力切替失敗({name}): {e}")
        return False


# ---------------------------------------------------------------------------
# スクリーンショット
# ---------------------------------------------------------------------------
def shot_worker(shot_dir: str, interval: float):
    n = 0
    # 起動直後に 1 枚
    while not stop_event.is_set():
        n += 1
        ts = dt.datetime.now().strftime("%H%M%S")
        path = os.path.join(shot_dir, f"{n:04d}_{ts}.png")
        try:
            subprocess.run(["screencapture", "-x", path], check=False)
        except Exception as e:  # noqa: BLE001
            log(f"スクショ失敗: {e}")
        # interval を細切れに待って停止に即応
        waited = 0.0
        while waited < interval and not stop_event.is_set():
            time.sleep(0.5)
            waited += 0.5
    log(f"スクショ終了: {n} 枚")


# ---------------------------------------------------------------------------
# 音声収録（連続）→ セグメント WAV
# ---------------------------------------------------------------------------
def resolve_input_device(spec: str) -> int:
    """--device の指定を実際のデバイス番号へ解決する。

    数字ならそのまま番号として扱う。文字列なら入力デバイス名に対する
    部分一致（大小無視）で番号を引く。番号ずれ（デバイス着脱で index が
    変わる）に強くするため、収録開始時点で毎回解決する。
    """
    spec = str(spec).strip()
    if spec.lstrip("-").isdigit():
        return int(spec)
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] > 0 and spec.lower() in d["name"].lower():
            return i
    raise ValueError(f"入力デバイスが見つかりません: {spec!r}")


def record_and_segment(device, seg_dir: str, seg_q: "queue.Queue",
                       segment_sec: float, session_start: dt.datetime):
    device = resolve_input_device(device)
    dev_info = sd.query_devices(device)
    sr = int(dev_info["default_samplerate"])
    ch = min(2, int(dev_info["max_input_channels"]))
    log(f"収録開始: device={device} '{dev_info['name']}' sr={sr} ch={ch}")

    audio_q: "queue.Queue" = queue.Queue()

    def cb(indata, frames, time_info, status):  # noqa: ANN001
        if status:
            pass
        audio_q.put(indata.copy())

    buf = np.zeros((0,), dtype=np.float32)
    seg_target = int(segment_sec * sr)
    seg_idx = 0

    def flush_segment(samples: np.ndarray, idx: int):
        offset = idx * segment_sec
        path = os.path.join(seg_dir, f"seg_{idx:03d}.wav")
        sf.write(path, samples, sr)
        seg_q.put((idx, path, offset))

    with sd.InputStream(device=device, samplerate=sr, channels=ch,
                        dtype="float32", callback=cb, blocksize=0):
        while not stop_event.is_set():
            try:
                block = audio_q.get(timeout=0.5)
            except queue.Empty:
                continue
            mono = block.mean(axis=1) if block.ndim > 1 else block
            buf = np.concatenate([buf, mono.astype(np.float32)])
            while len(buf) >= seg_target:
                flush_segment(buf[:seg_target], seg_idx)
                buf = buf[seg_target:]
                seg_idx += 1
        # 残りをフラッシュ
        if len(buf) > sr * 1.0:  # 1 秒未満は捨てる
            flush_segment(buf, seg_idx)
    seg_q.put(None)  # 終端
    log("収録終了")


# ---------------------------------------------------------------------------
# 文字起こしワーカー
# ---------------------------------------------------------------------------
def transcribe_worker(seg_q: "queue.Queue", transcript_path: str,
                      model: str, lang, session_start: dt.datetime):
    tr = Transcriber(
        model, language=lang, beam_size=1,
        condition_on_previous_text=True,  # 長尺の連続性を優先
    )
    log(f"Whisper {model} 読み込み中...")
    tr.load()
    log("Whisper 読み込み完了")

    total_lines = 0
    with open(transcript_path, "a", encoding="utf-8") as f:
        while True:
            item = seg_q.get()
            if item is None:
                break
            idx, path, offset = item
            try:
                # ファイルパス渡し → faster-whisper 側で 16k へ復号/リサンプル
                _, segs = tr.iter_segments(path, vad_filter=True, strict=False)
                seg_lines = 0
                for seg in segs:
                    text = (seg.text or "").strip()
                    if not text:
                        continue
                    when = session_start + dt.timedelta(seconds=offset + seg.start)
                    f.write(f"[{when.strftime('%H:%M:%S')}] {text}\n")
                    seg_lines += 1
                f.flush()
                total_lines += seg_lines
                log(f"seg {idx:03d} 文字起こし: {seg_lines} 行 (累計 {total_lines})")
            except Exception as e:  # noqa: BLE001
                log(f"seg {idx:03d} 文字起こし失敗: {type(e).__name__}: {e}")
    log(f"文字起こし終了: 累計 {total_lines} 行")


# ---------------------------------------------------------------------------
def wait_until(target: dt.datetime):
    while True:
        now = dt.datetime.now()
        if now >= target:
            return
        remain = (target - now).total_seconds()
        log(f"開始待機中... {target.strftime('%H:%M:%S')} まで残り {int(remain)}s")
        time.sleep(min(30, max(1, remain)))


def parse_start_at(s: str) -> dt.datetime:
    if s.lower() == "now":
        return dt.datetime.now()
    hh, mm = s.split(":")
    now = dt.datetime.now()
    target = now.replace(hour=int(hh), minute=int(mm), second=0, microsecond=0)
    if target < now:
        target += dt.timedelta(days=1)
    return target


def main() -> int:
    ap = argparse.ArgumentParser(description="Zoom ライブ文字起こし＋スクショ収録")
    ap.add_argument("--start-at", default="now", help="開始時刻 HH:MM または now")
    ap.add_argument("--minutes", type=float, default=90.0, help="収録分数")
    ap.add_argument("--shot-interval", type=float, default=60.0, help="スクショ間隔秒")
    ap.add_argument("--device", default="BlackHole",
                    help="入力デバイス。番号か名前の部分一致(既定: BlackHole)")
    ap.add_argument("--model", default="base")
    ap.add_argument("--lang", default="ja")
    ap.add_argument("--segment", type=float, default=300.0, help="文字起こしセグメント秒")
    ap.add_argument("--switch-to", default="MultiOutput", help="開始時に切替える出力名")
    ap.add_argument("--no-switch", action="store_true", help="音声出力を切替えない")
    ap.add_argument("--output", default=None, help="出力ディレクトリ")
    args = ap.parse_args()

    start_dt = parse_start_at(args.start_at)
    if args.start_at.lower() != "now":
        wait_until(start_dt)
    session_start = dt.datetime.now()

    stamp = session_start.strftime("%Y%m%d_%H%M%S")
    outdir = args.output or os.path.expanduser(
        f"~/Documents/Talksribe/zoom_session_{stamp}"
    )
    seg_dir = os.path.join(outdir, "segments")
    shot_dir = os.path.join(outdir, "screenshots")
    os.makedirs(seg_dir, exist_ok=True)
    os.makedirs(shot_dir, exist_ok=True)
    transcript_path = os.path.join(outdir, "transcript.md")
    with open(transcript_path, "w", encoding="utf-8") as f:
        f.write(f"# Zoom セッション文字起こし {session_start.strftime('%Y-%m-%d %H:%M')}\n")
        f.write(f"# device={args.device} model={args.model} lang={args.lang} "
                f"segment={args.segment}s\n\n")
    log(f"出力先: {outdir}")

    # 音声出力切替
    prev_output = ""
    if not args.no_switch:
        prev_output = get_output_device()
        set_output_device(args.switch_to)

    seg_q: "queue.Queue" = queue.Queue()
    threads = [
        threading.Thread(target=record_and_segment,
                         args=(args.device, seg_dir, seg_q, args.segment, session_start),
                         daemon=True),
        threading.Thread(target=transcribe_worker,
                         args=(seg_q, transcript_path, args.model, args.lang, session_start),
                         daemon=True),
        threading.Thread(target=shot_worker,
                         args=(shot_dir, args.shot_interval), daemon=True),
    ]
    for t in threads:
        t.start()

    # 本体待機（停止に即応）
    end_dt = session_start + dt.timedelta(minutes=args.minutes)
    try:
        while dt.datetime.now() < end_dt and not stop_event.is_set():
            time.sleep(1)
    except KeyboardInterrupt:
        log("中断シグナル受信")
    stop_event.set()
    log("停止処理中... 収録スレッドの後片付けを待機")

    # 収録スレッド(0)とスクショ(2)を join、その後 seg_q 終端で文字起こし完了
    threads[0].join(timeout=30)
    threads[2].join(timeout=10)
    threads[1].join(timeout=600)  # 残セグメントの文字起こし完了待ち

    # 音声出力復元
    if not args.no_switch and prev_output:
        set_output_device(prev_output)

    log(f"完了: {transcript_path}")
    print(f"\n=== セッション完了 ===\n出力: {outdir}\n"
          f"文字起こし: {transcript_path}\nスクショ: {shot_dir}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
