#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""録音済みの音声を、マイクなしでライブ経路(engine)に流して計測する。

- offline: 音声を chunk_sec ごとに engine._process_audio へ渡す。決定的で、速度の計測に向く。
- realtime: ファイル源のキャプチャ(start_file_capture)で実時間どおりに流し、engine を
  本物と同じ経路で動かす。実時間に追いつくか(tail_sec、取りこぼし)を見る。
- batch: 同じ録音をファイルとして一括で文字起こしする(VAD あり、前の文を引き継ぐ)。
  ライブの結果との比較の基準にする(scripts/compare_runs.py)。

マイクは開かない。設定は AppSettings() を新しく作り、本物の config.json は読まない・書かない。
ログと保存先は --out の下に向け、本物の ~/Documents/Talksribe には書かない。

使い方:
    python3 scripts/replay_live.py <audio> [--mode offline|realtime|batch] [--profile balanced]
        [--model base] [--lang ja] [--out DIR] [--ref REF.txt]
        [--tune KEY=VALUE ...] [--silence-end 1.4] [--asr-normalize off] [--vocab FILE]
出力: DIR/transcript.txt(確定した順の番号と本文)、DIR/metrics.json
"""
from __future__ import annotations

import argparse
import contextlib
import dataclasses
import json
import os
import re
import sys
import threading
import time
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import applog  # noqa: E402
import eval_metrics  # noqa: E402
from eval_metrics import char_errors, percentile  # noqa: E402
from settings import (POWER_PROFILES, AppSettings, apply_tuning_overrides,  # noqa: E402
                      hotwords_from_vocabulary, load_vocabulary)

SR = 16000


# ----------------------------------------------------------------------
# 計測用のラッパー
# ----------------------------------------------------------------------

class TimedTranscriber:
    """本物の Transcriber(または MlxTranscriber)を包み、呼び出しごとに (種類, 経過秒, 音声秒) を記録する。"""

    def __init__(self, inner):
        object.__setattr__(self, "inner", inner)
        object.__setattr__(self, "calls", [])

    def __setattr__(self, name, value):
        if name in ("inner", "calls"):
            object.__setattr__(self, name, value)
        else:
            setattr(self.inner, name, value)

    def _timed(self, kind, fn, audio, *a, **k):
        t0 = time.perf_counter()
        try:
            return fn(audio, *a, **k)
        finally:
            self.calls.append((kind, time.perf_counter() - t0, len(audio) / SR))

    def collect_texts(self, audio, *a, **k):
        return self._timed("texts", self.inner.collect_texts, audio, *a, **k)

    def __getattr__(self, name):
        # collect_segments は内側に在るときだけ(計測つきで)見せる。偽物の Transcriber では
        # 属性が無いことにして、engine が collect_texts に切り替えられるようにする
        if name == "collect_segments":
            fn = getattr(self.inner, "collect_segments")   # 無ければ AttributeError
            return lambda audio, *a, **k: self._timed("segments", fn, audio, *a, **k)
        return getattr(self.inner, name)


class ResourceMonitor:
    """1 秒ごとに CPU 使用率と RSS(子プロセスを含む)を取る。"""

    def __init__(self):
        import psutil

        self._psutil = psutil
        self._proc = psutil.Process()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self.cpu = []
        self.rss_peak = 0

    def _sample(self):
        try:
            self.cpu.append(self._proc.cpu_percent(None))
            rss = self._proc.memory_info().rss
            for c in self._proc.children(recursive=True):
                try:
                    rss += c.memory_info().rss
                except Exception:
                    pass
            self.rss_peak = max(self.rss_peak, rss)
        except Exception:
            pass

    def _run(self):
        self._proc.cpu_percent(None)
        while not self._stop.wait(1.0):
            self._sample()

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join(2)
        self._sample()

    @property
    def cpu_avg(self):
        return sum(self.cpu) / len(self.cpu) if self.cpu else 0.0

    @property
    def rss_peak_mb(self):
        return self.rss_peak / (1024 * 1024)


@contextlib.contextmanager
def isolated_log(out_dir):
    """ログを out_dir に向ける(本物の debug.log に書かない)。"""
    old = (applog.LOG_DIR, applog.LOG_PATH)
    applog._reset_logger()
    applog.LOG_DIR = out_dir
    applog.LOG_PATH = os.path.join(out_dir, "engine.log")
    try:
        yield
    finally:
        applog._reset_logger()
        applog.LOG_DIR, applog.LOG_PATH = old


def _audio_seconds(path):
    from faster_whisper.audio import decode_audio

    audio = decode_audio(path, sampling_rate=SR)
    return audio, len(audio) / SR


def _call_stats(calls):
    secs = [c[1] for c in calls]
    rtfs = [c[1] / c[2] for c in calls if c[2] > 0]
    return {
        "calls": len(calls),
        "call_sec_p50": percentile(secs, 50),
        "call_sec_p95": percentile(secs, 95),
        "call_sec_max": max(secs) if secs else 0.0,
        "call_rtf_p95": percentile(rtfs, 95),
    }


def _write_outputs(out_dir, transcripts, metrics):
    with open(os.path.join(out_dir, "transcript.txt"), "w", encoding="utf-8") as f:
        for i, text in enumerate(transcripts, 1):
            f.write(f"{i:04d} {text}\n")
    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)


def _drain(eng, transcripts, alerts=None):
    import queue

    while True:
        try:
            ev = eng.events.get_nowait()
        except queue.Empty:
            return
        if ev[0] == "transcript":
            transcripts.append(ev[2])
        elif ev[0] == "alert" and alerts is not None:
            alerts.append(ev[1])


def _apply_live_thresholds(transcriber, tuning, vocab_terms):
    """engine._begin_stream と同じ設定(語彙・ライブの閾値)を transcriber に入れる。"""
    transcriber.hotwords = hotwords_from_vocabulary(vocab_terms)
    if hasattr(transcriber, "no_speech_threshold"):
        transcriber.no_speech_threshold = tuning.live_no_speech_threshold
        transcriber.log_prob_threshold = tuning.live_log_prob_threshold


@contextlib.contextmanager
def patched_vocabulary(vocab_terms):
    """engine の語彙の読み込みを差し替える(本物の ~/Documents/Talksribe/vocabulary.txt を読まない)。"""
    import engine as engine_module

    old = engine_module.load_vocabulary
    engine_module.load_vocabulary = lambda *a, **k: list(vocab_terms)
    try:
        yield
    finally:
        engine_module.load_vocabulary = old


def _line_metrics(transcripts, lang):
    return eval_metrics.line_stats(transcripts, lang)


# ----------------------------------------------------------------------
# offline
# ----------------------------------------------------------------------

def run_offline(audio_path, settings, transcriber, out_dir, *, vocab_terms=None) -> dict:
    """マイクもワーカーも使わず、engine の処理を同じ順で直接呼ぶ。transcriber は TimedTranscriber。"""
    from engine import TranscriptionEngine
    from vad import VoiceActivityDetector

    os.makedirs(out_dir, exist_ok=True)
    audio, audio_sec = _audio_seconds(audio_path)
    mon = ResourceMonitor()
    with isolated_log(out_dir):
        eng = TranscriptionEngine(settings, transcriber_factory=lambda s, l, b: transcriber,
                                  save_dir=out_dir, record_audio=False, prune_audio=False)
        eng.vad = VoiceActivityDetector(eng.vad_backend, eng.sr, aggressiveness=eng.vad_aggressiveness)
        eng.vad.load()
        eng.transcriber = transcriber
        # _begin_stream を通らないので、同じ設定をここで入れる
        _apply_live_thresholds(transcriber, eng.tuning, vocab_terms or [])
        eng._reset_live_state()
        eng._ensure_live_preprocessor()
        eng.save_path = os.path.join(out_dir, "engine.txt")
        with open(eng.save_path, "w", encoding="utf-8") as f:
            f.write("# replay offline\n\n")
        eng.is_recording = True

        n = int(SR * eng.chunk_sec)
        mon.start()
        t0 = time.perf_counter()
        for i in range(0, len(audio), n):
            eng._process_audio(audio[i:i + n])
        if len(eng.speech_buf) > 0:
            eng._finalize_all(reason="end")
        proc_sec = time.perf_counter() - t0
        mon.stop()

        transcripts = []
        _drain(eng, transcripts)

    metrics = {
        "audio_sec": audio_sec, "proc_sec": proc_sec, "rtf": proc_sec / audio_sec if audio_sec else 0.0,
        **_call_stats(transcriber.calls),
        "tail_sec": None, "dropped_sec": None, "alerts": [],
        "cpu_avg": mon.cpu_avg, "rss_peak_mb": mon.rss_peak_mb, "errors": None,
        "tuning": dataclasses.asdict(settings.live_tuning), "live_stats": eng.live_stats,
        "line_stats": _line_metrics(transcripts, settings.language),
    }
    _write_outputs(out_dir, transcripts, metrics)
    return metrics


# ----------------------------------------------------------------------
# realtime
# ----------------------------------------------------------------------

def run_realtime(audio_path, settings, transcriber, out_dir, *, speed=1.0, vocab_terms=None) -> dict:
    """ファイル源のキャプチャで実時間どおりに流す。transcriber は TimedTranscriber(未読み込みでもよい)。"""
    import queue

    from capture import start_file_capture
    from engine import TranscriptionEngine

    os.makedirs(out_dir, exist_ok=True)
    _, audio_sec = _audio_seconds(audio_path)
    mon = ResourceMonitor()
    transcripts, alerts, stamps = [], [], []
    with isolated_log(out_dir), patched_vocabulary(vocab_terms or []):
        eng = TranscriptionEngine(
            settings,
            capture_factory=lambda sr, cs, name, rec: start_file_capture(sr, cs, audio_path, speed),
            transcriber_factory=lambda s, l, b: transcriber,
            save_dir=out_dir, record_audio=False, prune_audio=False)

        def pump(timeout):
            try:
                ev = eng.events.get(timeout=timeout)
            except queue.Empty:
                return None
            if ev[0] == "transcript":
                transcripts.append(ev[2])
                stamps.append(time.monotonic())
            elif ev[0] == "alert":
                alerts.append(ev[1])
            return ev

        eng.run()
        mon.start()
        t_start = time.monotonic()
        eng.start()
        play_end = t_start + (audio_sec / speed if speed > 0 else 0.0)
        deadline = play_end + 3.0
        while time.monotonic() < deadline:
            pump(0.2)
        eng.stop()
        stop_deadline = time.monotonic() + 120.0
        while time.monotonic() < stop_deadline:
            ev = pump(0.2)
            if ev is not None and ev[0] == "toggle" and ev[1] == "録音開始":
                break
        proc_sec = time.monotonic() - t_start
        eng.shutdown()
        if eng._worker is not None:
            eng._worker.join(5)
        mon.stop()
        _drain(eng, transcripts, alerts)

    tail = max(0.0, stamps[-1] - play_end) if stamps else 0.0
    drops = [a for a in alerts if "処理の遅れで音声" in a]
    dropped = 0.0
    for a in drops:
        m = re.search(r"処理の遅れで音声\s*(\d+)秒", a)
        if m:
            dropped += float(m.group(1))
    metrics = {
        "audio_sec": audio_sec, "proc_sec": proc_sec, "rtf": proc_sec / audio_sec if audio_sec else 0.0,
        **_call_stats(transcriber.calls),
        "tail_sec": tail, "dropped_sec": dropped, "alerts": alerts,
        "cpu_avg": mon.cpu_avg, "rss_peak_mb": mon.rss_peak_mb, "errors": None,
        "tuning": dataclasses.asdict(settings.live_tuning), "live_stats": eng.live_stats,
        "line_stats": _line_metrics(transcripts, settings.language),
    }
    _write_outputs(out_dir, transcripts, metrics)
    return metrics


# ----------------------------------------------------------------------
# batch
# ----------------------------------------------------------------------

def run_batch(audio_path, settings, transcriber, out_dir, *, vocab_terms=None) -> dict:
    """同じ録音をファイルとして一括で文字起こしする(ファイル文字起こしと同じ流儀)。

    transcriber は condition_on_previous_text=True の Transcriber を包んだ TimedTranscriber。
    """
    os.makedirs(out_dir, exist_ok=True)
    _, audio_sec = _audio_seconds(audio_path)
    transcriber.hotwords = hotwords_from_vocabulary(vocab_terms or [])
    mon = ResourceMonitor()
    mon.start()
    t0 = time.perf_counter()
    info, segs = transcriber.inner.iter_segments(audio_path, vad_filter=True, strict=False)
    transcripts = [s.text.strip() for s in segs if s.text.strip()]
    proc_sec = time.perf_counter() - t0
    mon.stop()
    metrics = {
        "audio_sec": audio_sec, "proc_sec": proc_sec, "rtf": proc_sec / audio_sec if audio_sec else 0.0,
        **_call_stats([("batch", proc_sec, audio_sec)]),
        "tail_sec": None, "dropped_sec": None, "alerts": [],
        "cpu_avg": mon.cpu_avg, "rss_peak_mb": mon.rss_peak_mb, "errors": None,
        "tuning": dataclasses.asdict(settings.live_tuning), "live_stats": None,
        "line_stats": _line_metrics(transcripts, settings.language),
    }
    _write_outputs(out_dir, transcripts, metrics)
    return metrics


# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------

def build_transcriber(args, lang, beam, *, load, condition_on_previous_text=False):
    if args.backend == "mlx":
        from bench_backends import MlxTranscriber

        inner = MlxTranscriber(args.mlx_model, language=lang, beam_size=beam)
    else:
        from transcriber import Transcriber

        inner = Transcriber(args.model, compute_type=args.compute_type,
                            cpu_threads=args.cpu_threads, language=lang, beam_size=beam,
                            condition_on_previous_text=condition_on_previous_text)
    if load:
        inner.load()
    return TimedTranscriber(inner)


def main() -> int:
    ap = argparse.ArgumentParser(description="録音済み音声をライブ経路に流して計測する(マイクなし)")
    ap.add_argument("audio")
    ap.add_argument("--mode", choices=["offline", "realtime", "batch"], default="offline")
    ap.add_argument("--profile", choices=list(POWER_PROFILES), default="balanced")
    ap.add_argument("--model", default=None, help="既定はプロファイルの model_size")
    ap.add_argument("--compute-type", default="int8")
    ap.add_argument("--beam", type=int, default=None, help="既定はプロファイルの beam_size")
    ap.add_argument("--cpu-threads", type=int, default=0)
    ap.add_argument("--lang", default="ja")
    ap.add_argument("--vocab", default=None)
    ap.add_argument("--context-chars", type=int, default=None)
    ap.add_argument("--tune", action="append", default=[], metavar="KEY=VALUE",
                    help="LiveTuning の値を上書きする(何度でも書ける)")
    ap.add_argument("--silence-end", type=float, default=None)
    ap.add_argument("--preroll", type=float, default=None)
    ap.add_argument("--gate-margin-db", type=float, default=None)
    ap.add_argument("--asr-normalize", choices=["on", "off"], default=None)
    ap.add_argument("--agc", choices=["on", "off"], default=None)
    ap.add_argument("--halluc-filter", choices=["on", "off"], default=None)
    ap.add_argument("--temperature", default=None, help='例: "0.0,0.4"')
    ap.add_argument("--backend", choices=["faster", "mlx"], default="faster")
    ap.add_argument("--mlx-model", default=None)
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--ref", default=None)
    args = ap.parse_args()

    if not os.path.exists(args.audio):
        print(f"ファイルが見つかりません: {args.audio}", file=sys.stderr)
        return 1
    if args.backend == "mlx" and not args.mlx_model:
        print("--backend mlx には --mlx-model が必要です", file=sys.stderr)
        return 1
    if args.backend == "mlx" and args.mode == "batch":
        print("--mode batch は --backend faster だけに対応しています", file=sys.stderr)
        return 1

    prof = POWER_PROFILES[args.profile]
    args.model = args.model or prof["model_size"]
    beam = args.beam if args.beam is not None else prof["beam_size"]
    out = args.out or os.path.expanduser(
        f"~/Documents/Talksribe/bench/replay_{datetime.now():%Y%m%d_%H%M%S}")
    os.makedirs(out, exist_ok=True)

    settings = AppSettings()
    settings.save = lambda *a, **k: None   # 本物の config.json に書かない
    settings.power_profile = args.profile
    settings.model_size = args.model
    settings.language = args.lang

    # --tune を先に、近道を後に当てる(両方あれば近道が勝つ)
    pairs = list(args.tune or [])
    if args.context_chars is not None:
        pairs.append(f"context_chars={args.context_chars}")
    shortcuts = (("silence_end_sec", args.silence_end), ("preroll_sec", args.preroll),
                 ("gate_margin_db", args.gate_margin_db), ("asr_normalize", args.asr_normalize),
                 ("agc_enabled", args.agc), ("halluc_filter", args.halluc_filter),
                 ("temperature", args.temperature))
    pairs += [f"{k}={v}" for k, v in shortcuts if v is not None]
    try:
        apply_tuning_overrides(settings.live_tuning, pairs)
    except ValueError as e:
        print(f"--tune: {e}", file=sys.stderr)
        return 2

    # 語彙は --vocab のファイルだけ(本物の ~/Documents/Talksribe/vocabulary.txt は読まない)
    vocab_terms = load_vocabulary(args.vocab) if args.vocab else []

    timed = build_transcriber(args, args.lang, beam, load=(args.mode in ("offline", "batch")),
                              condition_on_previous_text=(args.mode == "batch"))
    if args.mode == "offline":
        metrics = run_offline(args.audio, settings, timed, out, vocab_terms=vocab_terms)
    elif args.mode == "batch":
        metrics = run_batch(args.audio, settings, timed, out, vocab_terms=vocab_terms)
    else:
        metrics = run_realtime(args.audio, settings, timed, out, speed=args.speed,
                               vocab_terms=vocab_terms)

    full = {"audio": args.audio, "mode": args.mode, "backend": args.backend,
            "model": args.mlx_model if args.backend == "mlx" else args.model,
            "compute_type": args.compute_type, "beam": beam, "cpu_threads": args.cpu_threads,
            "profile": args.profile, "lang": args.lang, "vocab": args.vocab,
            "context_chars": settings.live_tuning.context_chars}
    full.update(metrics)
    if args.ref:
        with open(args.ref, encoding="utf-8") as f:
            ref = f.read()
        with open(os.path.join(out, "transcript.txt"), encoding="utf-8") as f:
            hyp = "\n".join(line.split(" ", 1)[1] if " " in line else "" for line in f.read().splitlines())
        full["errors"] = char_errors(ref, hyp)
        if args.lang == "en":
            full["word_errors"] = eval_metrics.word_errors(ref, hyp)
    with open(os.path.join(out, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(full, f, ensure_ascii=False, indent=2)

    cer = f"{full['errors']['cer']:.3f}" if full["errors"] else "-"
    print(f"rtf={full['rtf']:.3f} call_p95={full['call_sec_p95']:.2f}s cer={cer} "
          f"calls={full['calls']} out={out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
