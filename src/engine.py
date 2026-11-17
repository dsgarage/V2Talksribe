#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""リアルタイム文字起こしエンジン。

設計の要:
- 音声状態 (speech_buf / VAD / Whisper / capture) を **単一のワーカースレッドが
  独占所有** する。UI(rumps) とはコマンドキュー(入力)とイベントキュー(出力)でのみ
  通信し、共有可変状態のロックなし競合を構造的に排除する。
- 停止時の確定処理もワーカースレッドで完結させ、メインスレッド(UI)を一切ブロックしない。
- **張り付き脱出弁**: バッファ絶対上限と連続空確定カウンタで、VAD 誤検出による
  バッファ張り付き→幻聴空転の無限ループから必ず抜ける。
"""
from __future__ import annotations

import dataclasses
import os
import queue
import threading
import time
import traceback
from collections import deque
from datetime import datetime
from typing import Optional

import numpy as np

import applog
import level_meter
from audio_preprocess import LiveAudioPreprocessor
from audio_store import prune_recordings, session_part_path
from capture import CaptureHandle, start_capture
from hallucination import collapse_repeats, is_stock_phrase, norm_phrase
from live_level import FRAME_SAMPLES, LevelTracker, normalize_for_asr
from model_loader import LoadJob, hf_cache_bytes
from segmenter import SegInfo, dedupe_prefix, plan_window_commit
from settings import SAVE_DIR, AppSettings, hotwords_from_vocabulary, load_vocabulary
from transcriber import Transcriber
from vad import VoiceActivityDetector

# --- タイミング定数 ---
PARTIAL_MIN_SEC = 1.5
VAD_FRAME_SAMPLES = FRAME_SAMPLES   # 4096
LONG_SPEECH_WINDOW_SEC = 20.0
# 張り付き脱出弁
MAX_SPEECH_BUF_SEC = 28.0        # これを超えたら VAD に関係なく強制確定
WINDOW_RETRY_SEC = 4.0
IDLE_STATUS_SEC = 3.0
# capture 監視
CAPTURE_STALL_SEC = 3.0
CAPTURE_START_TIMEOUT_SEC = 8.0
CAPTURE_BACKOFF_SEC = (1.0, 2.0, 4.0, 8.0, 15.0)
MAX_CAPTURE_FAILURES = 5
CAPTURE_HEALTHY_SEC = 10.0
SILENT_INPUT_PEAK = 1e-4
SILENT_INPUT_NOTIFY_SEC = 30.0
# ワーカーの例外・読み込み・取りこぼし
WORKER_ERROR_WINDOW_SEC = 60.0
MAX_WORKER_ERRORS_IN_WINDOW = 3
LOAD_TIMEOUT_SEC = 180.0
DOWNLOAD_STALL_SEC = 120.0
DOWNLOAD_PROBE_SEC = 5.0
DROP_ALERT_INTERVAL_SEC = 30.0
LOAD_STATUS_INTERVAL_SEC = 1.0
# 録音ファイルの自動整理
RETENTION_DAYS = 14
RETENTION_MAX_BYTES = 5 * 1024**3


def _log(msg: str):
    applog.log(msg)


class TranscriptionEngine:
    """ライブ文字起こしの状態機械。ワーカースレッド 1 本で駆動する。"""

    def __init__(self, settings: AppSettings, *, capture_factory=None, transcriber_factory=None,
                 vad_factory=None, save_dir: Optional[str] = None, record_audio: bool = True,
                 prune_audio: bool = True):
        self.settings = settings
        self.sr = 16000
        self.capture_factory = capture_factory or start_capture
        self.transcriber_factory = transcriber_factory or (
            lambda size, lang, beam: Transcriber(size, language=lang, beam_size=beam))
        self.vad_factory = vad_factory or (
            lambda b, sr, agg: VoiceActivityDetector(b, sr, aggressiveness=agg))
        self.save_dir = save_dir or SAVE_DIR
        self.audio_dir = os.path.join(self.save_dir, "audio")
        self.record_audio = record_audio
        # 古い録音の自動整理はユーザー承認済み(2026-11-17)。テストと再生では False を渡す
        self.prune_audio = prune_audio

        # UI との通信路
        self.events: "queue.Queue" = queue.Queue()
        self._commands: "queue.Queue" = queue.Queue()

        self._worker: Optional[threading.Thread] = None
        self._shutdown = threading.Event()

        # 実行時状態（すべてワーカースレッドのみが触る）
        self.is_recording = False
        self.transcriber: Optional[Transcriber] = None
        self.vad: Optional[VoiceActivityDetector] = None
        self.live_preprocessor: Optional[LiveAudioPreprocessor] = None

        self.speech_buf = np.zeros(0, dtype=np.float32)
        self.silence_count = 0
        self.last_partial_at = 0.0
        self.last_stats_at = 0.0
        self._empty_window_streak = 0
        self.rec_start_time: Optional[float] = None
        self.total_utterances = 0
        self.save_path: Optional[str] = None

        # capture 子プロセス
        self._capture: Optional[CaptureHandle] = None
        self._capture_started = False
        self._capture_opened_at = 0.0
        self._capture_failures = 0
        self._next_reconnect_at = 0.0
        self._use_default_device = False
        self._active_device_name: Optional[str] = None
        self._healthy_since: Optional[float] = None
        self._silent_since: Optional[float] = None
        self._silent_alerted = False
        self._last_drop_total = 0
        self._last_drop_record = 0
        self._last_drop_alert_at = 0.0
        self.last_audio_received_at = 0.0

        # ワーカーの例外・読み込みジョブ・セッション
        self._worker_errors: deque = deque()
        self._load_job: Optional[LoadJob] = None
        self._abandoned_job: Optional[LoadJob] = None
        self._loaded_key: Optional[tuple] = None
        self._load_last_status_at = 0.0
        self._dl_bytes = 0
        self._dl_progress_at = 0.0
        self._dl_probe_at = 0.0
        self._pending_restart = False
        self._session_ts: Optional[str] = None
        self._part = 0
        self.heartbeat_at = time.monotonic()
        self._crashed = False

        self._apply_profile_params()
        self._init_segmentation_state()
        self._level = LevelTracker(self.sr, self.tuning)
        self._reset_live_stats()
        self._committed_tail = ""
        self._last_publish_at = 0.0               # time.monotonic()
        self._last_published_norm = ""
        self._stream_t0: Optional[float] = None   # started を受けた時刻
        self._audio_recv_sec = 0.0                # started 以降に受け取った音声の秒

    def _init_segmentation_state(self):
        """音の切り方の状態を初期値にする(同じセッションの文脈・統計・床と利得は触らない)。"""
        self.speech_buf = np.zeros(0, dtype=np.float32)
        self._frame_rem = np.zeros(0, dtype=np.float32)
        self._preroll: deque = deque()            # (frame, rms_db) の列
        self._preroll_samples = 0
        self._in_speech = False
        self._silence_sec = 0.0
        self._buf_flags: list = []                # speech_buf のフレームごと: 発話と判定したか
        self._buf_rms: list = []                  # 同: RMS dB(加工前の音で測る)
        self._buf_probs: list = []                # 同: VAD の確率(投票率)
        self._next_window_at = LONG_SPEECH_WINDOW_SEC
        self._next_partial_at = 0.0
        self._last_idle_status_at = 0.0
        self._draining = False
        self._continuing_window = False           # 長い発話の窓の続きか(dedupe_prefix を掛ける条件)

    def _reset_live_stats(self):
        self.live_stats = {"utt_total": 0, "utt_skipped": 0, "halluc_dropped": 0, "repeats_collapsed": 0,
                           "dup_dropped": 0, "partials": 0, "partials_skipped_backlog": 0, "windows": 0}

    # ==================================================================
    # プロファイル由来パラメータ
    # ==================================================================

    def _apply_profile_params(self):
        prof = self.settings.profile()
        self.beam_size = prof["beam_size"]
        self.chunk_sec = prof["chunk_sec"]
        self.vad_backend = prof["vad_backend"]
        self.vad_aggressiveness = prof["vad_aggressiveness"]
        self.partial_enabled = prof["partial_enabled"]
        self.partial_interval_sec = prof["partial_interval_sec"]
        self.partial_window_sec = prof["partial_window_sec"]
        self.status_every = prof["status_every"]
        self.tuning = self.settings.live_tuning   # プロファイルを変えても同じ値

    # ==================================================================
    # 公開 API（任意スレッドから呼べる。実処理はワーカーで行う）
    # ==================================================================

    def run(self):
        if self._worker is not None:
            return
        self._worker = threading.Thread(target=self._loop, daemon=True)
        self._worker.start()

    def start(self):
        self._commands.put(("start",))

    def stop(self):
        self._commands.put(("stop",))

    def set_language(self, lang: Optional[str]):
        self._commands.put(("set_language", lang))

    def set_model(self, size: str):
        self._commands.put(("set_model", size))

    def set_profile(self, name: str):
        self._commands.put(("set_profile", name))

    def toggle(self):
        self._commands.put(("toggle",))

    def set_device(self, name: Optional[str]):
        """入力デバイスを名前で指定する(None = システム既定)。"""
        self._commands.put(("set_device", name))

    def set_preprocess(self):
        # settings.preprocess は呼び出し側で更新済み
        self._commands.put(("set_preprocess",))

    def shutdown(self):
        self._shutdown.set()
        self._commands.put(("shutdown",))

    @property
    def is_loading(self) -> bool:
        return self._load_job is not None

    def worker_alive(self) -> bool:
        return self._worker is not None and self._worker.is_alive()

    def ensure_worker(self) -> bool:
        """ワーカーが死んでいて shutdown でなければ起こし直す。起こし直したら True。"""
        if self._shutdown.is_set() or self.worker_alive():
            return False
        _log("worker thread is dead → restart")
        self._crashed = True
        self._worker = None
        self.run()
        return True

    # ==================================================================
    # イベント送出
    # ==================================================================

    def _emit(self, *event):
        self.events.put(event)

    def _emit_runtime_detail(self, message: str):
        speech_sec = len(self.speech_buf) / self.sr if self.sr else 0.0
        self._emit("detail", f"{message} / buf {speech_sec:.1f}s")

    def _report(self, context: str, error: Exception, *, detail: Optional[str] = None):
        _log(f"{context} error: {error}")
        _log(traceback.format_exc())
        if detail:
            self._emit("detail", detail)

    def _alert(self, text: str):
        _log(f"alert: {text}")
        self._emit("alert", text)

    def _elapsed_str(self) -> str:
        if self.rec_start_time is None:
            return "0:00"
        elapsed = int(time.time() - self.rec_start_time)
        h, m, s = elapsed // 3600, (elapsed % 3600) // 60, elapsed % 60
        if h > 0:
            return f"{h}:{m:02d}:{s:02d}"
        return f"{m}:{s:02d}"

    # ==================================================================
    # ワーカーメインループ
    # ==================================================================

    def _loop(self):
        while not self._shutdown.is_set():
            self.heartbeat_at = time.monotonic()
            try:
                if self._crashed:
                    self._crashed = False
                    self._recover_recording("ワーカーを再起動しました")
                self._drain_commands()
                if self._shutdown.is_set():
                    break
                if self.is_recording:
                    self._pump_audio()
                else:
                    # アイドル時はコマンド待ち（軽負荷）。読み込み中は経過を見るので短く待つ
                    if self._load_job is not None:
                        self._poll_load_job()
                    self._wait_command(0.1 if self._load_job is not None else 0.2)
            except Exception as e:
                self._on_worker_error(e)
        self._close_capture()

    def _wait_command(self, timeout: float):
        try:
            cmd = self._commands.get(timeout=timeout)
        except queue.Empty:
            return
        self._handle_command(cmd)

    def _drain_commands(self):
        while True:
            try:
                cmd = self._commands.get_nowait()
            except queue.Empty:
                break
            self._handle_command(cmd)

    def _handle_command(self, cmd):
        action = cmd[0]
        try:
            if action == "start":
                self._handle_start()
            elif action == "stop":
                self._handle_stop()
            elif action == "set_language":
                self._handle_set_language(cmd[1])
            elif action == "set_model":
                self._handle_set_model(cmd[1])
            elif action == "set_profile":
                self._handle_set_profile(cmd[1])
            elif action == "toggle":
                if self.is_recording or self._load_job is not None:
                    self._handle_stop()
                else:
                    self._handle_start()
            elif action == "set_device":
                self._handle_set_device(cmd[1])
            elif action == "set_preprocess":
                self._handle_set_preprocess()
            elif action == "shutdown":
                if self._load_job is not None:
                    self._load_job.cancel()
                if self.is_recording:
                    self._handle_stop()
        except Exception as e:
            self._report(f"command:{action}", e, detail=f"コマンド処理に失敗: {action}")

    # ==================================================================
    # start / stop
    # ==================================================================

    def _handle_start(self):
        if self.is_recording or self._load_job is not None:
            self._emit("toggle", "録音停止" if self.is_recording else "読み込みを取り消す")
            return
        model = self.settings.model_size
        want_key = (model, self.beam_size, self.vad_backend)
        if (self.transcriber is not None and self.vad is not None
                and self.transcriber.model_size == model and self._loaded_key == want_key):
            self.transcriber.language = self.settings.language
            self.transcriber.beam_size = self.beam_size
            try:
                self.vad.reset()
            except Exception as e:
                self._report("vad.reset", e)
            self._begin_stream_safely()
            return
        job = self._abandoned_job
        if job is not None and not job.done.is_set() and job.key == want_key:
            # 取り消した読み込みがまだ走っていて同じ条件なら引き取る
            job.cancelled = False
            self._abandoned_job = None
        else:
            job = LoadJob(
                transcriber_factory=self.transcriber_factory, vad_factory=self.vad_factory,
                model_size=model, language=self.settings.language, beam_size=self.beam_size,
                vad_backend=self.vad_backend, vad_aggressiveness=self.vad_aggressiveness, sr=self.sr,
            )
            job.start()
        self._load_job = job
        self._load_last_status_at = 0.0
        self._dl_bytes = 0
        self._dl_progress_at = time.monotonic()
        self._dl_probe_at = 0.0
        self._emit("toggle", "読み込みを取り消す")
        self._emit("status", f"読み込み中: Whisper {model}")
        self._emit("detail", "録音開始準備: モデル読み込み")

    def _begin_stream_safely(self):
        try:
            self._begin_stream()
        except Exception as e:
            self._report("_handle_start", e, detail="録音開始に失敗")
            self._emit("toggle", "録音開始")
            self._emit("status", f"エラー: {e}")
            self._emit("notify", "Talksribe", "エラー", f"録音開始に失敗: {str(e)[:80]}")

    def _poll_load_job(self):
        job = self._load_job
        model = job.model_size
        now = time.monotonic()
        if job.done.is_set():
            self._load_job = None
            if job.error:
                self._emit("toggle", "録音開始")
                self._emit("status", f"エラー: {job.error}")
                self._emit("notify", "Talksribe", "エラー", f"モデルの読み込みに失敗: {job.error[:80]}")
                self._alert(f"モデルの読み込みに失敗: {job.error}")
                return
            self.transcriber = job.transcriber
            self.vad = job.vad
            # silero が webrtc に切り替わった場合も要求した key を入れる(次回の開始で試し直さない)
            self._loaded_key = job.key
            for w in job.warnings:
                self._alert(w)
                self._emit("notify", "Talksribe", "注意", w[:120])
            self._begin_stream_safely()
            return
        if job.phase == "download":
            if now - self._dl_probe_at >= DOWNLOAD_PROBE_SEC:
                self._dl_probe_at = now
                b = hf_cache_bytes(model)
                if b > self._dl_bytes:
                    self._dl_bytes = b
                    self._dl_progress_at = now
            if now - self._dl_progress_at > DOWNLOAD_STALL_SEC:
                self._timeout_load(
                    f"モデル {model} のダウンロードが {int(DOWNLOAD_STALL_SEC)} 秒進みません。ネットワークを確認してください")
                return
        elif now - job.phase_started_at > LOAD_TIMEOUT_SEC:
            self._timeout_load(f"読み込みが {int(LOAD_TIMEOUT_SEC)} 秒で終わりません ({job.phase})")
            return
        if now - self._load_last_status_at >= LOAD_STATUS_INTERVAL_SEC:
            self._load_last_status_at = now
            sec = int(now - job.started_at)
            if job.phase == "download":
                self._emit("status", f"ダウンロード中: Whisper {model} ({self._dl_bytes // (1024 * 1024)}MB, {sec}秒)")
            else:
                self._emit("status", f"読み込み中: Whisper {model} ({sec}秒)")

    def _timeout_load(self, reason: str):
        job = self._load_job
        job.cancel()
        self._abandoned_job = job
        self._load_job = None
        self._emit("toggle", "録音開始")
        self._emit("status", f"時間切れ: {reason}")
        self._emit("notify", "Talksribe", "時間切れ", reason[:120])
        self._alert(reason)

    def _ensure_live_preprocessor(self):
        pre = self.settings.preprocess
        if self.tuning.agc_enabled or self.tuning.asr_normalize:
            pre = dataclasses.replace(pre, normalize_enabled=False)   # ライブでは既存のノーマライズを使わない
        self.live_preprocessor = LiveAudioPreprocessor(pre, self.sr) if pre.enabled() else None

    def _begin_stream(self):
        self._reset_live_state()
        self._ensure_live_preprocessor()
        if self.transcriber is not None:
            t = self.tuning
            try:
                self.transcriber.hotwords = hotwords_from_vocabulary(
                    load_vocabulary(os.path.join(self.save_dir, "vocabulary.txt")))
            except Exception as e:
                self._report("load_vocabulary", e)
            if hasattr(self.transcriber, "no_speech_threshold"):
                self.transcriber.no_speech_threshold = t.live_no_speech_threshold
                self.transcriber.log_prob_threshold = t.live_log_prob_threshold

        ts = datetime.now().strftime("%Y%m%d%H%M%S")
        self._session_ts = ts
        self._part = 0
        self._use_default_device = False
        self._capture_failures = 0
        self._next_reconnect_at = 0.0
        if self.record_audio and self.prune_audio:
            try:
                moved = prune_recordings(self.audio_dir, max_age_days=RETENTION_DAYS,
                                         max_total_bytes=RETENTION_MAX_BYTES, protect_prefix=ts)
                if moved:
                    _log(f"prune_recordings: {len(moved)} 件をゴミ箱へ")
            except Exception as e:
                self._report("prune_recordings", e)
        os.makedirs(self.save_dir, exist_ok=True)
        self.save_path = os.path.join(self.save_dir, f"{ts}.txt")
        with open(self.save_path, "w", encoding="utf-8") as f:
            f.write(f"# Talksribe 録音 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            if self.record_audio:
                f.write(f"# 音声: audio/{ts}_NN.wav(再接続のたびに NN が増えます)\n")
            f.write("\n")

        self.is_recording = True
        # キャプチャの失敗は _open_capture が握って再接続に回すので、ここでは例外を出さない
        self._open_capture()

        self._emit("title", "REC")
        self._emit("toggle", "録音停止")
        self._emit("status", "録音中 0:00")
        self._emit("detail", "待機: 音声入力を監視中")
        self._emit("notify", "Talksribe", "録音開始", f"保存先: {self.save_path}")
        _log(f"begin_stream: device_name={self.settings.input_device_name!r} "
             f"model={self.settings.model_size} vad={self.vad_backend} "
             f"audio={self.record_audio} prune={self.prune_audio}")

    def _reset_live_state(self):
        self._init_segmentation_state()
        self.silence_count = 0
        self.last_partial_at = 0.0
        self.last_stats_at = 0.0
        self._empty_window_streak = 0
        self._level.reset()
        self._reset_live_stats()
        self.rec_start_time = time.time()
        self.total_utterances = 0
        self._silent_since = None
        self._silent_alerted = False
        self._last_drop_total = 0
        self._last_drop_record = 0
        self._last_drop_alert_at = 0.0
        self._worker_errors.clear()

    def _handle_stop(self):
        if self._load_job is not None:
            job = self._load_job
            job.cancel()
            self._abandoned_job = job
            self._load_job = None
            self._emit("toggle", "録音開始")
            self._emit("status", "読み込みを取り消しました")
            self._emit("detail", "待機")
            return
        if not self.is_recording:
            self._emit("toggle", "録音開始")
            return
        self._emit("detail", "停止中: 残り音声を確定")
        # 残っているキャプチャを吸い出してバッファへ
        self._drain_capture_into_buffer()
        self._close_capture()

        # 表示を先に戻してから残りを確定する(確定は文字起こしで数秒かかる)
        elapsed = self._elapsed_str()
        self.is_recording = False
        self._emit("title", "MIC")
        self._emit("toggle", "録音開始")
        self._emit("status", "停止しました。残りの音声を文字にしています...")

        self._flush_on_stop()

        if self.save_path:
            text = f"保存済み ({elapsed}, {self.total_utterances}件)"
            if self.record_audio and self._session_ts:
                text += f" 音声: audio/{self._session_ts}_*.wav"
            self._emit("status", text)
        else:
            self._emit("status", "停止")
        self._emit("detail", "待機")
        self.rec_start_time = None

    def _restart_stream(self):
        """録音中の設定変更で使う。現在の発話を確定→停止→新ストリーム開始。"""
        was_recording = self.is_recording
        if was_recording:
            self._handle_stop()
            self._handle_start()

    # ==================================================================
    # capture 管理
    # ==================================================================

    def _open_capture(self):
        self._part += 1
        record_path = (session_part_path(self.audio_dir, self._session_ts, self._part)
                       if self.record_audio and self._session_ts else None)
        if record_path:
            os.makedirs(self.audio_dir, exist_ok=True)
        name = None if self._use_default_device else self.settings.input_device_name
        now = time.monotonic()
        self._capture_started = False
        self._capture_opened_at = now
        self.last_audio_received_at = now
        self._healthy_since = None
        self._stream_t0 = None
        try:
            self._capture = self.capture_factory(self.sr, self.chunk_sec, name, record_path)
        except Exception as e:
            self._capture = None
            self._report("_open_capture", e)
            self._on_capture_lost(f"開始に失敗: {e}")

    def _close_capture(self):
        h = self._capture
        self._capture = None
        if h is not None:
            try:
                h.stop()
            except Exception as e:
                self._report("_close_capture", e)

    def _drain_capture_into_buffer(self):
        pipe = self._capture.audio if self._capture is not None else None
        if pipe is None:
            return
        self._draining = True   # 確定の文字起こしは呼ばず、バッファにためるだけ
        try:
            while pipe.poll(0):
                raw = pipe.recv_bytes()
                audio = np.frombuffer(raw, dtype=np.float32)
                self._process_audio(audio)
        except (EOFError, OSError):
            pass
        finally:
            self._draining = False

    def _poll_ctrl(self) -> bool:
        """キャプチャ子プロセスからの状態の知らせを処理する。接続を失ったら False。"""
        h = self._capture
        try:
            while h is not None and h.ctrl.poll(0):
                msg = h.ctrl.recv()
                t = msg.get("type")
                if t == "started":
                    self._capture_started = True
                    self._stream_t0 = time.monotonic()
                    self._audio_recv_sec = 0.0
                    self._active_device_name = msg.get("device_name")
                    self.last_audio_received_at = time.monotonic()
                    _log(f"capture started: {msg}")
                    if msg.get("fell_back") and msg.get("requested"):
                        text = (f"入力デバイス '{msg['requested']}' が見つからないため"
                                f"既定の入力 '{msg.get('device_name')}' を使います")
                        self._alert(text)
                        self._emit("notify", "Talksribe", "入力デバイス", text[:120])
                elif t == "error":
                    self._on_capture_lost(msg.get("message", "不明なエラー"))
                    return False
                elif t == "stats":
                    self._handle_capture_stats(msg)
                # "eof" は何もしない(再生で使う)
        except (EOFError, OSError) as e:
            _log(f"ctrl pipe error: {type(e).__name__}: {e}")
            self._on_capture_lost("入力プロセスとの接続が切れました")
            return False
        return True

    def _handle_capture_stats(self, msg: dict):
        live = int(msg.get("dropped_live_chunks", 0))
        rec = int(msg.get("dropped_record_chunks", 0))
        total = live + rec
        if total > self._last_drop_total:
            _log(f"capture dropped: live={live} record={rec} flags={msg.get('status_flags')}")
            now = time.monotonic()
            if now - self._last_drop_alert_at >= DROP_ALERT_INTERVAL_SEC:
                self._last_drop_alert_at = now
                if rec > self._last_drop_record:
                    text = f"処理の遅れで音声 {rec * self.chunk_sec:.0f}秒ぶんを録音ファイルにも書けていません"
                else:
                    text = (f"処理の遅れで音声 {live * self.chunk_sec:.0f}秒ぶんを文字起こしできませんでした"
                            f"(録音ファイルには残っています)")
                self._alert(text)
                self._emit("notify", "Talksribe", "音声の取りこぼし", text[:120])
            self._last_drop_total = total
            self._last_drop_record = rec

    def _on_capture_lost(self, reason: str):
        self._close_capture()
        self._capture_failures += 1
        n = self._capture_failures
        _log(f"capture lost: {reason} (failures={n}, default={self._use_default_device})")
        if n > MAX_CAPTURE_FAILURES:
            if not self._use_default_device and self.settings.input_device_name:
                name = self.settings.input_device_name
                # 既定に切り替えたあとは、その録音の間は元のデバイスに戻らない
                self._use_default_device = True
                self._capture_failures = 0
                text = f"'{name}' に接続できないため既定の入力に切り替えます"
                self._alert(text)
                self._emit("notify", "Talksribe", "入力デバイス", text[:120])
                self._next_reconnect_at = time.monotonic()
                return
            self._abort_recording(f"入力に接続できません: {reason}")
            return
        delay = CAPTURE_BACKOFF_SEC[min(n - 1, len(CAPTURE_BACKOFF_SEC) - 1)]
        self._next_reconnect_at = time.monotonic() + delay
        self._alert(f"入力が切れました ({reason})。{delay:.0f}秒後に再接続 ({n}/{MAX_CAPTURE_FAILURES})")

    def _mark_healthy(self):
        if not self._capture_started:
            return
        now = time.monotonic()
        if self._healthy_since is None:
            self._healthy_since = now
        elif now - self._healthy_since >= CAPTURE_HEALTHY_SEC and self._capture_failures:
            self._capture_failures = 0

    def _check_input_level(self, raw: np.ndarray):
        peak = float(np.max(np.abs(raw))) if raw.size else 0.0
        now = time.monotonic()
        if peak < SILENT_INPUT_PEAK:
            if self._silent_since is None:
                self._silent_since = now
            elif not self._silent_alerted and now - self._silent_since >= SILENT_INPUT_NOTIFY_SEC:
                self._silent_alerted = True
                dev = self._active_device_name or self.settings.input_device_name or "システム既定"
                text = f"入力が {int(SILENT_INPUT_NOTIFY_SEC)} 秒間まったく無音です: {dev}。入力デバイスを確認してください"
                self._alert(text)
                self._emit("notify", "Talksribe", "無音", text[:120])
        else:
            self._silent_since = None
            self._silent_alerted = False

    # ==================================================================
    # ワーカーの例外と回復
    # ==================================================================

    def _on_worker_error(self, e: Exception):
        _log(f"worker error: {type(e).__name__}: {e}")
        _log(traceback.format_exc())
        now = time.monotonic()
        self._worker_errors.append(now)
        while self._worker_errors and now - self._worker_errors[0] > WORKER_ERROR_WINDOW_SEC:
            self._worker_errors.popleft()
        if not self.is_recording:
            self._alert(f"処理中にエラー: {type(e).__name__}: {e}")
            return
        if len(self._worker_errors) >= MAX_WORKER_ERRORS_IN_WINDOW:
            self._abort_recording(f"エラーが続いたため ({type(e).__name__}: {e})")
            return
        self._emit("notify", "Talksribe", "エラー",
                   f"処理中にエラー: {type(e).__name__}: {str(e)[:60]}。録音を再開します")
        self._recover_recording(f"処理中にエラー: {type(e).__name__}: {e}")

    def _recover_recording(self, reason: str):
        """キャプチャを閉じ、バッファを捨てて開き直す。セッション・txt・経過時間はそのまま。"""
        if not self.is_recording:
            return
        self._close_capture()
        self._init_segmentation_state()
        self.silence_count = 0
        self._empty_window_streak = 0
        self.last_partial_at = 0.0
        if self.vad is not None:
            try:
                self.vad.reset()
            except Exception as e2:
                self._report("vad.reset", e2)
        self._next_reconnect_at = time.monotonic()
        self._alert(f"録音を再開します: {reason}")

    def _abort_recording(self, reason: str):
        self._close_capture()
        if len(self.speech_buf) > self.sr * 0.3 and self.transcriber is not None:
            try:
                self._finalize_all(reason="abort")
            except Exception as e:
                self._report("_abort_recording finalize", e)
        self.is_recording = False
        self._emit("title", "MIC")
        self._emit("toggle", "録音開始")
        self._emit("status", f"停止: {reason}")
        self._emit("detail", "待機")
        where = f"{self.audio_dir}/{self._session_ts}_*.wav" if self.record_audio and self._session_ts else "なし"
        text = f"録音を止めました: {reason}。音声: {where}"
        self._emit("notify", "Talksribe", "録音停止", text[:120])
        self._alert(text)
        self.rec_start_time = None

    # ==================================================================
    # 音声処理
    # ==================================================================

    def _pump_audio(self):
        now = time.monotonic()
        if self._capture is None:
            if now >= self._next_reconnect_at:
                self._open_capture()
            else:
                self._wait_command(0.2)
            return  # ここで無音の処理を呼ばない(空回りの原因)
        if not self._poll_ctrl():
            return
        h = self._capture
        if h is None:
            return
        try:
            if h.audio.poll(min(self.chunk_sec, 0.3)):
                raw = h.audio.recv_bytes()
                audio = np.frombuffer(raw, dtype=np.float32)
                self.last_audio_received_at = time.monotonic()
                self._emit("level", level_meter.window_levels(audio, self.sr))
                self._audio_recv_sec += len(audio) / self.sr
                self._check_input_level(audio)
                self._mark_healthy()
                self._process_audio(audio)
                return
        except (EOFError, OSError) as e:
            _log(f"_pump_audio pipe error: {type(e).__name__}: {e}")
            self._on_capture_lost("入力ストリームが切断")
            return
        now = time.monotonic()
        if not h.is_alive():
            self._on_capture_lost("入力プロセスが終了")
        elif not self._capture_started and now - self._capture_opened_at > CAPTURE_START_TIMEOUT_SEC:
            self._on_capture_lost("入力の開始が時間切れ")
        elif self._capture_started and now - self.last_audio_received_at >= CAPTURE_STALL_SEC:
            self._on_capture_lost(f"入力が {int(CAPTURE_STALL_SEC)} 秒止まりました")

    def _preprocess(self, chunk: np.ndarray) -> np.ndarray:
        if self.live_preprocessor is None:
            return chunk
        return self.live_preprocessor.process(chunk)

    def _backlog_sec(self) -> float:
        if self._stream_t0 is None:
            return 0.0          # 再生の offline など、実時間でないとき
        return max(0.0, (time.monotonic() - self._stream_t0) - self._audio_recv_sec)

    def _process_audio(self, audio: np.ndarray):
        if len(audio) == 0:
            return
        chunk = self._preprocess(audio)
        data = np.concatenate([self._frame_rem, chunk]) if len(self._frame_rem) else chunk
        n = (len(data) // VAD_FRAME_SAMPLES) * VAD_FRAME_SAMPLES
        for i in range(0, n, VAD_FRAME_SAMPLES):
            self._process_frame(np.ascontiguousarray(data[i:i + VAD_FRAME_SAMPLES], dtype=np.float32))
        self._frame_rem = data[n:].copy()

    def _vad_prob(self, x: np.ndarray) -> float:
        if self.vad is None:
            return 0.0
        fn = getattr(self.vad, "speech_prob", None)
        if fn is not None:
            return float(fn(x))
        return 1.0 if self.vad.is_speech(x) else 0.0      # テストの FakeVAD など

    def _append_frame(self, f: np.ndarray, flag: bool, rms_db: float, prob: float):
        self.speech_buf = np.concatenate([self.speech_buf, f])
        self._buf_flags.append(flag)
        self._buf_rms.append(rms_db)
        self._buf_probs.append(prob)

    def _process_frame(self, f: np.ndarray):
        t = self.tuning
        frame_sec = len(f) / self.sr
        now = time.monotonic()
        lvl = self._level.update(f)
        prob = self._vad_prob(self._level.vad_input(f)) if lvl.gated else 0.0
        silero = getattr(self.vad, "backend", "webrtc") == "silero"
        enter, stay = (t.silero_enter, t.silero_stay) if silero else (t.webrtc_enter, t.webrtc_stay)
        speech = prob >= (stay if self._in_speech else enter)

        if speech:
            if not self._in_speech:
                pre = list(self._preroll)
                self.speech_buf = np.concatenate([p[0] for p in pre] + [f]) if pre else f.copy()
                self._buf_flags = [False] * len(pre) + [True]
                self._buf_rms = [p[1] for p in pre] + [lvl.rms_db]
                self._buf_probs = [0.0] * len(pre) + [prob]
                self._preroll.clear()
                self._preroll_samples = 0
                self._in_speech = True
            else:
                self._append_frame(f, True, lvl.rms_db, prob)
            self._silence_sec = 0.0
            if self._draining:
                return
            speech_dur = len(self.speech_buf) / self.sr
            if now - self.last_stats_at >= 1.0:
                self._emit_runtime_detail("録音中: 音声検出")
                self.last_stats_at = now
            if speech_dur >= MAX_SPEECH_BUF_SEC:
                self._finalize_window(force=True)
            elif speech_dur >= LONG_SPEECH_WINDOW_SEC and speech_dur >= self._next_window_at:
                self._finalize_window(force=False)
            else:
                self._maybe_partial(now, speech_dur)
            return

        if self._in_speech:
            self._append_frame(f, False, lvl.rms_db, prob)
            self._silence_sec += frame_sec
            content_sec = len(self.speech_buf) / self.sr - self._silence_sec
            need = t.short_utt_silence_sec if content_sec < t.short_utt_sec else t.silence_end_sec
            if self._silence_sec >= need and not self._draining:
                self._finalize_utterance(reason="silence")
        else:
            self._preroll.append((f, lvl.rms_db))
            self._preroll_samples += len(f)
            limit = int(t.preroll_sec * self.sr)
            while self._preroll and self._preroll_samples - len(self._preroll[0][0]) >= limit:
                self._preroll_samples -= len(self._preroll.popleft()[0])
            if now - self._last_idle_status_at >= IDLE_STATUS_SEC:
                self._last_idle_status_at = now
                self._emit("status", f"録音中 {self._elapsed_str()} ({self.total_utterances}件)")
                self._emit_runtime_detail("待機: 音声待ち")

    # ------------------------------------------------------------------
    # 区間の受け入れ条件
    # ------------------------------------------------------------------

    def _utterance_stats(self) -> dict:
        """speech_buf 全体(プリロールと末尾の無音を含む)について計算する。"""
        flags = self._buf_flags
        idx = [i for i, v in enumerate(flags) if v]
        fs = VAD_FRAME_SAMPLES / self.sr
        if not idx:
            return {"voiced_sec": 0.0, "ratio": 0.0, "snr_db": -99.0, "peak_prob": 0.0, "voiced_rms": []}
        span = idx[-1] - idx[0] + 1                 # 最初の発話フレームから最後の発話フレームまで
        voiced_rms = [self._buf_rms[i] for i in idx]
        return {"voiced_sec": len(idx) * fs, "ratio": len(idx) / span,
                "snr_db": float(np.median(voiced_rms)) - self._level.floor_db,
                "peak_prob": max(self._buf_probs[i] for i in idx), "voiced_rms": voiced_rms}

    def _accept_utterance(self, st: dict) -> tuple:
        t = self.tuning
        if st["voiced_sec"] < t.min_voiced_sec:
            return False, f"voiced {st['voiced_sec']:.2f}s"
        if st["ratio"] < t.min_voiced_ratio:
            return False, f"ratio {st['ratio']:.2f}"
        if st["snr_db"] < t.min_snr_db:
            return False, f"snr {st['snr_db']:.1f}dB"
        silero = getattr(self.vad, "backend", "webrtc") == "silero"
        if silero and st["peak_prob"] < t.utt_min_peak_prob:
            return False, f"prob {st['peak_prob']:.2f}"
        return True, ""

    # ------------------------------------------------------------------
    # Whisper の呼び出し
    # ------------------------------------------------------------------

    def _asr_opts(self, audio_sec: float, *, partial: bool = False) -> dict:
        t = self.tuning
        return {"temperature": t.temperature,
                "max_new_tokens": int(min(t.tokens_cap, t.tokens_base + t.tokens_per_sec * audio_sec)),
                "beam_size": t.partial_beam if partial else None,
                "without_timestamps": partial}

    def _transcribe_segments(self, audio: np.ndarray, voiced_rms: list, *, initial_prompt=None,
                             partial=False) -> list:
        """SegInfo の列を返す。transcriber に collect_segments が無ければ collect_texts で代用する。"""
        x = normalize_for_asr(audio, voiced_rms, self.tuning)
        opts = self._asr_opts(len(x) / self.sr, partial=partial)
        fn = getattr(self.transcriber, "collect_segments", None)
        if fn is not None:
            return fn(x, initial_prompt=initial_prompt, **opts)
        texts = self.transcriber.collect_texts(x)          # テストの FakeTranscriber など。オプションは渡さない
        dur = len(x) / self.sr
        return [SegInfo(0.0, dur, s, 0.0, 0.0) for s in texts if s]

    # ------------------------------------------------------------------
    # 幻聴の除去・文脈
    # ------------------------------------------------------------------

    def _filter_segments(self, segs: list, st: dict) -> list:
        t = self.tuning
        out = []
        lang = getattr(self.transcriber, "language", None)
        for s in segs:
            text = s.text.strip()
            if not text:
                continue
            if not t.halluc_filter:
                out.append(text)
                continue
            weak = (st["voiced_sec"] < t.weak_voiced_sec or s.no_speech_prob > t.weak_no_speech_prob
                    or s.avg_logprob < t.weak_avg_logprob)
            # 弱くない区間は 5 回以上続いたときだけまとめる('very very very good' などを壊さない)
            text2, collapsed = collapse_repeats(text, t.repeat_min if weak else max(t.repeat_min, 5))
            if collapsed:
                self.live_stats["repeats_collapsed"] += 1
                _log(f"halluc collapse: {text!r} -> {text2!r}")
            text = text2
            if weak and is_stock_phrase(text, lang):
                self.live_stats["halluc_dropped"] += 1
                _log(f"halluc drop: stock weak={weak} {text!r}")
                continue
            n = norm_phrase(text)
            if (weak and n and n == self._last_published_norm
                    and time.monotonic() - self._last_publish_at <= t.dup_window_sec):
                self.live_stats["dup_dropped"] += 1
                _log(f"halluc drop: dup {text!r}")
                continue
            out.append(text)
        return out

    def _context_prompt(self) -> Optional[str]:
        t = self.tuning
        if t.context_chars <= 0 or not self._committed_tail:
            return None
        if time.monotonic() - self._last_publish_at > t.context_max_gap_sec:
            return None
        return self._committed_tail[-t.context_chars:]

    def _publish_texts(self, texts):
        if not texts:
            return
        ts = time.strftime("%H:%M:%S")
        full = " ".join(texts)
        self.total_utterances += 1
        self._emit("transcript", ts, full)
        self._emit("status", f"録音中 {self._elapsed_str()} ({self.total_utterances}件)")
        self._emit_runtime_detail("録音中: 確定結果を保存")
        if self.save_path:
            try:
                with open(self.save_path, "a", encoding="utf-8") as f:
                    f.write(f"[{ts}] {full}\n")
            except OSError as e:
                self._report("_publish_texts save", e, detail="保存に失敗")
        self._last_publish_at = time.monotonic()
        self._last_published_norm = norm_phrase(full)
        self._committed_tail = (self._committed_tail + full)[-200:]

    # ------------------------------------------------------------------
    # 部分更新(テンポ)
    # ------------------------------------------------------------------

    def _maybe_partial(self, now: float, speech_dur: float):
        t = self.tuning
        if not (self.partial_enabled and self._commands.empty()):     # #9 の条件はそのまま
            return
        if speech_dur < t.partial_min_speech_sec or now < self._next_partial_at:
            return
        if now - self.last_partial_at < self.partial_interval_sec:
            return
        if self._backlog_sec() > t.partial_max_backlog_sec:
            self.live_stats["partials_skipped_backlog"] += 1
            return
        self._do_partial()

    def _do_partial(self):
        t0 = time.monotonic()
        try:
            self._emit_runtime_detail("文字起こし中: 部分更新")
            k = max(1, int(self.partial_window_sec * self.sr / VAD_FRAME_SAMPLES))
            audio = self.speech_buf[-k * VAD_FRAME_SAMPLES:]
            rms = [r for r, fl in zip(self._buf_rms[-k:], self._buf_flags[-k:]) if fl]
            segs = self._transcribe_segments(audio, rms, partial=True)
            text = " ".join(s.text.strip() for s in segs if s.text.strip())
            if text:
                text, _ = collapse_repeats(text, self.tuning.repeat_min)
                self._emit("partial", text[-80:])
            self.live_stats["partials"] += 1
        except Exception as e:
            self._report("_do_partial", e, detail="部分文字起こしに失敗")
        finally:
            done = time.monotonic()
            self.last_partial_at = done
            self._next_partial_at = done + max(self.partial_interval_sec,
                                               self.tuning.partial_cost_factor * (done - t0))

    # ------------------------------------------------------------------
    # 確定
    # ------------------------------------------------------------------

    def _clear_buffer(self):
        self.speech_buf = np.zeros(0, dtype=np.float32)
        self._buf_flags = []
        self._buf_rms = []
        self._buf_probs = []
        self._in_speech = False
        self._silence_sec = 0.0
        self._next_window_at = LONG_SPEECH_WINDOW_SEC
        self.last_partial_at = 0.0
        self._next_partial_at = 0.0

    def _cut_buffer(self, cut_sec: float) -> int:
        """先頭から cut_sec ぶんを捨てる(フレームの境目にそろえる)。捨てたフレーム数を返す。"""
        k = int(round(cut_sec * self.sr / VAD_FRAME_SAMPLES))
        k = max(0, min(k, len(self.speech_buf) // VAD_FRAME_SAMPLES))
        self.speech_buf = self.speech_buf[k * VAD_FRAME_SAMPLES:]
        self._buf_flags = self._buf_flags[k:]
        self._buf_rms = self._buf_rms[k:]
        self._buf_probs = self._buf_probs[k:]
        return k

    def _finalize_utterance(self, reason: str):
        self.live_stats["utt_total"] += 1
        st = self._utterance_stats()
        ok, why = self._accept_utterance(st)
        if ok and self.transcriber is not None:
            try:
                self._emit_runtime_detail("文字起こし中: 発話を確定")
                segs = self._transcribe_segments(self.speech_buf, st["voiced_rms"],
                                                 initial_prompt=self._context_prompt())
                texts = self._filter_segments(segs, st)
                if texts and self._continuing_window:
                    first = dedupe_prefix(self._committed_tail, texts[0])
                    texts = ([first] if first else []) + texts[1:]
                self._publish_texts(texts)
            except Exception as e:
                self._report("_finalize_utterance", e, detail="発話の確定に失敗")
        elif not ok:
            self.live_stats["utt_skipped"] += 1
            _log(f"utt skip ({reason}): {why} len={len(self.speech_buf)/self.sr:.2f}s")
        self._clear_buffer()
        self._continuing_window = False
        self._emit("partial", "")       # 途中経過の表示を消す

    def _finalize_all(self, reason: str = ""):
        self._finalize_utterance(reason)

    def _finalize_window(self, force: bool):
        self.live_stats["windows"] += 1
        buf_sec = len(self.speech_buf) / self.sr
        st = self._utterance_stats()
        plan = None
        try:
            self._emit_runtime_detail("文字起こし中: 長尺発話を分割")
            segs = self._transcribe_segments(self.speech_buf, st["voiced_rms"],
                                             initial_prompt=self._context_prompt())
            plan = plan_window_commit(segs, buf_sec, force=force)
            texts = self._filter_segments(plan.segs, st)
            if texts:
                if self._continuing_window:
                    first = dedupe_prefix(self._committed_tail, texts[0])
                    texts = ([first] if first else []) + texts[1:]
                self._publish_texts(texts)
                self._continuing_window = True
            if not segs:
                _log(f"window: 文字が出ず {plan.cut_sec:.1f} 秒を捨てました(録音ファイルには残っています)")
        except Exception as e:
            self._report("_finalize_window", e, detail="長尺発話の確定に失敗")
        cut_sec = plan.cut_sec if plan is not None else (buf_sec if force else max(0.0, buf_sec - 2.0))
        k = self._cut_buffer(cut_sec)
        remain = len(self.speech_buf) / self.sr
        # 1 フレームも切れなかったときは、毎フレーム空回りしないよう待つ
        self._next_window_at = (remain + WINDOW_RETRY_SEC) if k == 0 else LONG_SPEECH_WINDOW_SEC
        self.last_partial_at = 0.0
        if len(self.speech_buf) == 0:
            self._in_speech = False
            self._silence_sec = 0.0

    def _flush_on_stop(self):
        if self.transcriber is None or len(self.speech_buf) == 0:
            self._clear_buffer()
            return
        while len(self.speech_buf) / self.sr > MAX_SPEECH_BUF_SEC:
            self._finalize_window(force=True)
        if len(self.speech_buf) > self.sr * 0.3:
            self._finalize_utterance(reason="stop")
        else:
            self._clear_buffer()

    # ==================================================================
    # 設定変更ハンドラ（ワーカースレッド内で実行）
    # ==================================================================

    def _handle_set_language(self, lang: Optional[str]):
        self.settings.language = lang
        self.settings.save()
        if self.transcriber is not None:
            self.transcriber.language = lang
        self._emit("status", f"言語 → {lang or '自動検出'}")

    def _cancel_load_for_restart(self):
        """読み込み中の設定変更: 今の読み込みを取り消し、新しい条件で読み込み直す。"""
        if self._load_job is not None:
            job = self._load_job
            job.cancel()
            self._abandoned_job = job
            self._load_job = None
            self._pending_restart = True

    def _handle_set_model(self, size: str):
        if size == self.settings.model_size:
            return
        self._cancel_load_for_restart()
        self.settings.model_size = size
        self.settings.save()
        self.transcriber = None  # 次回ロードで作り直し
        self._loaded_key = None
        self._emit("status", f"モデル → {size}")
        if self.is_recording:
            self._restart_stream()
        elif self._pending_restart:
            self._pending_restart = False
            self._handle_start()

    def _handle_set_profile(self, name: str):
        if name == self.settings.power_profile:
            return
        self._cancel_load_for_restart()
        self.settings.power_profile = name
        # プロファイル既定モデルを実効モデルへ反映
        self.settings.model_size = self.settings.profile()["model_size"]
        self.settings.save()
        self._apply_profile_params()
        self.transcriber = None
        self.vad = None
        self._loaded_key = None
        self._emit("status", f"電力 → {self.settings.profile()['label']}")
        if self.is_recording:
            self._restart_stream()
        elif self._pending_restart:
            self._pending_restart = False
            self._handle_start()

    def _handle_set_device(self, name: Optional[str]):
        self.settings.input_device_name = name
        self.settings.input_device_id = None
        self.settings.save()
        self._emit("status", f"入力デバイス → {name or 'システム既定'}")
        if self.is_recording:
            self._restart_stream()

    def _handle_set_preprocess(self):
        self.settings.save()
        self._ensure_live_preprocessor()
        self._emit("status", f"音声前処理 → {self.settings.preprocess.label()}")
