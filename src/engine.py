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

import os
import queue
import threading
import time
import traceback
from datetime import datetime
from typing import Optional

import numpy as np

from audio_preprocess import LiveAudioPreprocessor
from capture import start_capture, stop_capture
from settings import SAVE_DIR, AppSettings
from transcriber import Transcriber
from vad import VoiceActivityDetector

LOG_PATH = os.path.join(SAVE_DIR, "debug.log")

# --- タイミング定数 ---
PARTIAL_MIN_SEC = 1.5
LONG_SPEECH_WINDOW_SEC = 12.0
LONG_SPEECH_STEP_SEC = 10.0
# 張り付き脱出弁
MAX_SPEECH_BUF_SEC = 18.0        # これを超えたら VAD に関係なく強制確定
MAX_EMPTY_WINDOW_STREAK = 2      # 連続でこの回数だけ空確定したらバッファ全破棄
# capture 監視
CAPTURE_STALL_SEC = 3.0
CAPTURE_RESTART_COOLDOWN_SEC = 8.0


def _log(msg: str):
    try:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(f"[{time.strftime('%H:%M:%S')}] {msg}\n")
    except OSError:
        pass


class TranscriptionEngine:
    """ライブ文字起こしの状態機械。ワーカースレッド 1 本で駆動する。"""

    def __init__(self, settings: AppSettings):
        self.settings = settings
        self.sr = 16000

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
        self._capture_pipe = None
        self._capture_stop = None
        self._capture_proc = None
        self.last_audio_received_at = 0.0
        self.last_capture_restart_at = 0.0
        self.capture_restart_count = 0

        self._apply_profile_params()

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
        self.silence_limit = max(1, int(0.8 / self.chunk_sec))

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

    def set_device(self, device_id: Optional[int]):
        self._commands.put(("set_device", device_id))

    def set_preprocess(self):
        # settings.preprocess は呼び出し側で更新済み
        self._commands.put(("set_preprocess",))

    def shutdown(self):
        self._shutdown.set()
        self._commands.put(("shutdown",))

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
            self._drain_commands()
            if self._shutdown.is_set():
                break
            if self.is_recording:
                self._pump_audio()
            else:
                # アイドル時はコマンド待ち（軽負荷）
                try:
                    cmd = self._commands.get(timeout=0.2)
                    self._handle_command(cmd)
                except queue.Empty:
                    pass
        self._stop_capture()

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
            elif action == "set_device":
                self._handle_set_device(cmd[1])
            elif action == "set_preprocess":
                self._handle_set_preprocess()
            elif action == "shutdown":
                if self.is_recording:
                    self._handle_stop()
        except Exception as e:
            self._report(f"command:{action}", e, detail=f"コマンド処理に失敗: {action}")

    # ==================================================================
    # start / stop
    # ==================================================================

    def _handle_start(self):
        if self.is_recording:
            return
        self._emit("toggle", "読み込み中...")
        self._emit("status", "モデル読み込み中...")
        self._emit("detail", "録音開始準備")
        try:
            self._ensure_loaded()
            self._begin_stream()
        except Exception as e:
            self._report("_handle_start", e, detail="録音開始に失敗")
            self._emit("toggle", "録音開始")
            self._emit("status", f"エラー: {e}")
            self._emit("notify", "Talksribe", "エラー", f"録音開始に失敗: {str(e)[:80]}")

    def _ensure_loaded(self):
        # Whisper
        if self.transcriber is None or self.transcriber.model_size != self.settings.model_size:
            self._emit("status", f"Whisper {self.settings.model_size} 読み込み中...")
            self._emit("detail", f"モデル準備: Whisper {self.settings.model_size}")
            self.transcriber = Transcriber(
                self.settings.model_size,
                language=self.settings.language,
                beam_size=self.beam_size,
            )
            self.transcriber.load()
        else:
            self.transcriber.language = self.settings.language
            self.transcriber.beam_size = self.beam_size

        # VAD
        if self.vad is None or self.vad.backend != self.vad_backend:
            self._emit("detail", f"モデル準備: VAD({self.vad_backend})")
            self.vad = VoiceActivityDetector(
                self.vad_backend,
                self.sr,
                aggressiveness=self.vad_aggressiveness,
            )
            self.vad.load()
        else:
            self.vad.reset()

    def _ensure_live_preprocessor(self):
        if self.settings.preprocess.enabled():
            self.live_preprocessor = LiveAudioPreprocessor(self.settings.preprocess, self.sr)
        else:
            self.live_preprocessor = None

    def _begin_stream(self):
        self._reset_live_state()
        self._ensure_live_preprocessor()

        ts = datetime.now().strftime("%Y%m%d%H%M%S")
        self.save_path = os.path.join(SAVE_DIR, f"{ts}.txt")
        with open(self.save_path, "w", encoding="utf-8") as f:
            f.write(f"# Talksribe 録音 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        self._capture_pipe, self._capture_stop, self._capture_proc = start_capture(
            sr=self.sr, chunk_sec=self.chunk_sec, device_id=self.settings.input_device_id,
        )
        self.last_audio_received_at = time.monotonic()
        self.is_recording = True

        self._emit("title", "REC")
        self._emit("toggle", "録音停止")
        self._emit("status", "録音中 0:00")
        self._emit("detail", "待機: 音声入力を監視中")
        self._emit("notify", "Talksribe", "録音開始", f"保存先: {self.save_path}")
        _log(f"begin_stream: device={self.settings.input_device_id} model={self.settings.model_size} vad={self.vad_backend}")

    def _reset_live_state(self):
        self.speech_buf = np.zeros(0, dtype=np.float32)
        self.silence_count = 0
        self.last_partial_at = 0.0
        self.last_stats_at = 0.0
        self._empty_window_streak = 0
        self.rec_start_time = time.time()
        self.total_utterances = 0
        self.last_capture_restart_at = 0.0
        self.capture_restart_count = 0

    def _handle_stop(self):
        if not self.is_recording:
            return
        self._emit("detail", "停止中: 残り音声を確定")
        # 残っているキャプチャを吸い出してバッファへ
        self._drain_capture_into_buffer()
        self._stop_capture()

        if len(self.speech_buf) > self.sr * 0.3 and self.transcriber is not None:
            self._finalize_all(reason="stop")

        elapsed = self._elapsed_str()
        self.is_recording = False
        self._emit("title", "MIC")
        self._emit("toggle", "録音開始")
        if self.save_path:
            self._emit("status", f"保存済み ({elapsed}, {self.total_utterances}件)")
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

    def _stop_capture(self):
        if self._capture_stop is not None and self._capture_proc is not None:
            try:
                stop_capture(self._capture_stop, self._capture_proc)
            except Exception as e:
                self._report("_stop_capture", e, detail="入力プロセスの停止に失敗")
        self._capture_pipe = None
        self._capture_stop = None
        self._capture_proc = None

    def _drain_capture_into_buffer(self):
        pipe = self._capture_pipe
        if pipe is None:
            return
        try:
            while pipe.poll(0):
                raw = pipe.recv_bytes()
                audio = np.frombuffer(raw, dtype=np.float32)
                self.speech_buf = np.concatenate([self.speech_buf, self._preprocess(audio)])
        except (EOFError, OSError):
            pass

    def _restart_capture(self):
        now = time.monotonic()
        if now - self.last_capture_restart_at < CAPTURE_RESTART_COOLDOWN_SEC:
            return
        self.last_capture_restart_at = now
        self.capture_restart_count += 1
        _log(f"_restart_capture: #{self.capture_restart_count}")
        self._emit("detail", f"入力停止を検知: 再接続中 ({self.capture_restart_count})")

        self._stop_capture()
        self._capture_pipe, self._capture_stop, self._capture_proc = start_capture(
            sr=self.sr, chunk_sec=self.chunk_sec, device_id=self.settings.input_device_id,
        )
        self.last_audio_received_at = time.monotonic()

    # ==================================================================
    # 音声処理
    # ==================================================================

    def _pump_audio(self):
        pipe = self._capture_pipe
        if pipe is None:
            self._register_silence()
            return
        try:
            if pipe.poll(min(self.chunk_sec, 0.3)):
                raw = pipe.recv_bytes()
                audio = np.frombuffer(raw, dtype=np.float32)
                self.last_audio_received_at = time.monotonic()
                self._process_audio(audio)
                return
        except (EOFError, OSError) as e:
            _log(f"_pump_audio pipe error: {type(e).__name__}: {e}")
            self._emit("detail", "入力ストリームが切断")
            self._restart_capture()
            return
        # このポーリング周期で音声が来なかった → stall 監視
        if (
            self.last_audio_received_at
            and time.monotonic() - self.last_audio_received_at >= CAPTURE_STALL_SEC
        ):
            self._restart_capture()

    def _preprocess(self, chunk: np.ndarray) -> np.ndarray:
        if self.live_preprocessor is None:
            return chunk
        return self.live_preprocessor.process(chunk)

    def _process_audio(self, audio: np.ndarray):
        if len(audio) < 512:
            return
        chunk = self._preprocess(audio)

        if self.vad is not None and self.vad.is_speech(chunk):
            self.silence_count = 0
            self.speech_buf = np.concatenate([self.speech_buf, chunk])
            speech_dur = len(self.speech_buf) / self.sr
            now = time.monotonic()

            if now - self.last_stats_at >= 1.0:
                self._emit_runtime_detail("録音中: 音声検出")
                self.last_stats_at = now

            if speech_dur >= MAX_SPEECH_BUF_SEC:
                # 絶対上限: 何があっても確定して進める
                self._finalize_all(reason="max-buffer")
            elif speech_dur >= LONG_SPEECH_WINDOW_SEC:
                self._finalize_window()
            elif (
                self.partial_enabled
                and speech_dur >= PARTIAL_MIN_SEC
                and now - self.last_partial_at >= self.partial_interval_sec
            ):
                self._do_partial()
        else:
            self._register_silence()

    def _register_silence(self):
        self.silence_count += 1
        if self.silence_count >= self.silence_limit and len(self.speech_buf) > self.sr * 0.3:
            self._finalize_all(reason="silence")
        elif self.silence_count >= self.silence_limit:
            self.speech_buf = np.zeros(0, dtype=np.float32)
            self._empty_window_streak = 0
        if self.silence_count % self.status_every == 0:
            self._emit("status", f"録音中 {self._elapsed_str()} ({self.total_utterances}件)")
            self._emit_runtime_detail("待機: 音声待ち")

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

    def _do_partial(self):
        try:
            self._emit_runtime_detail("文字起こし中: 部分更新")
            max_samples = int(self.sr * self.partial_window_sec)
            audio = self.speech_buf[-max_samples:] if len(self.speech_buf) > max_samples else self.speech_buf
            texts = self.transcriber.collect_texts(audio)
            if texts:
                self._emit("status", f"{self._elapsed_str()} ...{' '.join(texts)[:40]}")
            self.last_partial_at = time.monotonic()
        except Exception as e:
            self._report("_do_partial", e, detail="部分文字起こしに失敗")

    def _finalize_window(self):
        window_samples = int(self.sr * LONG_SPEECH_WINDOW_SEC)
        step_samples = int(self.sr * LONG_SPEECH_STEP_SEC)
        window_audio = self.speech_buf[:window_samples].copy()
        cleared = False
        try:
            self._emit_runtime_detail("文字起こし中: 長尺発話を分割")
            texts = self.transcriber.collect_texts(window_audio, emit_until_sec=LONG_SPEECH_STEP_SEC)
            if texts:
                self._empty_window_streak = 0
                self._publish_texts(texts)
            else:
                self._empty_window_streak += 1
        except Exception as e:
            self._report("_finalize_window", e, detail="長尺発話の確定に失敗")
        finally:
            if self._empty_window_streak >= MAX_EMPTY_WINDOW_STREAK:
                # 張り付き脱出弁: 中身が無音/幻聴なのでバッファを全破棄する
                _log(
                    f"escape valve: empty_window_streak={self._empty_window_streak} "
                    f"→ speech_buf({len(self.speech_buf)}) を破棄"
                )
                self.speech_buf = np.zeros(0, dtype=np.float32)
                self._empty_window_streak = 0
                self.silence_count = 0
                cleared = True
            if not cleared:
                self.speech_buf = self.speech_buf[step_samples:]
            self.last_partial_at = 0.0

    def _finalize_all(self, reason: str = ""):
        try:
            self._emit_runtime_detail("文字起こし中: 発話を確定")
            texts = self.transcriber.collect_texts(self.speech_buf)
            self._publish_texts(texts)
        except Exception as e:
            self._report("_finalize_all", e, detail="発話の確定に失敗")
        self.speech_buf = np.zeros(0, dtype=np.float32)
        self.last_partial_at = 0.0
        self._empty_window_streak = 0

    # ==================================================================
    # 設定変更ハンドラ（ワーカースレッド内で実行）
    # ==================================================================

    def _handle_set_language(self, lang: Optional[str]):
        self.settings.language = lang
        self.settings.save()
        if self.transcriber is not None:
            self.transcriber.language = lang
        self._emit("status", f"言語 → {lang or '自動検出'}")

    def _handle_set_model(self, size: str):
        if size == self.settings.model_size:
            return
        self.settings.model_size = size
        self.settings.save()
        self.transcriber = None  # 次回ロードで作り直し
        self._emit("status", f"モデル → {size}")
        if self.is_recording:
            self._restart_stream()

    def _handle_set_profile(self, name: str):
        if name == self.settings.power_profile:
            return
        self.settings.power_profile = name
        # プロファイル既定モデルを実効モデルへ反映
        self.settings.model_size = self.settings.profile()["model_size"]
        self.settings.save()
        self._apply_profile_params()
        self.transcriber = None
        self.vad = None
        self._emit("status", f"電力 → {self.settings.profile()['label']}")
        if self.is_recording:
            self._restart_stream()

    def _handle_set_device(self, device_id: Optional[int]):
        self.settings.input_device_id = device_id
        self.settings.save()
        self._emit("status", f"入力デバイス → {device_id}")
        if self.is_recording:
            self._restart_stream()

    def _handle_set_preprocess(self):
        self.settings.save()
        self._ensure_live_preprocessor()
        self._emit("status", f"音声前処理 → {self.settings.preprocess.label()}")
