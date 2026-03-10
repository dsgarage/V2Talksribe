#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""macOS メニューバー リアルタイム文字起こしアプリ

rumps + sounddevice + Silero VAD + Faster-Whisper

長時間カンファレンス対応:
- デフォルト base モデル (RTF=0.15, バッテリー省電力)
- 無音時は sleep 間隔を広げて CPU 節約
- 長い発言は約10秒ごとに順次確定
- 経過時間・認識件数をステータス表示
- AudioQueue で取りこぼしなし
"""
from __future__ import annotations

import atexit
import io
import os
import queue
import signal
import subprocess
import threading
import time
from collections import deque
from datetime import datetime
from typing import Optional

import numpy as np
import rumps

from audio_worker import start_capture, stop_capture

# --- 定数 ---

MAX_LINES = 30
MAX_MENU_CHARS = 55
SAVE_DIR = os.path.expanduser("~/Documents/Talksribe")
os.makedirs(SAVE_DIR, exist_ok=True)
LOG_PATH = os.path.join(SAVE_DIR, "debug.log")

def _log(msg: str):
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(f"[{time.strftime('%H:%M:%S')}] {msg}\n")

# 省電力: 連続無音がこの回数を超えたら sleep 間隔を広げる
IDLE_THRESHOLD = 6       # 3秒間無音 (0.5s * 6)
IDLE_SLEEP_SEC = 1.5     # 省電力モード時の sleep 間隔
PARTIAL_MIN_SEC = 1.5
PARTIAL_STATUS_INTERVAL_SEC = 2.5
PARTIAL_WINDOW_SEC = 6.0
LONG_SPEECH_WINDOW_SEC = 12.0
LONG_SPEECH_STEP_SEC = 10.0


def copy_to_clipboard(text: str):
    p = subprocess.Popen(["pbcopy"], stdin=subprocess.PIPE)
    p.communicate(text.encode("utf-8"))


class AudioQueue:
    """マイク入力チャンクを蓄積するキュー。取りこぼしなし。"""

    def __init__(self):
        self._q: queue.Queue = queue.Queue()

    def put(self, chunk: np.ndarray):
        self._q.put(chunk.astype(np.float32))

    def drain(self) -> Optional[np.ndarray]:
        chunks = []
        while True:
            try:
                chunks.append(self._q.get_nowait())
            except queue.Empty:
                break
        if not chunks:
            return None
        return np.concatenate(chunks)

    def clear(self):
        while not self._q.empty():
            try:
                self._q.get_nowait()
            except queue.Empty:
                break


class TalksribeMenuBar(rumps.App):
    def __init__(self):
        super().__init__("MIC", quit_button=None)

        # 設定
        self.sr = 16000
        self.model_size = "small"
        self.language = "en"
        self.compute_device = "cpu"
        self.compute_type = "int8"
        self.beam_size = 3
        self.chunk_sec = 0.5
        self.vad_threshold = 0.3
        self.input_device_id: Optional[int] = None

        # 状態
        self.is_recording = False
        self.is_model_loaded = False
        self.whisper_model = None
        self.vad_model = None
        self.audio_q = AudioQueue()
        self.transcript_lines: deque = deque(maxlen=MAX_LINES)
        self.speech_buf = np.zeros(0, dtype=np.float32)
        self.silence_count = 0
        self.last_partial_at = 0.0
        self.stop_event = threading.Event()
        self.save_path: Optional[str] = None
        self.rec_start_time: Optional[float] = None
        self.total_utterances = 0

        # 子プロセス（マイク取得用）
        self._capture_pipe = None
        self._capture_stop = None
        self._capture_proc = None

        # 異常終了時にも子プロセスを確実に終了
        atexit.register(self._cleanup_capture)

        # バックグラウンド → メインスレッド通信
        self._pending: queue.Queue = queue.Queue()

        # --- メニュー ---
        self.toggle_item = rumps.MenuItem("録音開始", callback=self.on_toggle)
        self.file_item = rumps.MenuItem("ファイルを文字起こし...", callback=self.on_file_transcribe)
        self.status_item = rumps.MenuItem("状態: 待機中")
        self.status_item.set_callback(None)
        self.is_file_processing = False
        self.file_cancel_event = threading.Event()

        self.copy_item = rumps.MenuItem("全文コピー", callback=self.on_copy_all)
        self.copy_last_item = rumps.MenuItem("最新行コピー", callback=self.on_copy_last)
        self.clear_item = rumps.MenuItem("履歴クリア", callback=self.on_clear)
        self.open_folder_item = rumps.MenuItem("保存フォルダを開く", callback=self.on_open_folder)

        self.lang_menu = rumps.MenuItem("言語")
        self._lang_items = {}
        for code, label in [("ja", "日本語"), ("en", "English"), ("zh", "中文"), (None, "自動検出")]:
            checked = "● " if code == self.language else "  "
            item = rumps.MenuItem(f"{checked}{label}", callback=self.on_change_language)
            item._lang_code = code
            item._lang_label = label
            self._lang_items[str(code)] = item
            self.lang_menu.add(item)

        self.model_menu = rumps.MenuItem("モデル")
        self._model_items = {}
        for size in ["tiny", "base", "small", "medium", "large-v3"]:
            checked = "● " if size == self.model_size else "  "
            item = rumps.MenuItem(f"{checked}{size}", callback=self.on_change_model)
            item._model_size = size
            self._model_items[size] = item
            self.model_menu.add(item)

        self.device_menu = rumps.MenuItem("入力デバイス")
        self._device_items = {}
        self._build_device_menu()

        self.transcript_header = rumps.MenuItem("--- 文字起こし ---")
        self.transcript_header.set_callback(None)
        self.quit_item = rumps.MenuItem("終了", callback=self.on_quit)

        self.menu = [
            self.toggle_item,
            self.file_item,
            self.status_item,
            None,
            self.copy_item,
            self.copy_last_item,
            self.clear_item,
            self.open_folder_item,
            None,
            self.lang_menu,
            self.model_menu,
            self.device_menu,
            None,
            self.transcript_header,
            self.quit_item,
        ]

    def _build_device_menu(self):
        import sounddevice as sd
        devices = sd.query_devices()
        default_id = sd.default.device[0]
        for i, d in enumerate(devices):
            if d["max_input_channels"] > 0:
                checked = "● " if i == default_id else "  "
                name = d["name"]
                item = rumps.MenuItem(f"{checked}{name}", callback=self.on_change_device)
                item._dev_id = i
                item._dev_name = name
                self._device_items[i] = item
                self.device_menu.add(item)

    # =========================================================
    # rumps.Timer — メインスレッドで 0.3秒ごとにキューを処理
    # =========================================================

    @rumps.timer(0.3)
    def poll_queue(self, _):
        while True:
            try:
                msg = self._pending.get_nowait()
            except queue.Empty:
                break
            action = msg[0]
            if action == "status":
                self.status_item.title = f"状態: {msg[1]}"
            elif action == "title":
                self.title = msg[1]
            elif action == "toggle":
                self.toggle_item.title = msg[1]
            elif action == "transcript":
                ts, text = msg[1], msg[2]
                self.transcript_lines.append((ts, text))
                display = text if len(text) <= MAX_MENU_CHARS else text[:MAX_MENU_CHARS] + "..."
                mi = rumps.MenuItem(f"[{ts}] {display}", callback=self.on_copy_line)
                mi._full_text = text
                self.menu.insert_after(self.transcript_header.title, mi)
            elif action == "file_item_reset":
                self.file_item.title = "ファイルを文字起こし..."
            elif action == "notify":
                rumps.notification(msg[1], msg[2], msg[3])

    # =========================================================
    # 経過時間フォーマット
    # =========================================================

    def _elapsed_str(self) -> str:
        if self.rec_start_time is None:
            return "0:00"
        elapsed = int(time.time() - self.rec_start_time)
        h, m, s = elapsed // 3600, (elapsed % 3600) // 60, elapsed % 60
        if h > 0:
            return f"{h}:{m:02d}:{s:02d}"
        return f"{m}:{s:02d}"

    # =========================================================
    # 録音開始 / 停止
    # =========================================================

    def on_toggle(self, sender):
        if self.is_recording:
            self._stop_recording()
        else:
            self._start_recording()

    def _start_recording(self):
        if self.is_recording or getattr(self, '_is_starting', False):
            return
        self._is_starting = True
        self.toggle_item.title = "読み込み中..."
        self.status_item.title = "状態: モデル読み込み中..."

        def worker():
            try:
                if not self.is_model_loaded:
                    self._load_models()
                elif self.vad_model is not None:
                    self.vad_model.reset_states()
                self._begin_stream()
            except Exception as e:
                _log(f"_start_recording error: {e}")
                import traceback
                _log(traceback.format_exc())
                self._pending.put(("status", f"エラー: {e}"))
                self._pending.put(("toggle", "録音開始"))
                self._pending.put(("notify", "Talksribe", "エラー", str(e)[:100]))
            finally:
                self._is_starting = False

        threading.Thread(target=worker, daemon=True).start()

    def _load_models(self):
        from faster_whisper import WhisperModel
        _log(f"_load_models: loading whisper model={self.model_size}, device={self.compute_device}, compute_type={self.compute_type}")
        self._pending.put(("status", f"Whisper {self.model_size} 読み込み中..."))
        self.whisper_model = WhisperModel(
            self.model_size, device=self.compute_device, compute_type=self.compute_type
        )
        _log("_load_models: whisper loaded, loading VAD...")
        self._pending.put(("status", "VAD モデル読み込み中..."))
        import torch
        self.vad_model, _ = torch.hub.load(
            repo_or_dir="snakers4/silero-vad", model="silero_vad", verbose=False
        )
        self.vad_model.reset_states()
        self.is_model_loaded = True
        _log(f"_load_models: all models loaded (whisper={self.model_size})")

    def _begin_stream(self):
        _log(f"begin_stream: device={self.input_device_id}, sr={self.sr}")
        self.stop_event.clear()
        self.audio_q.clear()
        self.speech_buf = np.zeros(0, dtype=np.float32)
        self.silence_count = 0
        self.last_partial_at = 0.0
        self.rec_start_time = time.time()
        self.total_utterances = 0

        ts = datetime.now().strftime("%Y%m%d%H%M%S")
        self.save_path = os.path.join(SAVE_DIR, f"{ts}.txt")
        with open(self.save_path, "w", encoding="utf-8") as f:
            f.write(f"# Talksribe 録音 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        # マイク取得を子プロセスで実行（メインプロセスにマイクアイコンが出ない）
        self._capture_pipe, self._capture_stop, self._capture_proc = start_capture(
            sr=self.sr, chunk_sec=self.chunk_sec, device_id=self.input_device_id,
        )
        self.is_recording = True

        self._pending.put(("title", "REC"))
        self._pending.put(("toggle", "録音停止"))
        self._pending.put(("status", "録音中 0:00"))
        self._pending.put(("notify", "Talksribe", "録音開始", f"保存先: {self.save_path}"))

        threading.Thread(target=self._pipe_reader, daemon=True).start()
        threading.Thread(target=self._process_loop, daemon=True).start()

    def _stop_recording(self):
        self.stop_event.set()
        self._cleanup_capture()

        remaining = self.audio_q.drain()
        if remaining is not None:
            self.speech_buf = np.concatenate([self.speech_buf, remaining])

        if len(self.speech_buf) > self.sr * 0.3 and self.whisper_model:
            self._do_finalize()

        elapsed = self._elapsed_str()
        self.is_recording = False
        self.title = "MIC"
        self.toggle_item.title = "録音開始"

        if self.save_path:
            self.status_item.title = f"状態: 保存済み ({elapsed}, {self.total_utterances}件)"
        else:
            self.status_item.title = "状態: 停止"
        self.rec_start_time = None

    # =========================================================
    # 子プロセスからの音声読み取り
    # =========================================================

    def _pipe_reader(self):
        """子プロセスから pipe 経由で音声チャンクを受け取り AudioQueue に積む。"""
        pipe = self._capture_pipe
        chunk_bytes = int(self.sr * self.chunk_sec) * 4  # float32 = 4 bytes
        while not self.stop_event.is_set():
            try:
                if pipe.poll(0.5):
                    raw = pipe.recv_bytes()
                    audio = np.frombuffer(raw, dtype=np.float32)
                    self.audio_q.put(audio)
            except (EOFError, OSError):
                break

    def _cleanup_capture(self):
        """子プロセスを確実に終了させる。"""
        if self._capture_stop is not None and self._capture_proc is not None:
            try:
                stop_capture(self._capture_stop, self._capture_proc)
            except Exception as e:
                _log(f"_cleanup_capture error: {e}")
            self._capture_pipe = None
            self._capture_stop = None
            self._capture_proc = None

    # =========================================================
    # 処理ループ（バックグラウンドスレッド）
    # =========================================================

    def _process_loop(self):
        silence_limit = max(1, int(0.8 / self.chunk_sec))

        while not self.stop_event.is_set():
            # 省電力: 無音が続いたら sleep を長くする
            if self.silence_count > IDLE_THRESHOLD:
                time.sleep(IDLE_SLEEP_SEC)
            else:
                time.sleep(self.chunk_sec)

            new_audio = self.audio_q.drain()
            if new_audio is None or len(new_audio) < 512:
                self.silence_count += 1
                if self.silence_count >= silence_limit and len(self.speech_buf) > self.sr * 0.3:
                    self._do_finalize()
                elif self.silence_count >= silence_limit:
                    self.speech_buf = np.zeros(0, dtype=np.float32)
                # 定期的に経過時間を更新
                if self.silence_count % 4 == 0:
                    self._pending.put(("status", f"録音中 {self._elapsed_str()} ({self.total_utterances}件)"))
                continue

            chunk_samples = int(self.sr * self.chunk_sec)
            pos = 0
            while pos < len(new_audio):
                chunk = new_audio[pos:pos + chunk_samples]
                pos += chunk_samples

                if len(chunk) < 512:
                    continue

                if self._check_vad(chunk):
                    self.silence_count = 0
                    self.speech_buf = np.concatenate([self.speech_buf, chunk])
                    speech_dur = len(self.speech_buf) / self.sr

                    if speech_dur >= LONG_SPEECH_WINDOW_SEC:
                        self._do_finalize_window()
                    elif (
                        speech_dur >= PARTIAL_MIN_SEC
                        and time.monotonic() - self.last_partial_at >= PARTIAL_STATUS_INTERVAL_SEC
                    ):
                        self._do_partial()
                else:
                    self.silence_count += 1
                    if self.silence_count >= silence_limit and len(self.speech_buf) > self.sr * 0.3:
                        self._do_finalize()
                    elif self.silence_count >= silence_limit:
                        self.speech_buf = np.zeros(0, dtype=np.float32)

    def _check_vad(self, audio: np.ndarray) -> bool:
        import torch
        if len(audio) < 512:
            return False
        try:
            probs = []
            for i in range(0, len(audio) - 512 + 1, 512):
                tensor = torch.from_numpy(audio[i:i + 512]).float()
                probs.append(self.vad_model(tensor, self.sr).item())
            result = (sum(probs) / len(probs)) > self.vad_threshold if probs else False
            return result
        except Exception as e:
            _log(f"_check_vad error: {e}")
            return False

    def _collect_texts(self, audio: np.ndarray, emit_until_sec: Optional[float] = None) -> list[str]:
        segs, _ = self.whisper_model.transcribe(
            audio,
            language=self.language,
            beam_size=self.beam_size,
            vad_filter=False,
        )

        texts = []
        fallback = []
        for seg in segs:
            text = seg.text.strip()
            if not text:
                continue
            if emit_until_sec is None:
                texts.append(text)
                continue
            if seg.end <= emit_until_sec:
                texts.append(text)
            elif seg.start < emit_until_sec:
                fallback.append(text)

        return texts or fallback[:1]

    def _publish_texts(self, texts: list[str]):
        if not texts:
            return

        ts = time.strftime("%H:%M:%S")
        full = " ".join(texts)
        self.total_utterances += 1
        self._pending.put(("transcript", ts, full))
        self._pending.put(("notify", "Talksribe", ts, full[:100]))
        self._pending.put(("status", f"録音中 {self._elapsed_str()} ({self.total_utterances}件)"))
        if self.save_path:
            with open(self.save_path, "a", encoding="utf-8") as f:
                f.write(f"[{ts}] {full}\n")

    def _do_partial(self):
        try:
            max_samples = int(self.sr * PARTIAL_WINDOW_SEC)
            audio = self.speech_buf[-max_samples:] if len(self.speech_buf) > max_samples else self.speech_buf
            texts = self._collect_texts(audio)
            if texts:
                self._pending.put(("status", f"{self._elapsed_str()} ...{' '.join(texts)[:40]}"))
            self.last_partial_at = time.monotonic()
        except Exception:
            pass

    def _do_finalize_window(self):
        window_samples = int(self.sr * LONG_SPEECH_WINDOW_SEC)
        step_samples = int(self.sr * LONG_SPEECH_STEP_SEC)
        window_audio = self.speech_buf[:window_samples].copy()
        _log(
            f"_do_finalize_window: speech_buf len={len(self.speech_buf)}, "
            f"window_dur={len(window_audio) / self.sr:.2f}s"
        )

        try:
            texts = self._collect_texts(window_audio, emit_until_sec=LONG_SPEECH_STEP_SEC)
            _log(f"_do_finalize_window: texts={texts[:3]}")
            self._publish_texts(texts)
        except Exception as e:
            _log(f"_do_finalize_window error: {e}")
            import traceback
            _log(traceback.format_exc())
        finally:
            self.speech_buf = self.speech_buf[step_samples:]
            self.last_partial_at = 0.0

    def _do_finalize(self):
        _log(f"_do_finalize: speech_buf len={len(self.speech_buf)}, dur={len(self.speech_buf)/self.sr:.2f}s")
        try:
            texts = self._collect_texts(self.speech_buf)
            _log(f"_do_finalize: texts={texts[:3]}")
            self._publish_texts(texts)
        except Exception as e:
            _log(f"_do_finalize error: {e}")
            import traceback
            _log(traceback.format_exc())

        self.speech_buf = np.zeros(0, dtype=np.float32)
        self.last_partial_at = 0.0

    # =========================================================
    # ファイル文字起こし
    # =========================================================

    def on_file_transcribe(self, sender):
        # 処理中なら中断
        if self.is_file_processing:
            self.file_cancel_event.set()
            self.file_item.title = "中断中..."
            return

        filepath = self._open_file_dialog()
        if not filepath:
            return

        self.is_file_processing = True
        self.file_cancel_event.clear()
        self.file_item.title = "⏹ 中断"
        self.status_item.title = f"状態: ファイル読み込み中..."

        def worker():
            try:
                _log(f"file_transcribe start: {filepath}")
                self._transcribe_file(filepath)
                if self.file_cancel_event.is_set():
                    _log(f"file_transcribe cancelled: {filepath}")
                else:
                    _log(f"file_transcribe done: {filepath}")
            except Exception as e:
                _log(f"file_transcribe error: {e}")
                import traceback
                _log(traceback.format_exc())
                self._pending.put(("status", f"エラー: {e}"))
                self._pending.put(("notify", "Talksribe", "エラー", str(e)[:100]))
            finally:
                self.is_file_processing = False
                self.file_cancel_event.clear()
                self._pending.put(("file_item_reset",))
                if not self.is_recording:
                    self._pending.put(("status", "待機中"))

        threading.Thread(target=worker, daemon=True).start()

    def _open_file_dialog(self) -> Optional[str]:
        """osascript でファイル選択ダイアログを表示する。"""
        script = '''
tell application "System Events"
    activate
    set f to choose file with prompt "音声ファイルを選択してください" of type {"public.audio", "public.movie"}
    return POSIX path of f
end tell
'''
        try:
            result = subprocess.run(
                ["osascript", "-e", script],
                capture_output=True, text=True, timeout=120,
            )
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip()
        except Exception:
            pass
        return None

    def _transcribe_file(self, filepath: str):
        import soundfile as sf
        import librosa

        basename = os.path.splitext(os.path.basename(filepath))[0]
        self._pending.put(("status", f"読み込み中: {os.path.basename(filepath)}"))

        # 音声読み込み
        try:
            audio, sr_orig = sf.read(filepath, dtype="float32", always_2d=False)
        except Exception:
            # soundfile で読めない形式は ffmpeg で変換
            import tempfile
            tmp_wav = os.path.join(tempfile.gettempdir(), "talksribe_tmp.wav")
            result = subprocess.run(
                ["ffmpeg", "-y", "-i", filepath, "-ar", "16000", "-ac", "1", "-f", "wav", tmp_wav],
                capture_output=True, timeout=300,
            )
            if result.returncode != 0:
                raise RuntimeError(f"ffmpeg 変換失敗: {result.stderr.decode()[:200]}")
            audio, sr_orig = sf.read(tmp_wav, dtype="float32", always_2d=False)
            os.unlink(tmp_wav)

        if audio.ndim == 2:
            audio = np.mean(audio, axis=1)
        if sr_orig != 16000:
            audio = librosa.resample(audio, orig_sr=sr_orig, target_sr=16000, res_type="kaiser_best")
        sr = 16000

        duration = len(audio) / sr
        dur_str = f"{int(duration // 60)}:{int(duration % 60):02d}"
        self._pending.put(("status", f"文字起こし中: {basename} ({dur_str})"))

        # モデルロード
        if not self.is_model_loaded:
            self._load_models()

        # ファイル用: まず vad_filter=True で試し、結果0件なら vad_filter=False で再試行
        _log(f"_transcribe_file: audio len={len(audio)}, dur={duration:.1f}s, sr={sr}, model={self.model_size}")

        if self.file_cancel_event.is_set():
            return

        segs, info = self.whisper_model.transcribe(
            audio, language=self.language,
            beam_size=5, vad_filter=True,
        )

        # セグメントを1件ずつ処理（中断チェック付き）
        seg_list = []
        for seg in segs:
            if self.file_cancel_event.is_set():
                break
            seg_list.append(seg)
        _log(f"_transcribe_file: vad_filter=True → {len(seg_list)} segments")

        if not seg_list and not self.file_cancel_event.is_set():
            _log("_transcribe_file: vad_filter=True で結果0件 → vad_filter=False で再試行")
            segs2, info = self.whisper_model.transcribe(
                audio, language=self.language,
                beam_size=5, vad_filter=False,
            )
            for seg in segs2:
                if self.file_cancel_event.is_set():
                    break
                seg_list.append(seg)
            _log(f"_transcribe_file: vad_filter=False → {len(seg_list)} segments")

        cancelled = self.file_cancel_event.is_set()

        # 結果をファイルに保存（中断時も途中結果を保存）
        ts = datetime.now().strftime("%Y%m%d%H%M%S")
        suffix = f"_{basename}_中断" if cancelled else f"_{basename}"
        out_path = os.path.join(SAVE_DIR, f"{ts}{suffix}.txt")
        count = 0
        lang = info.language if info else self.language
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(f"# {os.path.basename(filepath)}\n")
            header = f"# 長さ: {dur_str}  言語: {lang}  モデル: {self.model_size}"
            if cancelled:
                header += "  ※中断"
            f.write(header + "\n\n")
            for seg in seg_list:
                text = seg.text.strip()
                if not text:
                    continue
                count += 1
                start_ts = self._format_ts(seg.start)
                end_ts = self._format_ts(seg.end)
                line = f"[{start_ts} → {end_ts}] {text}"
                f.write(line + "\n")
                self._pending.put(("transcript", start_ts, text))

        if cancelled:
            self._pending.put(("notify", "Talksribe", "文字起こし中断",
                              f"{basename}: {count}件まで保存 → {out_path}"))
            self._pending.put(("status", f"中断: {basename} ({count}件) → {os.path.basename(out_path)}"))
        else:
            self._pending.put(("notify", "Talksribe", "文字起こし完了",
                              f"{basename}: {count}セグメント → {out_path}"))
            self._pending.put(("status", f"完了: {basename} ({count}件) → {os.path.basename(out_path)}"))

        # Finder で保存先を表示
        if count > 0:
            subprocess.Popen(["open", "-R", out_path])

    @staticmethod
    def _format_ts(sec: float) -> str:
        h = int(sec // 3600)
        m = int((sec % 3600) // 60)
        s = int(sec % 60)
        if h > 0:
            return f"{h}:{m:02d}:{s:02d}"
        return f"{m:02d}:{s:02d}"

    # =========================================================
    # コピー / クリア / フォルダ
    # =========================================================

    def on_copy_line(self, sender):
        text = getattr(sender, "_full_text", sender.title)
        copy_to_clipboard(text)
        self.status_item.title = "状態: コピー済み"

    def on_copy_all(self, sender):
        if not self.transcript_lines:
            rumps.notification("Talksribe", "", "履歴がありません")
            return
        full = "\n".join(text for _, text in self.transcript_lines)
        copy_to_clipboard(full)
        self.status_item.title = f"状態: 全文コピー ({len(self.transcript_lines)}行)"

    def on_copy_last(self, sender):
        if not self.transcript_lines:
            rumps.notification("Talksribe", "", "履歴がありません")
            return
        _, text = self.transcript_lines[-1]
        copy_to_clipboard(text)
        self.status_item.title = "状態: 最新行コピー済み"

    def on_clear(self, sender):
        self.transcript_lines.clear()
        for k in list(self.menu.keys()):
            if k.startswith("[") and ":" in k[:12]:
                try:
                    del self.menu[k]
                except Exception:
                    pass
        self.status_item.title = "状態: 履歴クリア"

    def on_open_folder(self, sender):
        subprocess.Popen(["open", SAVE_DIR])

    # =========================================================
    # 設定変更
    # =========================================================

    def on_change_language(self, sender):
        self.language = sender._lang_code
        for _, item in self._lang_items.items():
            prefix = "● " if item._lang_code == sender._lang_code else "  "
            item.title = f"{prefix}{item._lang_label}"
        self.status_item.title = f"状態: 言語 → {sender._lang_label}"

    def on_change_model(self, sender):
        new_size = sender._model_size
        if new_size == self.model_size:
            return
        was_rec = self.is_recording
        if was_rec:
            self._stop_recording()
        self.model_size = new_size
        for size, item in self._model_items.items():
            prefix = "● " if size == new_size else "  "
            item.title = f"{prefix}{size}"
        self.whisper_model = None
        self.vad_model = None
        self.is_model_loaded = False
        self.status_item.title = f"状態: モデル → {new_size}"
        if was_rec:
            self._start_recording()

    def on_change_device(self, sender):
        new_id = sender._dev_id
        was_rec = self.is_recording
        if was_rec:
            self._stop_recording()
        self.input_device_id = new_id
        for did, item in self._device_items.items():
            prefix = "● " if did == new_id else "  "
            item.title = f"{prefix}{item._dev_name}"
        self.status_item.title = f"状態: デバイス → {sender._dev_name}"
        if was_rec:
            self._start_recording()

    def on_quit(self, sender):
        if self.is_recording:
            self._stop_recording()
        self._cleanup_capture()
        rumps.quit_application()


def main():
    TalksribeMenuBar().run()


if __name__ == "__main__":
    main()
