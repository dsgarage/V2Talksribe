#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""macOS メニューバー UI（薄い表示層）。

音声処理は engine.TranscriptionEngine（単一ワーカースレッド）に委譲し、
本モジュールは「コマンド送信」と「イベント受信 → メニュー更新」だけを担う。
状態の所有はエンジン側にあるため、UI スレッドは重い処理をせずフリーズしない。
"""
from __future__ import annotations

import faulthandler
import fcntl
import json
import os
import platform
import queue
import signal
import subprocess
import sys
import threading
import time
import traceback
from collections import deque
from typing import Optional

import rumps

try:
    import AppKit
except Exception:  # 無ければ診断の該当項目が n/a になるだけ
    AppKit = None
try:
    import psutil
except Exception:
    psutil = None

import applog
import devices
from engine import TranscriptionEngine
from file_transcribe import FileTranscriber
from settings import POWER_PROFILES, SAVE_DIR, AppSettings

MAX_LINES = 30
MAX_MENU_CHARS = 55
LOG_PATH = applog.LOG_PATH
MAX_EVENTS_PER_TICK = 200
WORKER_STALL_ALERT_SEC = 120.0
DIAG_FORCE_SEC = 600.0
SYSTEM_DEFAULT_LABEL = "システム既定"
NOTIFY_BACKEND = "rumps"          # 決定済み: まず rumps。手で確かめて出なければ "osascript" に変える
_LOCK_FD = None
_FAULT_FH = None

os.makedirs(SAVE_DIR, exist_ok=True)


def _log(msg: str):
    applog.log(msg)


def copy_to_clipboard(text: str):
    p = subprocess.Popen(["pbcopy"], stdin=subprocess.PIPE)
    p.communicate(text.encode("utf-8"))


LANG_LABELS = [("ja", "日本語"), ("en", "English"), ("zh", "中文"), (None, "自動検出")]
MODEL_SIZES = ["tiny", "base", "small", "medium", "large-v3"]


class TalksribeMenuBar(rumps.App):
    def __init__(self):
        super().__init__("MIC", quit_button=None)

        self.settings = AppSettings.load()
        self.engine = TranscriptionEngine(self.settings)
        self.engine.run()

        # ライブ確定行の履歴（UI 側で保持）
        self.transcript_lines: deque = deque(maxlen=MAX_LINES)
        self.transcript_items: deque = deque()
        self._transcript_menu_seq = 0

        # ファイル文字起こし
        self.is_file_processing = False
        self.file_cancel_event = threading.Event()

        # --- メニュー項目 ---
        self.toggle_item = rumps.MenuItem("録音開始", callback=self.on_toggle)
        self._toggle_pending = False
        self._toggle_pending_at = 0.0
        self.file_item = rumps.MenuItem("ファイルを文字起こし...", callback=self.on_file_transcribe)
        self.status_item = rumps.MenuItem("状態: 待機中")
        self.status_item.set_callback(None)
        self.detail_item = rumps.MenuItem("内部状態: 待機")
        self.detail_item.set_callback(None)
        self.alert_item = rumps.MenuItem("注意: なし")
        self.alert_item.set_callback(None)
        runtime_label = "パッケージ版" if getattr(sys, "frozen", False) else "ソース版"
        self.runtime_item = rumps.MenuItem(f"実行: {runtime_label}")
        self.runtime_item.set_callback(None)

        self.copy_item = rumps.MenuItem("全文コピー", callback=self.on_copy_all)
        self.copy_last_item = rumps.MenuItem("最新行コピー", callback=self.on_copy_last)
        self.clear_item = rumps.MenuItem("履歴クリア", callback=self.on_clear)
        self.open_folder_item = rumps.MenuItem("保存フォルダを開く", callback=self.on_open_folder)
        self.open_log_item = rumps.MenuItem("デバッグログを開く", callback=self.on_open_log)

        self.lang_menu = rumps.MenuItem("言語")
        self._lang_items = {}
        for code, label in LANG_LABELS:
            item = rumps.MenuItem(f"{self._check(code == self.settings.language)}{label}", callback=self.on_change_language)
            item._lang_code = code
            item._lang_label = label
            self._lang_items[str(code)] = item
            self.lang_menu.add(item)

        self.model_menu = rumps.MenuItem("モデル")
        self._model_items = {}
        for size in MODEL_SIZES:
            item = rumps.MenuItem(f"{self._check(size == self.settings.model_size)}{size}", callback=self.on_change_model)
            item._model_size = size
            self._model_items[size] = item
            self.model_menu.add(item)

        self.power_menu = rumps.MenuItem("電力")
        self._power_items = {}
        for profile_name in ["eco", "balanced", "accuracy"]:
            profile = POWER_PROFILES[profile_name]
            item = rumps.MenuItem(
                f"{self._check(profile_name == self.settings.power_profile)}{profile['label']}",
                callback=self.on_change_power_profile,
            )
            item._power_profile = profile_name
            item._power_label = profile["label"]
            self._power_items[profile_name] = item
            self.power_menu.add(item)

        self.preprocess_menu = rumps.MenuItem("音声前処理")
        self.preprocess_normalize_item = rumps.MenuItem("  ノーマライズ", callback=self.on_toggle_preprocess_normalize)
        self.preprocess_denoise_item = rumps.MenuItem("  ノイズ除去", callback=self.on_toggle_preprocess_denoise)
        self.preprocess_menu.add(self.preprocess_normalize_item)
        self.preprocess_menu.add(self.preprocess_denoise_item)
        self._refresh_preprocess_menu()

        self.device_menu = rumps.MenuItem("入力デバイス")
        self._device_items = {}
        self._pending_alerts: list = []
        devs = devices.list_input_devices()
        if self.settings.migrate_input_device_name({d["index"]: d["name"] for d in devs}):
            try:
                self.settings.save()
            except Exception as e:
                _log(f"settings save failed: {e}")
            self._pending_alerts.append(
                f"入力デバイスを名前で保存しました: {self.settings.input_device_name or 'システム既定'}")
        self._build_device_menu(devs)

        self._last_diag_line = None
        self._last_diag_at = 0.0
        self._stall_alerted = False
        rumps.events.on_sleep.register(lambda: _log("system sleep"))
        rumps.events.on_wake.register(lambda: _log("system wake"))

        self.transcript_header = rumps.MenuItem("--- 文字起こし ---")
        self.transcript_header.set_callback(None)
        self.quit_item = rumps.MenuItem("終了", callback=self.on_quit)

        self.menu = [
            self.toggle_item,
            self.file_item,
            self.status_item,
            self.detail_item,
            self.alert_item,
            self.runtime_item,
            None,
            self.copy_item,
            self.copy_last_item,
            self.clear_item,
            self.open_folder_item,
            self.open_log_item,
            None,
            self.lang_menu,
            self.power_menu,
            self.preprocess_menu,
            self.model_menu,
            self.device_menu,
            None,
            self.transcript_header,
            self.quit_item,
        ]

    # ------------------------------------------------------------------
    @staticmethod
    def _check(active: bool) -> str:
        return "● " if active else "  "

    def _emit(self, *event):
        """ファイル処理などからイベントバス(engine.events)へ流し込む。"""
        self.engine.events.put(event)

    def _build_device_menu(self, devs=None):
        if devs is None:
            devs = devices.list_input_devices()
        # rumps の clear() は、子を 1 つも足していない項目(_menu が None)では落ちる
        if self.device_menu._menu is not None:
            self.device_menu.clear()
        self._device_items = {}
        selected = self.settings.input_device_name
        sel_norm = devices.normalize_device_name(selected) if selected else None
        item = rumps.MenuItem(f"{self._check(sel_norm is None)}{SYSTEM_DEFAULT_LABEL}",
                              callback=self.on_change_device)
        item._dev_name = None
        item._dev_label = SYSTEM_DEFAULT_LABEL
        self._device_items[None] = item
        self.device_menu.add(item)
        for d in devs:
            name = d["name"]
            norm = devices.normalize_device_name(name)
            label = norm + ("(既定)" if d.get("is_default") else "")
            item = rumps.MenuItem(f"{self._check(norm == sel_norm)}{label}", callback=self.on_change_device)
            item._dev_name = norm
            item._dev_label = label
            key = norm if norm not in self._device_items else f"{norm} ({d['index']})"
            self._device_items[key] = item
            self.device_menu.add(item)
        self.device_menu.add(None)
        self.device_menu.add(rumps.MenuItem("一覧を更新", callback=self.on_refresh_devices))

    def _refresh_preprocess_menu(self):
        pre = self.settings.preprocess
        self.preprocess_normalize_item.title = f"{self._check(pre.normalize_enabled)}ノーマライズ"
        self.preprocess_denoise_item.title = f"{self._check(pre.denoise_enabled)}ノイズ除去"

    # ==================================================================
    # エンジンからのイベントを 0.3 秒ごとにメインスレッドで処理
    # ==================================================================

    @rumps.timer(0.3)
    def poll_events(self, _):
        while self._pending_alerts:
            self._set_alert(self._pending_alerts.pop(0))
        for _i in range(MAX_EVENTS_PER_TICK):
            try:
                msg = self.engine.events.get_nowait()
            except queue.Empty:
                break
            try:
                self._handle_event(msg)
            except Exception as e:
                _log(f"poll_events error: {e}\n{traceback.format_exc()}")
        else:
            q = self.engine.events.qsize()
            if q > 1000:
                _log(f"poll_events backlog: {q}")

    def _handle_event(self, msg):
        action = msg[0]
        if action == "status":
            self.status_item.title = f"状態: {msg[1]}"
        elif action == "detail":
            self.detail_item.title = f"内部状態: {msg[1]}"
        elif action == "title":
            self.title = msg[1]
        elif action == "toggle":
            self._toggle_pending = False
            self.toggle_item.title = msg[1]
        elif action == "transcript":
            self._append_transcript(msg[1], msg[2])
        elif action == "file_item_reset":
            self.file_item.title = "ファイルを文字起こし..."
        elif action == "alert":
            self._set_alert(msg[1])
        elif action == "devices":
            self._build_device_menu(msg[1])
            self.status_item.title = f"状態: 入力デバイス {len(msg[1])} 件"
        elif action == "notify":
            self._notify(msg[1], msg[2], msg[3])

    def _set_alert(self, text: str):
        self.alert_item.title = f"注意: {time.strftime('%H:%M:%S')} {text[:60]}"

    def _notify(self, title: str, subtitle: str, message: str):
        if NOTIFY_BACKEND == "osascript":
            script = (f"display notification {json.dumps(message, ensure_ascii=False)} "
                      f"with title {json.dumps(title, ensure_ascii=False)} "
                      f"subtitle {json.dumps(subtitle, ensure_ascii=False)}")
            try:
                subprocess.Popen(["osascript", "-e", script])
            except Exception as e:
                _log(f"notify(osascript) failed: {e}")
            return
        try:
            rumps.notification(title, subtitle, message)
        except Exception as e:
            _log(f"notify(rumps) failed: {e}")

    def _append_transcript(self, ts: str, text: str):
        self.transcript_lines.append((ts, text))
        display = text if len(text) <= MAX_MENU_CHARS else text[:MAX_MENU_CHARS] + "..."
        self._transcript_menu_seq += 1
        title = f"[{ts}] {display}"
        if title in self.menu.keys():
            title = f"{title} ({self._transcript_menu_seq})"
        mi = rumps.MenuItem(title, callback=self.on_copy_line)
        mi._full_text = text
        self.menu.insert_after(self.transcript_header.title, mi)
        self.transcript_items.append(mi)
        while len(self.transcript_items) > MAX_LINES:
            old_item = self.transcript_items.popleft()
            try:
                del self.menu[old_item.title]
            except Exception as e:
                _log(f"transcript prune error: {e}")

    # ==================================================================
    # 録音トグル
    # ==================================================================

    def on_toggle(self, sender):
        if self._toggle_pending:
            _log("ui: toggle ignored (pending)")
            return
        self._toggle_pending = True
        self._toggle_pending_at = time.monotonic()
        if self.engine.is_recording or self.engine.is_loading:
            # 押した瞬間に見た目を変える(ワーカーが文字起こし中でも反応が見える)
            self.toggle_item.title = "停止中..."
            self.status_item.title = "状態: 停止中..."
            _log("ui: stop clicked")
            self.engine.stop()
        else:
            self.toggle_item.title = "開始中..."
            self.status_item.title = "状態: 開始中..."
            _log("ui: start clicked")
            self.engine.start()

    # ==================================================================
    # ファイル文字起こし
    # ==================================================================

    def on_file_transcribe(self, sender):
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
        self.status_item.title = "状態: ファイル読み込み中..."
        self.detail_item.title = "内部状態: ファイル文字起こしを準備中"

        def worker():
            try:
                ft = FileTranscriber(self.settings, self._emit, self.file_cancel_event)
                ft.transcribe(filepath)
            except Exception as e:
                _log(f"file_transcribe error: {e}\n{traceback.format_exc()}")
                self._emit("status", f"エラー: {e}")
                self._emit("detail", "ファイル文字起こしに失敗")
                self._emit("notify", "Talksribe", "エラー", f"ファイル文字起こしに失敗: {str(e)[:80]}")
            finally:
                self.is_file_processing = False
                self.file_cancel_event.clear()
                self._emit("file_item_reset")
                if not self.engine.is_recording:
                    self._emit("status", "待機中")
                    self._emit("detail", "待機")

        threading.Thread(target=worker, daemon=True).start()

    def _open_file_dialog(self) -> Optional[str]:
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
            if result.returncode != 0:
                _log(f"_open_file_dialog non-zero exit: {result.returncode} stderr={result.stderr.strip()[:200]}")
        except Exception as e:
            _log(f"_open_file_dialog error: {e}")
        return None

    # ==================================================================
    # コピー / クリア / フォルダ
    # ==================================================================

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
        while self.transcript_items:
            item = self.transcript_items.pop()
            try:
                del self.menu[item.title]
            except Exception as e:
                _log(f"on_clear error: {e}")
        self.status_item.title = "状態: 履歴クリア"
        self.detail_item.title = "内部状態: 待機"

    def on_open_folder(self, sender):
        subprocess.Popen(["open", SAVE_DIR])

    def on_open_log(self, sender):
        subprocess.Popen(["open", LOG_PATH])

    # ==================================================================
    # 設定変更 → エンジンへコマンド送信 + チェックマーク更新
    # ==================================================================

    def on_change_language(self, sender):
        _log(f"ui: language -> {sender._lang_code}")
        self.engine.set_language(sender._lang_code)
        for _, item in self._lang_items.items():
            item.title = f"{self._check(item._lang_code == sender._lang_code)}{item._lang_label}"

    def on_change_model(self, sender):
        _log(f"ui: model -> {sender._model_size}")
        self.engine.set_model(sender._model_size)
        for size, item in self._model_items.items():
            item.title = f"{self._check(size == sender._model_size)}{size}"

    def on_change_power_profile(self, sender):
        _log(f"ui: profile -> {sender._power_profile}")
        self.engine.set_profile(sender._power_profile)
        for profile_name, item in self._power_items.items():
            item.title = f"{self._check(profile_name == sender._power_profile)}{item._power_label}"
        # プロファイル既定モデルへメニューも追従
        expected_model = POWER_PROFILES[sender._power_profile]["model_size"]
        for size, item in self._model_items.items():
            item.title = f"{self._check(size == expected_model)}{size}"

    def on_refresh_devices(self, sender):
        def worker():
            devs = devices.list_input_devices()
            self.engine.events.put(("devices", devs))
        threading.Thread(target=worker, daemon=True).start()
        self.status_item.title = "状態: 入力デバイスの一覧を更新中..."

    def on_change_device(self, sender):
        _log(f"ui: device -> {sender._dev_name}")
        self.engine.set_device(sender._dev_name)
        self.settings.input_device_name = sender._dev_name      # UI 側の表示用(保存は engine が行う)
        for key, item in self._device_items.items():
            item.title = f"{self._check(item._dev_name == sender._dev_name)}{item._dev_label}"

    def on_toggle_preprocess_normalize(self, sender):
        self.settings.preprocess.normalize_enabled = not self.settings.preprocess.normalize_enabled
        self._refresh_preprocess_menu()
        self.engine.set_preprocess()

    def on_toggle_preprocess_denoise(self, sender):
        self.settings.preprocess.denoise_enabled = not self.settings.preprocess.denoise_enabled
        self._refresh_preprocess_menu()
        self.engine.set_preprocess()

    def on_quit(self, sender):
        if self.engine.is_recording:
            self.engine.stop()
        self.engine.shutdown()
        rumps.quit_application()


    # ==================================================================
    # 見張り・診断
    # ==================================================================

    @rumps.timer(5)
    def watchdog(self, _):
        try:
            if self.engine.ensure_worker():
                _log("watchdog: worker restarted")
                self._set_alert("内部の処理を再起動しました")
            if self.engine.is_recording:
                age = time.monotonic() - self.engine.heartbeat_at
                if age > WORKER_STALL_ALERT_SEC and not self._stall_alerted:
                    self._stall_alerted = True
                    _log(f"watchdog: worker heartbeat stalled {age:.0f}s")
                    self._set_alert(f"処理が {int(age)} 秒止まっています")
                    if _FAULT_FH is not None:
                        faulthandler.dump_traceback(file=_FAULT_FH, all_threads=True)
                elif age <= WORKER_STALL_ALERT_SEC:
                    self._stall_alerted = False
            if self._toggle_pending and time.monotonic() - self._toggle_pending_at > 20.0:
                self._toggle_pending = False
                self.toggle_item.title = "録音停止" if (self.engine.is_recording or self.engine.is_loading) else "録音開始"
                _log("ui: toggle pending timeout → reset")
        except Exception as e:
            _log(f"watchdog error: {e}")

    @rumps.timer(60)
    def diag(self, _):
        parts = {}

        def put(key, fn):
            try:
                parts[key] = fn()
            except Exception:
                parts[key] = "n/a"
        put("worker_alive", self.engine.worker_alive)
        put("hb_age", lambda: round(time.monotonic() - self.engine.heartbeat_at, 1))
        put("events_q", self.engine.events.qsize)
        put("recording", lambda: self.engine.is_recording)
        put("loading", lambda: self.engine.is_loading)
        si = lambda: self._nsapp.nsstatusitem
        put("visible", lambda: bool(si().isVisible()))
        put("occluded", lambda: not (int(si().button().window().occlusionState()) & 2))
        put("x", lambda: round(float(si().button().window().frame().origin.x), 1))
        put("screen_w", lambda: round(float(AppKit.NSScreen.mainScreen().frame().size.width), 1))
        put("rss_mb", lambda: round(psutil.Process().memory_info().rss / 1024 / 1024, 1))
        put("num_fds", lambda: psutil.Process().num_fds())
        put("num_threads", lambda: psutil.Process().num_threads())
        key = {k: v for k, v in parts.items() if k not in ("hb_age", "rss_mb")}   # 変化の判定から外す
        line = " ".join(f"{k}={v}" for k, v in parts.items())
        now = time.monotonic()
        if key != self._last_diag_line or now - self._last_diag_at >= DIAG_FORCE_SEC:
            _log(f"diag: {line}")
            self._last_diag_line = key
            self._last_diag_at = now


def _acquire_single_instance_lock() -> bool:
    global _LOCK_FD
    path = os.path.join(SAVE_DIR, "talksribe.lock")
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        os.close(fd)
        return False
    os.ftruncate(fd, 0)
    os.write(fd, f"{os.getpid()}\n".encode())
    _LOCK_FD = fd                  # プロセスが終わるまで閉じない(落ちれば OS が外す)
    return True


def _redirect_stdio():
    try:
        if sys.stderr is not None and sys.stderr.isatty():
            return
    except Exception:
        pass
    path = os.path.join(SAVE_DIR, "stderr.log")
    try:
        if os.path.exists(path) and os.path.getsize(path) > 5 * 1024 * 1024:
            os.replace(path, path + ".1")
        f = open(path, "a", encoding="utf-8", buffering=1)
        f.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} pid={os.getpid()} =====\n")
        sys.stdout = f
        sys.stderr = f
    except OSError as e:
        _log(f"stderr redirect failed: {e}")


def _enable_faulthandler():
    global _FAULT_FH
    try:
        _FAULT_FH = open(os.path.join(SAVE_DIR, "faulthandler.log"), "a", encoding="utf-8")
        _FAULT_FH.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} pid={os.getpid()} =====\n")
        _FAULT_FH.flush()
        faulthandler.enable(file=_FAULT_FH, all_threads=True)
        faulthandler.register(signal.SIGUSR1, file=_FAULT_FH, all_threads=True)
    except Exception as e:
        _log(f"faulthandler setup failed: {e}")


def _log_startup():
    def get(fn):
        try:
            return fn()
        except Exception:
            return "n/a"
    pid = get(os.getpid)
    ppid = get(os.getppid)
    parent = get(lambda: psutil.Process().parent().name())
    exe = get(lambda: sys.executable)
    cwd = get(os.getcwd)
    macos = get(lambda: platform.mac_ver()[0])
    runtime = "パッケージ版" if getattr(sys, "frozen", False) else "ソース版"
    _log(f"startup: pid={pid} ppid={ppid} parent={parent} exe={exe} cwd={cwd} macos={macos} runtime={runtime}")


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    if not _acquire_single_instance_lock():
        _log(f"二重起動を検知: pid={os.getpid()} → 終了")
        subprocess.run(["osascript", "-e",
                        'display notification "Talksribe はすでに起動しています" with title "Talksribe"'],
                       capture_output=True, timeout=10)
        return
    _redirect_stdio()
    _enable_faulthandler()
    _log_startup()
    TalksribeMenuBar().run()


if __name__ == "__main__":
    main()
