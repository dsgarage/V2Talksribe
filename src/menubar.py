#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""macOS メニューバー UI（薄い表示層）。

音声処理は engine.TranscriptionEngine（単一ワーカースレッド）に委譲し、
本モジュールは「コマンド送信」と「イベント受信 → メニュー更新」だけを担う。
状態の所有はエンジン側にあるため、UI スレッドは重い処理をせずフリーズしない。
"""
from __future__ import annotations

import os
import queue
import subprocess
import sys
import threading
import traceback
from collections import deque
from typing import Optional

import rumps

from engine import TranscriptionEngine
from file_transcribe import FileTranscriber
from settings import POWER_PROFILES, SAVE_DIR, AppSettings

MAX_LINES = 30
MAX_MENU_CHARS = 55
LOG_PATH = os.path.join(SAVE_DIR, "debug.log")

os.makedirs(SAVE_DIR, exist_ok=True)


def _log(msg: str):
    import time
    try:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(f"[{time.strftime('%H:%M:%S')}] {msg}\n")
    except OSError:
        pass


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
        self.file_item = rumps.MenuItem("ファイルを文字起こし...", callback=self.on_file_transcribe)
        self.status_item = rumps.MenuItem("状態: 待機中")
        self.status_item.set_callback(None)
        self.detail_item = rumps.MenuItem("内部状態: 待機")
        self.detail_item.set_callback(None)
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
        self._build_device_menu()

        self.transcript_header = rumps.MenuItem("--- 文字起こし ---")
        self.transcript_header.set_callback(None)
        self.quit_item = rumps.MenuItem("終了", callback=self.on_quit)

        self.menu = [
            self.toggle_item,
            self.file_item,
            self.status_item,
            self.detail_item,
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

    def _build_device_menu(self):
        import sounddevice as sd
        devices = sd.query_devices()
        default_id = sd.default.device[0]
        selected = self.settings.input_device_id
        for i, d in enumerate(devices):
            if d["max_input_channels"] > 0:
                is_sel = i == selected if selected is not None else i == default_id
                name = d["name"]
                item = rumps.MenuItem(f"{self._check(is_sel)}{name}", callback=self.on_change_device)
                item._dev_id = i
                item._dev_name = name
                self._device_items[i] = item
                self.device_menu.add(item)

    def _refresh_preprocess_menu(self):
        pre = self.settings.preprocess
        self.preprocess_normalize_item.title = f"{self._check(pre.normalize_enabled)}ノーマライズ"
        self.preprocess_denoise_item.title = f"{self._check(pre.denoise_enabled)}ノイズ除去"

    # ==================================================================
    # エンジンからのイベントを 0.3 秒ごとにメインスレッドで処理
    # ==================================================================

    @rumps.timer(0.3)
    def poll_events(self, _):
        while True:
            try:
                msg = self.engine.events.get_nowait()
            except queue.Empty:
                break
            try:
                self._handle_event(msg)
            except Exception as e:
                _log(f"poll_events error: {e}\n{traceback.format_exc()}")

    def _handle_event(self, msg):
        action = msg[0]
        if action == "status":
            self.status_item.title = f"状態: {msg[1]}"
        elif action == "detail":
            self.detail_item.title = f"内部状態: {msg[1]}"
        elif action == "title":
            self.title = msg[1]
        elif action == "toggle":
            self.toggle_item.title = msg[1]
        elif action == "transcript":
            self._append_transcript(msg[1], msg[2])
        elif action == "file_item_reset":
            self.file_item.title = "ファイルを文字起こし..."
        elif action == "notify":
            rumps.notification(msg[1], msg[2], msg[3])

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
        if self.engine.is_recording:
            self.engine.stop()
        else:
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
        self.engine.set_language(sender._lang_code)
        for _, item in self._lang_items.items():
            item.title = f"{self._check(item._lang_code == sender._lang_code)}{item._lang_label}"

    def on_change_model(self, sender):
        self.engine.set_model(sender._model_size)
        for size, item in self._model_items.items():
            item.title = f"{self._check(size == sender._model_size)}{size}"

    def on_change_power_profile(self, sender):
        self.engine.set_profile(sender._power_profile)
        for profile_name, item in self._power_items.items():
            item.title = f"{self._check(profile_name == sender._power_profile)}{item._power_label}"
        # プロファイル既定モデルへメニューも追従
        expected_model = POWER_PROFILES[sender._power_profile]["model_size"]
        for size, item in self._model_items.items():
            item.title = f"{self._check(size == expected_model)}{size}"

    def on_change_device(self, sender):
        self.engine.set_device(sender._dev_id)
        for did, item in self._device_items.items():
            item.title = f"{self._check(did == sender._dev_id)}{item._dev_name}"

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


def main():
    TalksribeMenuBar().run()


if __name__ == "__main__":
    main()
