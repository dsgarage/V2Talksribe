#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ファイル文字起こし。

ライブ経路(engine)とは独立。専用の Transcriber を持ち、長尺の連続性を優先して
``condition_on_previous_text=True`` ＋ 緩めの信頼度フィルタで動く。
進捗・完了はコールバック ``emit(*event)`` で UI へ通知する。
"""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
import threading
from datetime import datetime, timedelta
from typing import Callable, Optional

from audio_preprocess import preprocess_audio_file
from settings import SAVE_DIR, AppSettings
from transcriber import Transcriber

FILE_PROGRESS_STEP_SEC = 20.0
FILE_PROGRESS_STEP_RATIO = 0.1

LOG_PATH = os.path.join(SAVE_DIR, "debug.log")


def _log(msg: str):
    try:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}\n")
    except OSError:
        pass


# ======================================================================
# メディア情報ヘルパ
# ======================================================================

def format_media_duration(duration: Optional[float]) -> str:
    if duration is None:
        return "不明"
    hours = int(duration // 3600)
    minutes = int((duration % 3600) // 60)
    seconds = int(duration % 60)
    if hours > 0:
        return f"{hours}:{minutes:02d}:{seconds:02d}"
    return f"{minutes}:{seconds:02d}"


def format_ts(sec: float) -> str:
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = int(sec % 60)
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def probe_media_duration_ffprobe(filepath: str) -> Optional[float]:
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", filepath,
        ],
        capture_output=True, text=True, timeout=30,
    )
    if result.returncode != 0:
        return None
    try:
        return float(result.stdout.strip())
    except ValueError:
        return None


def _get_filesystem_created_at(filepath: str) -> Optional[datetime]:
    try:
        stat_result = os.stat(filepath)
    except OSError:
        return None
    ts = getattr(stat_result, "st_birthtime", None)
    if ts is None:
        ts = stat_result.st_mtime
    return datetime.fromtimestamp(ts)


def _parse_media_datetime(raw: str) -> Optional[datetime]:
    value = raw.strip()
    if not value:
        return None
    normalized = value.replace("Z", "+00:00")
    if len(normalized) >= 5 and normalized[-5] in {"+", "-"} and normalized[-3] != ":":
        normalized = f"{normalized[:-2]}:{normalized[-2:]}"
    if len(normalized) >= 19 and normalized[4] == ":" and normalized[7] == ":":
        normalized = normalized.replace(":", "-", 2)
    try:
        dt = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if dt.tzinfo is not None:
        dt = dt.astimezone().replace(tzinfo=None)
    return dt


def _probe_media_start_at_ffprobe(filepath: str) -> Optional[datetime]:
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-show_entries",
            "format_tags=creation_time:format_tags=com.apple.quicktime.creationdate:"
            "stream_tags=creation_time:stream_tags=com.apple.quicktime.creationdate",
            "-of", "json", filepath,
        ],
        capture_output=True, text=True, timeout=30,
    )
    if result.returncode != 0:
        return None
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError:
        return None

    candidates = []
    format_tags = payload.get("format", {}).get("tags", {})
    if isinstance(format_tags, dict):
        candidates.extend([
            format_tags.get("com.apple.quicktime.creationdate"),
            format_tags.get("creation_time"),
        ])
    for stream in payload.get("streams", []):
        tags = stream.get("tags", {})
        if not isinstance(tags, dict):
            continue
        candidates.extend([
            tags.get("com.apple.quicktime.creationdate"),
            tags.get("creation_time"),
        ])
    for candidate in candidates:
        if not candidate:
            continue
        parsed = _parse_media_datetime(candidate)
        if parsed is not None:
            return parsed
    return None


def resolve_recording_start_at(
    filepath: str, duration: Optional[float]
) -> "tuple[Optional[datetime], str]":
    media_start_at = _probe_media_start_at_ffprobe(filepath)
    if media_start_at is not None:
        return media_start_at, "メディア埋め込み時刻"
    file_created_at = _get_filesystem_created_at(filepath)
    if file_created_at is None:
        return None, "基準時刻なし"
    if duration is None:
        return file_created_at, "ファイル作成日時"
    return file_created_at - timedelta(seconds=duration), "ファイル作成日時 - 長さ"


def format_transcript_line_ts(
    offset_sec: float, recording_start_at: Optional[datetime]
) -> str:
    if recording_start_at is None:
        return format_ts(offset_sec)
    return (recording_start_at + timedelta(seconds=offset_sec)).strftime("%H:%M:%S")


def convert_media_to_temp_wav(filepath: str) -> str:
    tmp_wav = os.path.join(tempfile.gettempdir(), f"talksribe_{os.getpid()}.wav")
    result = subprocess.run(
        ["ffmpeg", "-y", "-i", filepath, "-ar", "16000", "-ac", "1", "-f", "wav", tmp_wav],
        capture_output=True, timeout=300,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg 変換失敗: {result.stderr.decode()[:200]}")
    return tmp_wav


# ======================================================================
# ファイル文字起こし本体
# ======================================================================

class FileTranscriber:
    def __init__(
        self,
        settings: AppSettings,
        emit: Callable[..., None],
        cancel_event: threading.Event,
    ):
        self.settings = settings
        self.emit = emit
        self.cancel = cancel_event
        self._transcriber: Optional[Transcriber] = None

    def _ensure_transcriber(self):
        size = self.settings.model_size
        if self._transcriber is None or self._transcriber.model_size != size:
            self.emit("status", f"Whisper {size} 読み込み中...")
            self._transcriber = Transcriber(
                size,
                language=self.settings.language,
                beam_size=self.settings.profile()["beam_size"],
                condition_on_previous_text=True,  # ファイルは連続性を優先
            )
            self._transcriber.load()
        else:
            self._transcriber.language = self.settings.language
            self._transcriber.beam_size = self.settings.profile()["beam_size"]

    def transcribe(self, filepath: str) -> "tuple[int, bool, str]":
        import soundfile as sf

        basename = os.path.splitext(os.path.basename(filepath))[0]
        self.emit("status", f"読み込み中: {os.path.basename(filepath)}")
        self.emit("detail", "ファイル解析: メディア情報を取得中")

        try:
            duration = sf.info(filepath).duration
        except Exception as e:
            _log(f"file_transcribe sf.info failed: {e}")
            duration = probe_media_duration_ffprobe(filepath)

        dur_str = format_media_duration(duration)
        self.emit("status", f"文字起こし中: {basename} ({dur_str})")
        self.emit("detail", "ファイル処理: モデルを準備中")

        self._ensure_transcriber()
        if self.cancel.is_set():
            return 0, True, ""

        preprocess_result = None
        source = filepath
        fallback_path = None
        count = 0
        cancelled = False
        recording_start_at, timestamp_source = resolve_recording_start_at(filepath, duration)
        ts = datetime.now().strftime("%Y%m%d%H%M%S")
        out_path = os.path.join(SAVE_DIR, f"{ts}_{basename}.txt")

        try:
            if self.settings.preprocess.enabled():
                self.emit("detail", f"ファイル処理: 音声前処理 ({self.settings.preprocess.label()})")
                preprocess_result = preprocess_audio_file(filepath, self.settings.preprocess)
                source = preprocess_result.path

            _log(f"file_transcribe: source={filepath} dur={duration} model={self.settings.model_size}")
            self.emit("detail", "ファイル処理: VADありで文字起こし")
            info, segs = self._transcriber.iter_segments(source, vad_filter=True, strict=False)
            count, cancelled = self._write_segments(
                segs, out_path, os.path.basename(filepath), duration,
                (info.language if info else self.settings.language), basename,
                recording_start_at, timestamp_source, "VAD:on",
                preprocess_result.description if preprocess_result else None,
            )
            _log(f"file_transcribe: vad_filter=True → {count} segments")

            if count == 0 and not cancelled:
                _log("file_transcribe: VADあり0件 → VADなしで再試行")
                self.emit("detail", "ファイル処理: VADなしで再試行")
                info, segs = self._transcriber.iter_segments(source, vad_filter=False, strict=False)
                count, cancelled = self._write_segments(
                    segs, out_path, os.path.basename(filepath), duration,
                    (info.language if info else self.settings.language), basename,
                    recording_start_at, timestamp_source, "VAD:off",
                    preprocess_result.description if preprocess_result else None,
                )
                _log(f"file_transcribe: vad_filter=False → {count} segments")
        except Exception as first_error:
            _log(f"file_transcribe direct decode error: {first_error}")
            self.emit("detail", "ファイル処理: 変換して再試行")
            fallback_input = source if source != filepath else filepath
            fallback_path = convert_media_to_temp_wav(fallback_input)
            info, segs = self._transcriber.iter_segments(fallback_path, vad_filter=True, strict=False)
            count, cancelled = self._write_segments(
                segs, out_path, os.path.basename(filepath), duration,
                (info.language if info else self.settings.language), basename,
                recording_start_at, timestamp_source, "tmp wav",
                preprocess_result.description if preprocess_result else None,
            )
        finally:
            cleanup_paths = []
            if preprocess_result is not None:
                cleanup_paths.extend(preprocess_result.cleanup_paths)
            if fallback_path:
                cleanup_paths.append(fallback_path)
            for cleanup_path in cleanup_paths:
                if not cleanup_path or not os.path.exists(cleanup_path):
                    continue
                try:
                    os.unlink(cleanup_path)
                except OSError as e:
                    _log(f"file_transcribe cleanup failed: {e}")

        if self.cancel.is_set():
            cancelled = True
            interrupted_path = os.path.join(SAVE_DIR, f"{ts}_{basename}_中断.txt")
            if os.path.exists(out_path):
                os.replace(out_path, interrupted_path)
                out_path = interrupted_path

        if cancelled:
            self.emit("notify", "Talksribe", "文字起こし中断", f"{basename}: {count}件まで保存 → {out_path}")
            self.emit("status", f"中断: {basename} ({count}件) → {os.path.basename(out_path)}")
            self.emit("detail", "ファイル処理: 中断して保存")
        else:
            self.emit("notify", "Talksribe", "文字起こし完了", f"{basename}: {count}件 → {out_path}")
            self.emit("status", f"完了: {basename} ({count}件) → {os.path.basename(out_path)}")
            self.emit("detail", "ファイル処理: 保存完了")

        return count, cancelled, out_path

    def _write_segments(
        self, segs, out_path, source_name, duration, language, basename,
        recording_start_at, timestamp_source, progress_label, preprocess_label,
    ) -> "tuple[int, bool]":
        count = 0
        last_progress_sec = 0.0
        last_ratio = 0.0
        cancelled = False

        with open(out_path, "w", encoding="utf-8") as f:
            header_time = recording_start_at or datetime.now()
            f.write(f"# Talksribe 録音 {header_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            header = (
                f"# 元ファイル: {source_name}  長さ: {format_media_duration(duration)}  "
                f"言語: {language or self.settings.language}  モデル: {self.settings.model_size}"
            )
            if recording_start_at:
                header += f"  基準時刻: {recording_start_at.strftime('%Y-%m-%d %H:%M:%S')} ({timestamp_source})"
            if preprocess_label:
                header += f"  前処理: {preprocess_label}"
            f.write(header + "\n\n")

            for seg in segs:
                if self.cancel.is_set():
                    cancelled = True
                    break
                text = seg.text.strip()
                if not text:
                    continue
                count += 1
                line_ts = format_transcript_line_ts(seg.start, recording_start_at)
                f.write(f"[{line_ts}] {text}\n")

                should_report = False
                if seg.end - last_progress_sec >= FILE_PROGRESS_STEP_SEC:
                    should_report = True
                if duration:
                    ratio = min(1.0, seg.end / duration)
                    if ratio - last_ratio >= FILE_PROGRESS_STEP_RATIO:
                        should_report = True
                else:
                    ratio = None

                if should_report:
                    last_progress_sec = seg.end
                    if ratio is not None:
                        last_ratio = ratio
                        pct = int(ratio * 100)
                        self.emit(
                            "status",
                            f"文字起こし中: {basename} {pct}% "
                            f"({format_ts(seg.end)}/{format_media_duration(duration)})",
                        )
                    else:
                        self.emit("status", f"文字起こし中: {basename} {format_ts(seg.end)}")
                    self.emit("detail", f"ファイル処理: {progress_label} / {count}件")

        return count, cancelled
