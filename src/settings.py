#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""アプリ設定と永続化。

言語・モデル・電力プロファイル・入力デバイス・音声前処理を 1 つの
``AppSettings`` にまとめ、``~/Documents/Talksribe/config.json`` へ保存/復元する。
毎回デフォルト (eco / base / en) に戻ってしまう問題を解消する。
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field, fields
from typing import Optional

from audio_preprocess import AudioPreprocessSettings
from devices import normalize_device_name

SAVE_DIR = os.path.expanduser("~/Documents/Talksribe")
CONFIG_PATH = os.path.join(SAVE_DIR, "config.json")

DEFAULT_MODEL_SIZE = "base"
DEFAULT_POWER_PROFILE = "eco"
# 既定言語は「自動検出」(None)。en 固定による英訳幻聴を避ける。
DEFAULT_LANGUAGE: Optional[str] = None

# 電力プロファイル。model/vad/chunk などライブ処理のパラメータ束。
# ここを唯一の定義元にして engine / UI で共有する。
POWER_PROFILES = {
    "eco": {
        "label": "省電力",
        "model_size": "tiny",
        "beam_size": 1,
        "chunk_sec": 1.0,
        "vad_backend": "webrtc",
        "vad_aggressiveness": 2,
        "partial_enabled": False,
        "partial_interval_sec": 8.0,
        "partial_window_sec": 3.0,
        "idle_sleep_sec": 2.5,
        "status_every": 8,
    },
    "balanced": {
        "label": "標準",
        "model_size": "base",
        "beam_size": 1,
        "chunk_sec": 0.75,
        "vad_backend": "webrtc",
        "vad_aggressiveness": 2,
        "partial_enabled": True,
        "partial_interval_sec": 6.0,
        "partial_window_sec": 3.0,
        "idle_sleep_sec": 1.8,
        "status_every": 6,
    },
    "accuracy": {
        "label": "高精度",
        "model_size": "small",
        "beam_size": 2,
        "chunk_sec": 0.5,
        "vad_backend": "silero",
        "vad_aggressiveness": 2,
        "partial_enabled": True,
        "partial_interval_sec": 4.0,
        "partial_window_sec": 4.0,
        "idle_sleep_sec": 1.5,
        "status_every": 4,
    },
}


@dataclass
class AppSettings:
    """アプリ全体の永続設定。

    ``model_size`` は電力プロファイルの既定値を上書きできる実効モデル。
    ユーザーがモデルメニューで個別に選んだ場合はこちらが優先される。
    """

    language: Optional[str] = DEFAULT_LANGUAGE
    model_size: str = DEFAULT_MODEL_SIZE
    power_profile: str = DEFAULT_POWER_PROFILE
    input_device_id: Optional[int] = None
    input_device_name: Optional[str] = None
    preprocess: AudioPreprocessSettings = field(default_factory=AudioPreprocessSettings)

    def profile(self) -> dict:
        return POWER_PROFILES.get(self.power_profile, POWER_PROFILES[DEFAULT_POWER_PROFILE])

    def to_dict(self) -> dict:
        return asdict(self)

    def migrate_input_device_name(self, index_to_name: dict) -> bool:
        """旧版の番号指定を名前指定へ移す。移したら True。"""
        if self.input_device_name is not None or self.input_device_id is None:
            return False
        name = index_to_name.get(self.input_device_id)
        self.input_device_name = normalize_device_name(name) if name else None
        self.input_device_id = None
        return True

    @classmethod
    def from_dict(cls, data) -> "AppSettings":
        settings = cls()
        if not isinstance(data, dict):
            return settings

        for key in ("language", "model_size", "power_profile", "input_device_id", "input_device_name"):
            if key in data:
                setattr(settings, key, data[key])

        # 未知のプロファイル名が入っていたら既定へ戻す
        if settings.power_profile not in POWER_PROFILES:
            settings.power_profile = DEFAULT_POWER_PROFILE

        pre_data = data.get("preprocess")
        if isinstance(pre_data, dict):
            valid = {f.name for f in fields(AudioPreprocessSettings)}
            pre = AudioPreprocessSettings()
            for k, v in pre_data.items():
                if k in valid:
                    setattr(pre, k, v)
            settings.preprocess = pre

        return settings

    def save(self, path: str = CONFIG_PATH) -> None:
        """壊れた書き込みを避けるため一時ファイル経由で保存する。"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = f"{path}.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)

    @classmethod
    def load(cls, path: str = CONFIG_PATH) -> "AppSettings":
        """設定を読み込む。欠損・破損時は既定値にフォールバックする。"""
        try:
            with open(path, encoding="utf-8") as f:
                return cls.from_dict(json.load(f))
        except (FileNotFoundError, json.JSONDecodeError, OSError, ValueError):
            return cls()
