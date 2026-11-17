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
class LiveTuning:
    """ライブ経路の切り方・関門・幻聴対策・速さの値。初期値は計測(U7)で動かす。"""

    # --- 切り方 ---
    preroll_sec: float = 0.4            # 範囲 0.25〜0.6。発話の前に含める音
    silence_end_sec: float = 1.2        # 範囲 0.8〜1.6。この無音が続いたら確定
    short_utt_sec: float = 2.0          # 範囲 1.0〜3.0。これより短い発話は…
    short_utt_silence_sec: float = 1.6  # 範囲 1.2〜2.4。…この無音まで待って次の声とつなげる
    # --- 関門(床に合わせて動く) ---
    gate_margin_db: float = 6.0         # 範囲 3〜10。床 + この値 以上のフレームだけ VAD に回す
    floor_init_db: float = -60.0        # 床の初期値(固定)
    floor_rise_db_per_sec: float = 0.3  # 範囲 0.1〜1.0。床が上がる速さ(下がるのは即座)
    # --- VAD ---
    vad_on_agc: bool = True             # VAD に利得を掛けた音を渡す
    agc_enabled: bool = True            # VAD 用の利得を動かす
    agc_target_db: float = -26.0        # 範囲 -30〜-20
    agc_max_gain_db: float = 24.0       # 範囲 12〜30
    agc_tau_sec: float = 3.0            # 声の大きさの推定の時定数(固定)
    agc_slew_db_per_sec: float = 6.0    # 利得の変化の速さの上限(固定)
    silero_enter: float = 0.5           # 範囲 0.4〜0.6。発話に入る確率
    silero_stay: float = 0.35           # 範囲 0.25〜0.45。発話の中で続けるのに要る確率
    webrtc_enter: float = 0.6           # 範囲 0.5〜0.8(投票率)
    webrtc_stay: float = 0.4            # 範囲 0.3〜0.6
    # --- 区間の受け入れ条件(すべて満たしたら Whisper に回す) ---
    min_voiced_sec: float = 0.5         # 範囲 0.3〜0.8。発話と判定したフレームの合計
    min_voiced_ratio: float = 0.3       # 範囲 0.2〜0.5。発話フレーム / 区間の中身
    min_snr_db: float = 6.0             # 範囲 3〜10。声のフレームの RMS 中央値 − 床
    utt_min_peak_prob: float = 0.8      # 範囲 0.6〜0.9。区間の中の silero の確率の最大
    # --- Whisper に渡す音 ---
    asr_normalize: bool = True
    asr_target_db: float = -23.0        # 範囲 -26〜-18
    asr_max_gain_db: float = 24.0       # 範囲 12〜30
    # --- 幻聴 ---
    halluc_filter: bool = True
    weak_voiced_sec: float = 2.0        # 範囲 1.0〜3.0。これ未満の区間は「弱い」
    weak_no_speech_prob: float = 0.3    # 範囲 0.2〜0.5
    weak_avg_logprob: float = -0.5      # 範囲 -0.7〜-0.3
    repeat_min: int = 3                 # 範囲 2〜4。同じ語句がこの回数以上続いたら 1 つにまとめる
    dup_window_sec: float = 10.0        # 範囲 5〜30。直前と同じ文が弱い区間で出たら捨てる
    live_no_speech_threshold: float = 0.5   # 範囲 0.4〜0.6
    live_log_prob_threshold: float = -1.0   # 範囲 -1.0〜-0.6
    # --- 速さ ---
    temperature: tuple = (0.0,)         # 候補 (0.0,) / (0.0, 0.4) / Whisper 既定の 6 段
    tokens_base: int = 16               # max_new_tokens = min(tokens_cap, tokens_base + tokens_per_sec * 音声秒)
    tokens_per_sec: float = 15.0        # 範囲 10〜20
    tokens_cap: int = 220
    partial_min_speech_sec: float = 1.5
    partial_cost_factor: float = 4.0    # 範囲 2〜8。次の部分更新まで、前の呼び出し時間のこの倍は空ける
    partial_max_backlog_sec: float = 1.0  # 範囲 0.5〜2.0。処理の遅れがこれを超えたら部分更新を止める
    partial_beam: int = 1
    # --- 文脈 ---
    context_chars: int = 0              # 候補 0 / 60 / 120
    context_max_gap_sec: float = 10.0   # 前の確定からこの秒以内のときだけ文脈を渡す


def _coerce_tuning_value(default, v):
    """既定値の型にそろえる。変換できなければ例外。"""
    if isinstance(default, bool):
        return bool(v)
    if isinstance(default, int):
        return int(v)
    if isinstance(default, float):
        return float(v)
    if isinstance(default, tuple):
        if not isinstance(v, (list, tuple)):
            raise ValueError("list/tuple が必要です")
        return tuple(float(x) for x in v)
    return v


_TRUE_WORDS = ("true", "1", "on")
_FALSE_WORDS = ("false", "0", "off")


def apply_tuning_overrides(t: LiveTuning, pairs: list) -> LiveTuning:
    """["silence_end_sec=1.4", "asr_normalize=false", "temperature=0.0,0.4"] を t に当てる。

    未知のキー・変換できない値は ValueError(メッセージにキー名)。bool は true/false/1/0/on/off。
    tuple はカンマ区切り。
    """
    valid = {f.name for f in fields(LiveTuning)}
    for pair in pairs or []:
        if "=" not in pair:
            raise ValueError(f"KEY=VALUE の形で指定してください: {pair}")
        key, raw = pair.split("=", 1)
        key = key.strip()
        raw = raw.strip()
        if key not in valid:
            raise ValueError(f"未知のキー: {key}")
        default = getattr(t, key)
        try:
            if isinstance(default, bool):
                low = raw.lower()
                if low in _TRUE_WORDS:
                    val = True
                elif low in _FALSE_WORDS:
                    val = False
                else:
                    raise ValueError("true/false/1/0/on/off のいずれか")
            elif isinstance(default, tuple):
                val = tuple(float(x) for x in raw.split(",") if x.strip() != "")
                if not val:
                    raise ValueError("値が空です")
            elif isinstance(default, int):
                val = int(raw)
            elif isinstance(default, float):
                val = float(raw)
            else:
                val = raw
        except ValueError as e:
            raise ValueError(f"{key} の値を変換できません: {raw!r} ({e})")
        setattr(t, key, val)
    return t


VOCAB_PATH = os.path.join(SAVE_DIR, "vocabulary.txt")
VOCAB_HEADER = "# 1 行に 1 語。固有名詞・専門用語を書くと認識の手がかりになります\n"


def load_vocabulary(path: str = VOCAB_PATH, max_terms: int = 50) -> list:
    """1 行に 1 語。# で始まる行と空行は飛ばす。前後の空白を落とし、重複は最初のものだけ残す。
    ファイルが無い・読めないときは []。max_terms を超えた分は捨てる。"""
    try:
        with open(path, encoding="utf-8") as f:
            lines = f.read().splitlines()
    except (OSError, UnicodeDecodeError):
        return []
    out: list = []
    for line in lines:
        w = line.strip()
        if not w or w.startswith("#") or w in out:
            continue
        out.append(w)
        if len(out) >= max_terms:
            break
    return out


def hotwords_from_vocabulary(terms: list) -> Optional[str]:
    return "、".join(terms) if terms else None


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
    live_tuning: LiveTuning = field(default_factory=LiveTuning)

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

        tune_data = data.get("live_tuning")
        if isinstance(tune_data, dict):
            tuning = LiveTuning()
            for f in fields(LiveTuning):
                if f.name in tune_data:
                    try:
                        setattr(tuning, f.name,
                                _coerce_tuning_value(getattr(tuning, f.name), tune_data[f.name]))
                    except (TypeError, ValueError):
                        pass
            settings.live_tuning = tuning

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
