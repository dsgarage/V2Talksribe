#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""入力デバイスの解決と一覧取得。

デバイス番号は着脱で変わるため、名前で保存して起動のたびに番号へ解決する。
"""
from __future__ import annotations

import json
import subprocess
import sys
import unicodedata
from typing import Optional

import applog

_LIST_SCRIPT = (
    "import json, sounddevice as sd\n"
    "d = sd.default.device[0]\n"
    "out = [{'index': i, 'name': x['name'], 'is_default': i == d}\n"
    "       for i, x in enumerate(sd.query_devices()) if x['max_input_channels'] > 0]\n"
    "print(json.dumps(out))\n"
)


def normalize_device_name(name: str) -> str:
    """NFC にそろえ、方向制御文字 (U+200E / U+200F) と前後の空白を除く。"""
    s = unicodedata.normalize("NFC", name or "")
    s = s.replace("\u200e", "").replace("\u200f", "")
    return s.strip()


def _default_usable(devices: list, default_index: Optional[int]):
    if default_index is None or default_index < 0:
        return None
    for d in devices:
        if d["index"] == default_index and d["max_input_channels"] > 0:
            return d
    return None


def resolve_input_device(devices: list, wanted: Optional[str],
                         default_index: Optional[int]
                         ) -> "tuple[Optional[int], Optional[str], bool]":
    """(index, name, fell_back) を返す。使える入力が無ければ (None, None, True)。"""
    default = _default_usable(devices, default_index)
    if wanted is None:
        if default is None:
            return None, None, True
        return default["index"], default["name"], False

    key = normalize_device_name(wanted).casefold()
    inputs = [d for d in devices if d["max_input_channels"] > 0]
    for d in inputs:
        if normalize_device_name(d["name"]).casefold() == key:
            return d["index"], d["name"], False
    for d in inputs:
        if key and key in normalize_device_name(d["name"]).casefold():
            return d["index"], d["name"], False

    if default is None:
        return None, None, True
    return default["index"], default["name"], True


def query_input_devices_inprocess() -> "tuple[list, Optional[int]]":
    """このプロセスの PortAudio から (デバイス一覧, 既定の入力番号) を取る。"""
    import sounddevice as sd

    devices = [
        {"index": i, "name": d["name"], "max_input_channels": d["max_input_channels"]}
        for i, d in enumerate(sd.query_devices())
    ]
    default_in = sd.default.device[0]
    return devices, (default_in if default_in is not None else None)


def list_input_devices(timeout: float = 10.0) -> list:
    """入力のあるデバイスを [{"index","name","is_default"}] で返す。失敗時は空。"""
    try:
        if getattr(sys, "frozen", False):
            devices, default_in = query_input_devices_inprocess()
            return [
                {"index": d["index"], "name": d["name"], "is_default": d["index"] == default_in}
                for d in devices if d["max_input_channels"] > 0
            ]
        # PortAudio はデバイス一覧をプロセスごとに覚えたままなので、別プロセスで新しい一覧を取る
        r = subprocess.run([sys.executable, "-c", _LIST_SCRIPT],
                           capture_output=True, text=True, timeout=timeout)
        if r.returncode != 0:
            applog.log(f"list_input_devices failed rc={r.returncode}: {r.stderr.strip()[-200:]}")
            return []
        return json.loads(r.stdout.strip().splitlines()[-1])
    except Exception as e:
        applog.log(f"list_input_devices failed: {e}")
        return []
