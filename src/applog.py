#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""アプリ共通のログ出力。

親プロセスは RotatingFileHandler で回転しながら書き、子プロセス
(multiprocessing の worker)は追記だけを行う(回転は親だけが担当する)。
"""
from __future__ import annotations

import logging
import logging.handlers
import multiprocessing
import os
from datetime import datetime
from typing import Optional

LOG_DIR = os.path.expanduser("~/Documents/Talksribe")
LOG_PATH = os.path.join(LOG_DIR, "debug.log")
MAX_BYTES = 5 * 1024 * 1024
BACKUP_COUNT = 3

_logger: Optional[logging.Logger] = None


def _reset_logger() -> None:
    """ハンドラを閉じて作り直せる状態に戻す(テスト用)。"""
    global _logger
    lg = logging.getLogger("talksribe")
    for h in list(lg.handlers):
        lg.removeHandler(h)
        try:
            h.close()
        except Exception:
            pass
    _logger = None


def _get_logger() -> logging.Logger:
    """親プロセス用の logger を 1 回だけ作る。LOG_PATH などは作成時に読む。"""
    global _logger
    if _logger is None:
        os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
        lg = logging.getLogger("talksribe")
        lg.setLevel(logging.INFO)
        lg.propagate = False
        handler = logging.handlers.RotatingFileHandler(
            LOG_PATH, maxBytes=MAX_BYTES, backupCount=BACKUP_COUNT, encoding="utf-8"
        )
        handler.setFormatter(
            logging.Formatter("%(asctime)s [%(process)d] %(message)s", "%Y-%m-%d %H:%M:%S")
        )
        lg.addHandler(handler)
        _logger = lg
    return _logger


def log(msg: str) -> None:
    """ログを 1 行書く。例外は外に出さない。"""
    try:
        if multiprocessing.parent_process() is None:
            _get_logger().info(msg)
        else:
            os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
            with open(LOG_PATH, "a", encoding="utf-8") as f:
                f.write(f"{datetime.now():%Y-%m-%d %H:%M:%S} [{os.getpid()}] {msg}\n")
    except Exception:
        pass
