# -*- coding: utf-8 -*-
"""replay_live.py の offline 処理のテスト。マイクも本物のモデルも本物の保存先も使わない。"""
import importlib.util
import json
import os
import shutil
import tempfile
import unittest

import numpy as np
import soundfile as sf

from settings import AppSettings
from tests.fakes import FakeTranscriber

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_replay():
    spec = importlib.util.spec_from_file_location(
        "replay_live", os.path.join(ROOT, "scripts", "replay_live.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _snapshot(path):
    out = set()
    for d, _, files in os.walk(path):
        for f in files:
            out.add(os.path.join(d, f))
    return out


class ReplayOfflineTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="talksribe_replay_")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_run_offline(self):
        replay = _load_replay()
        sr = 16000
        rng = np.random.default_rng(0)
        t = np.arange(2 * sr) / sr
        audio = np.concatenate([
            rng.normal(0, 0.001, 2 * sr),
            0.3 * np.sin(2 * np.pi * 300 * t),
            rng.normal(0, 0.001, 2 * sr),
        ]).astype(np.float32)
        wav = os.path.join(self.tmp, "in.wav")
        sf.write(wav, audio, sr)
        out = os.path.join(self.tmp, "out")

        real_dir = os.path.expanduser("~/Documents/Talksribe")
        before = _snapshot(real_dir)

        settings = AppSettings()
        settings.save = lambda *a, **k: None
        settings.power_profile = "balanced"
        timed = replay.TimedTranscriber(FakeTranscriber(texts=["テスト"]))
        m = replay.run_offline(wav, settings, timed, out)

        for k in ("audio_sec", "proc_sec", "rtf", "calls", "call_sec_p95"):
            self.assertIn(k, m)
        self.assertAlmostEqual(m["audio_sec"], 6.0, delta=0.1)
        self.assertGreaterEqual(m["calls"], 1)
        with open(os.path.join(out, "transcript.txt"), encoding="utf-8") as f:
            self.assertTrue(any("テスト" in line for line in f))
        with open(os.path.join(out, "metrics.json"), encoding="utf-8") as f:
            self.assertEqual(json.load(f)["calls"], m["calls"])
        self.assertEqual(_snapshot(real_dir) - before, set())
