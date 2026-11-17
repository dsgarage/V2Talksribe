# -*- coding: utf-8 -*-
"""停止・開始の反応(#9)のテスト。マイクも本物の保存先も使わない。"""
import threading
import unittest

import numpy as np

from tests.fakes import FakeTranscriber, FakeVAD, FakeCaptureFactory
from tests.test_engine_recovery import EngineTestBase, LOUD


class GatedTranscriber(FakeTranscriber):
    """armed が立ったあとの collect_texts を gate が開くまで止める。"""

    def __init__(self, *a, gate=None, **kw):
        super().__init__(*a, **kw)
        self.gate = gate
        self.armed = False
        self.entered = threading.Event()

    def collect_texts(self, audio, emit_until_sec=None, **kw):
        if self.armed:
            self.entered.set()
            self.gate.wait()
        return super().collect_texts(audio, emit_until_sec=emit_until_sec, **kw)


class TestEngineResponsiveness(EngineTestBase):
    def test_01_stop_display_does_not_wait_for_finalize(self):
        gate = threading.Event()
        self.addCleanup(gate.set)
        made = []

        def tf(s, l, b):
            t = GatedTranscriber(s, l, b, gate=gate)
            made.append(t)
            return t

        eng = self.make_engine(transcriber_factory=tf)
        eng.partial_enabled = False   # 部分更新で止まらないようにする
        self.start_recording(eng)
        cap = self.factory.last
        for _ in range(4):            # 発話 2 秒ぶん。無音は流さない
            cap.feed(LOUD)
        made[0].armed = True
        eng.stop()
        # ブロックを解かないまま表示が戻る
        self.assertTrue(self.wait_event(eng, lambda e: e == ("toggle", "録音開始"), timeout=1.5))
        self.assertTrue(self.wait_event(eng, lambda e: e == ("title", "MIC"), timeout=1.5))
        self.assertTrue(made[0].entered.wait(1.5), "残りの確定が始まっていない")
        gate.set()
        self.assertTrue(self.wait_event(eng, lambda e: e[0] == "transcript", timeout=3.0))
        self.assertTrue(self.wait_event(
            eng, lambda e: e[0] == "status" and "保存済み" in e[1], timeout=3.0))
        statuses = [e for e in self.seen if e[0] == "status"]
        self.assertIn("保存済み", statuses[-1][1])

    def test_02_start_while_recording_returns_current_state(self):
        eng = self.make_engine()
        self.start_recording(eng)
        calls = len(self.factory.calls)
        eng.start()
        self.assertTrue(self.wait_event(eng, lambda e: e == ("toggle", "録音停止"), timeout=2.0))
        self.assertEqual(len(self.factory.calls), calls)

    def test_03_stop_while_idle_returns_current_state(self):
        eng = self.make_engine()
        eng.stop()
        self.assertTrue(self.wait_event(eng, lambda e: e == ("toggle", "録音開始"), timeout=2.0))

    def test_04_no_partial_while_command_waiting(self):
        eng = self.make_engine(run=False)
        t = FakeTranscriber()
        eng.transcriber = t
        eng.vad = FakeVAD()
        eng.is_recording = True
        eng.partial_enabled = True
        eng.last_partial_at = 0
        eng._commands.put(("set_preprocess",))
        for _ in range(4):
            eng._process_audio(LOUD)
        self.assertEqual(t.calls, 0)
        while not eng._commands.empty():
            eng._commands.get_nowait()
        eng.speech_buf = np.zeros(0, dtype=np.float32)
        eng.last_partial_at = 0
        for _ in range(4):
            eng._process_audio(LOUD)
        self.assertGreaterEqual(t.calls, 1)


if __name__ == "__main__":
    unittest.main()
