# -*- coding: utf-8 -*-
"""エンジンの回復・読み込み・通知のテスト。マイクも本物の保存先も使わない。"""
import os
import shutil
import tempfile
import threading
import time
import unittest
from unittest import mock

import numpy as np

import applog
import engine as engine_module
from engine import TranscriptionEngine
from settings import AppSettings
from tests.fakes import (FakeCaptureFactory, FakeTranscriber, FakeVAD, collect_events)


class AlwaysRaisingVAD(FakeVAD):
    def is_speech(self, audio):
        raise RuntimeError("fake vad error (always)")


LOUD = np.full(8000, 0.05, np.float32)
QUIET = np.zeros(8000, np.float32)


class EngineTestBase(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="talksribe_test_")
        self.engines = []
        self.seen = []
        # ログも本物の debug.log に書かず、一時フォルダに向ける
        applog._reset_logger()
        self._log_patches = [
            mock.patch.object(applog, "LOG_DIR", self.tmpdir),
            mock.patch.object(applog, "LOG_PATH", os.path.join(self.tmpdir, "debug.log")),
        ]
        for p in self._log_patches:
            p.start()

    def tearDown(self):
        for eng in self.engines:
            eng.shutdown()
            if eng._worker is not None:
                eng._worker.join(3)
        applog._reset_logger()
        for p in self._log_patches:
            p.stop()
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def patch_const(self, name, value):
        p = mock.patch.object(engine_module, name, value)
        p.start()
        self.addCleanup(p.stop)

    def make_engine(self, *, factory=None, transcriber_factory=None, vad_factory=None,
                    device_name=None, run=True, **kw):
        settings = AppSettings()
        settings.save = lambda *a, **k: None   # 本物の config.json に書かない
        settings.input_device_name = device_name
        self.factory = factory or FakeCaptureFactory()
        kw.setdefault("record_audio", False)
        kw.setdefault("prune_audio", False)
        eng = TranscriptionEngine(
            settings, capture_factory=self.factory,
            transcriber_factory=transcriber_factory or (lambda s, l, b: FakeTranscriber(s, l, b)),
            vad_factory=vad_factory or (lambda b, sr, agg: FakeVAD(b, sr, agg)),
            save_dir=self.tmpdir, **kw)
        self.engines.append(eng)
        if run:
            eng.run()
        return eng

    def wait_event(self, eng, pred, timeout=5.0):
        """これまでに読んだイベントと新しいイベントから pred に合うものを探す。"""
        if any(pred(e) for e in self.seen):
            return True
        evs = collect_events(eng, pred, timeout=timeout)
        self.seen.extend(evs)
        return any(pred(e) for e in evs)

    def wait_until(self, cond, timeout=5.0):
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            if cond():
                return True
            time.sleep(0.05)
        return cond()

    def has_alert(self, text):
        return any(e[0] == "alert" and text in e[1] for e in self.seen)

    def alert_pred(self, text):
        return lambda e: e[0] == "alert" and text in e[1]

    def start_recording(self, eng):
        eng.start()
        self.assertTrue(self.wait_event(eng, lambda e: e == ("title", "REC")), "REC にならない")


class TestEngineRecovery(EngineTestBase):
    def test_01_vad_exception_does_not_kill_worker(self):
        eng = self.make_engine(vad_factory=lambda b, sr, agg: FakeVAD(b, sr, agg, raise_once=True))
        self.start_recording(eng)
        self.factory.last.feed(LOUD)
        self.assertTrue(self.wait_event(eng, self.alert_pred("録音を再開します")))
        self.assertTrue(self.wait_until(lambda: len(self.factory.calls) == 2))
        self.assertTrue(eng.worker_alive())
        self.assertTrue(eng.is_recording)

    def test_02_stops_after_repeated_errors(self):
        eng = self.make_engine(vad_factory=lambda b, sr, agg: AlwaysRaisingVAD(b, sr, agg))
        self.start_recording(eng)
        stopped = lambda e: e[0] == "status" and e[1].startswith("停止:")
        end = time.monotonic() + 8
        while time.monotonic() < end and not any(stopped(e) for e in self.seen):
            try:
                self.factory.last.feed(LOUD)
            except OSError:
                pass
            self.seen.extend(collect_events(eng, stopped, timeout=0.1))
        self.assertTrue(any(stopped(e) for e in self.seen))
        self.assertTrue(self.wait_until(lambda: not eng.is_recording))
        self.assertTrue(eng.worker_alive())

    def test_03_eof_does_not_spin(self):
        self.patch_const("CAPTURE_BACKOFF_SEC", (0.2,) * 5)
        eng = self.make_engine()
        self.start_recording(eng)
        before = len(self.factory.calls)
        self.factory.last.die()
        time.sleep(1.0)
        self.assertLessEqual(len(self.factory.calls) - before, 2)
        self.assertTrue(self.wait_until(lambda: len(self.factory.calls) > before))
        self.assertTrue(eng.is_recording)

    def test_04_fall_back_to_default_then_stop(self):
        self.patch_const("CAPTURE_BACKOFF_SEC", (0.01,) * 5)
        factory = FakeCaptureFactory([Exception("x")] * 20)
        eng = self.make_engine(factory=factory, device_name="BlackHole 2ch")
        eng.start()
        stopped = lambda e: e[0] == "status" and e[1].startswith("停止: 入力に接続できません")
        self.assertTrue(self.wait_event(eng, stopped, timeout=9.0))
        names = [c["device_name"] for c in factory.calls]
        self.assertEqual(names[:6], ["BlackHole 2ch"] * 6)
        self.assertTrue(all(n is None for n in names[6:]))
        self.assertGreater(len(names), 6)
        self.assertTrue(self.has_alert("既定"))

    def test_05_stall_reconnects(self):
        self.patch_const("CAPTURE_STALL_SEC", 0.3)
        self.patch_const("CAPTURE_BACKOFF_SEC", (0.05,) * 5)
        eng = self.make_engine()
        self.start_recording(eng)
        self.assertTrue(self.wait_until(lambda: len(self.factory.calls) == 2))
        self.assertTrue(self.wait_event(eng, self.alert_pred("止まりました")))

    def test_06_silent_input_notified_once(self):
        self.patch_const("SILENT_INPUT_NOTIFY_SEC", 0.5)
        eng = self.make_engine()
        self.start_recording(eng)
        cap = self.factory.last
        for _ in range(10):
            cap.feed(QUIET)
            time.sleep(0.1)
        self.assertTrue(self.wait_event(eng, self.alert_pred("まったく無音")))
        time.sleep(0.3)
        self.seen.extend(collect_events(eng, lambda e: False, timeout=0.3))
        silent = [e for e in self.seen if e[0] == "alert" and "まったく無音" in e[1]]
        self.assertEqual(len(silent), 1)
        self.assertIn("Default Mic", silent[0][1])

    def test_07_drop_notified(self):
        eng = self.make_engine()
        self.start_recording(eng)
        self.factory.last.send_ctrl({"type": "stats", "dropped_live_chunks": 3,
                                     "dropped_record_chunks": 0, "status_flags": 0})
        self.assertTrue(self.wait_event(eng, self.alert_pred("文字起こしできませんでした")))

    def test_08_cancel_loading(self):
        block = threading.Event()
        self.addCleanup(block.set)
        eng = self.make_engine(
            transcriber_factory=lambda s, l, b: FakeTranscriber(s, l, b, block=block))
        eng.toggle()
        self.assertTrue(self.wait_event(eng, lambda e: e == ("toggle", "読み込みを取り消す")))
        eng.toggle()
        self.assertTrue(self.wait_event(
            eng, lambda e: e == ("status", "読み込みを取り消しました"), timeout=1.0))
        self.assertTrue(eng.worker_alive())
        self.assertFalse(eng.is_loading)

    def test_09_load_timeout(self):
        self.patch_const("LOAD_TIMEOUT_SEC", 0.3)
        block = threading.Event()
        self.addCleanup(block.set)
        eng = self.make_engine(
            transcriber_factory=lambda s, l, b: FakeTranscriber(s, l, b, block=block))
        eng.toggle()
        self.assertTrue(self.wait_event(
            eng, lambda e: e[0] == "status" and e[1].startswith("時間切れ")))

    def test_10_silero_failure_falls_back_to_webrtc(self):
        eng = self.make_engine(
            vad_factory=lambda b, sr, agg: FakeVAD(b, sr, agg, load_error=(b == "silero")))
        eng.vad_backend = "silero"
        self.start_recording(eng)
        self.assertTrue(self.wait_event(eng, self.alert_pred("webrtc に切り替えました")))
        self.assertEqual(eng.vad.backend, "webrtc")

    def test_11_device_not_found_notified(self):
        eng = self.make_engine(factory=FakeCaptureFactory(["ok_nostart"]))
        self.start_recording(eng)
        self.factory.last.started(name="BlackHole 2ch", requested="iPhone", fell_back=True)
        self.assertTrue(self.wait_event(eng, self.alert_pred("'iPhone' が見つからないため")))

    def test_12_worker_restart(self):
        eng = self.make_engine()
        self.start_recording(eng)

        def boom():
            raise SystemExit  # try/except Exception では捕まらない

        eng._pump_audio = boom
        self.assertTrue(self.wait_until(lambda: not eng.worker_alive(), timeout=3.0))
        del eng._pump_audio
        calls_before = len(self.factory.calls)
        hb_before = eng.heartbeat_at
        self.assertTrue(eng.ensure_worker())
        self.assertTrue(self.wait_until(lambda: eng.heartbeat_at > hb_before))
        self.assertTrue(self.wait_until(lambda: len(self.factory.calls) > calls_before))
        self.assertTrue(eng.worker_alive())

    def test_13_prune_called_with_protect_prefix(self):
        audio_dir = os.path.join(self.tmpdir, "audio")
        # (a) 既定どおり整理する
        with mock.patch.object(engine_module, "prune_recordings", return_value=[]) as m:
            eng = self.make_engine(record_audio=True, prune_audio=True)
            self.start_recording(eng)
            self.assertEqual(m.call_count, 1)
            self.assertEqual(m.call_args.args[0], audio_dir)
            self.assertEqual(m.call_args.kwargs["protect_prefix"], eng._session_ts)
        # (b) prune_audio=False なら呼ばない
        with mock.patch.object(engine_module, "prune_recordings", return_value=[]) as m:
            eng = self.make_engine(record_audio=True, prune_audio=False)
            self.start_recording(eng)
            m.assert_not_called()
        # (c) 引数を省くと True
        settings = AppSettings()
        eng = TranscriptionEngine(settings, save_dir=self.tmpdir)
        self.assertTrue(eng.prune_audio)


if __name__ == "__main__":
    unittest.main()
