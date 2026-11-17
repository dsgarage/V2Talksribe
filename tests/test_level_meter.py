# -*- coding: utf-8 -*-
"""入力レベルメーターのテスト。マイクも本物の保存先も使わない。"""
import unittest

import numpy as np

import level_meter as lm
from level_meter import LevelMeter, level_char, window_levels
from tests.fakes import collect_events
from tests.test_engine_recovery import EngineTestBase

SR = 16000


def rms_of_db(db):
    return 10 ** (db / 20.0)


class TestLevelChar(unittest.TestCase):
    def test_silent_and_quiet(self):
        self.assertEqual(level_char(0.0, 0.0), lm.QUIET)
        self.assertEqual(level_char(rms_of_db(-70), 0.01), lm.QUIET)

    def test_floor_and_top(self):
        self.assertEqual(level_char(rms_of_db(-60), 0.01), lm.BARS[0])
        self.assertEqual(level_char(rms_of_db(-20), 0.5), lm.BARS[-1])
        self.assertEqual(level_char(rms_of_db(-10), 0.5), lm.BARS[-1])

    def test_middle(self):
        expected = lm.BARS[int((-40 - lm.FLOOR_DB) / (lm.TOP_DB - lm.FLOOR_DB) * 8)]
        self.assertIn(expected, "▄▅")
        self.assertEqual(level_char(rms_of_db(-40), 0.1), expected)


class TestWindowLevels(unittest.TestCase):
    def count(self, sec):
        return len(window_levels(np.full(int(SR * sec), 0.1, np.float32), SR))

    def test_counts(self):
        self.assertEqual(self.count(1.0), 4)
        self.assertEqual(self.count(0.5), 2)
        self.assertEqual(self.count(0.6), 2)
        self.assertEqual(self.count(0.65), 3)
        self.assertEqual(window_levels(np.zeros(0, np.float32), SR), [])

    def test_values(self):
        (rms, peak), = window_levels(np.full(SR // 4, 0.1, np.float32), SR)
        self.assertAlmostEqual(rms, 0.1, places=5)
        self.assertAlmostEqual(peak, 0.1, places=5)


class TestLevelMeter(unittest.TestCase):
    LOUD = (rms_of_db(-30), 0.2)

    def test_padding(self):
        m = LevelMeter()
        m.push([self.LOUD, self.LOUD], 0.0)
        out = m.render(0.0)
        self.assertEqual(len(out), 5)
        self.assertEqual(out[:3], lm.QUIET * 3)
        self.assertNotIn(lm.QUIET, out[3:])

    def test_keeps_last_five(self):
        m = LevelMeter()
        m.push([(rms_of_db(-55), 0.01)] * 2 + [self.LOUD] * 5, 0.0)
        out = m.render(0.0)
        self.assertEqual(out, level_char(*self.LOUD) * 5)

    def test_clip_hold(self):
        m = LevelMeter()
        m.push([(0.7, 1.0)], 10.0)
        self.assertTrue(m.render(10.5).endswith("!"))
        self.assertTrue(m.render(11.9).endswith("!"))
        self.assertFalse(m.render(12.1).endswith("!"))

    def test_reset(self):
        m = LevelMeter()
        m.push([(0.7, 1.0)], 0.0)
        m.reset()
        self.assertEqual(m.render(0.0), lm.QUIET * 5)


class TestEngineLevelEvent(EngineTestBase):
    def test_level_event(self):
        eng = self.make_engine()
        self.start_recording(eng)
        self.factory.last.feed(np.full(SR // 2, 0.05, np.float32))
        evs = collect_events(eng, lambda e: e[0] == "level", timeout=5.0)
        lv = [e for e in evs if e[0] == "level"]
        self.assertTrue(lv, "level イベントが届かない")
        self.assertEqual(len(lv[0][1]), 2)
        for item in lv[0][1]:
            self.assertEqual(len(item), 2)


if __name__ == "__main__":
    unittest.main()
