# -*- coding: utf-8 -*-
import unittest

import numpy as np

from live_level import (FRAME_SAMPLES, LevelTracker, db_to_gain, frame_rms_db, normalize_for_asr,
                        soft_limit)
from settings import LiveTuning

SR = 16000


def sine(db, n=FRAME_SAMPLES, f=300):
    amp = db_to_gain(db) * np.sqrt(2.0)
    t = np.arange(n) / SR
    return (amp * np.sin(2 * np.pi * f * t)).astype(np.float32)


class LiveLevelTest(unittest.TestCase):
    def test_1_soft_limit(self):
        x = np.array([-2, -0.95, -0.5, 0, 0.5, 0.95, 2], np.float32)
        y = soft_limit(x)
        self.assertTrue(np.all(np.abs(y) < 1.0))
        self.assertEqual(float(y[2]), -0.5)
        self.assertEqual(float(y[4]), 0.5)
        self.assertTrue(np.all(np.sign(y) == np.sign(x)))
        self.assertTrue(np.all(np.diff(y) > 0))

    def test_2_floor(self):
        rng = np.random.default_rng(0)
        tr = LevelTracker(SR, LiveTuning())
        for _ in range(20):
            tr.update(rng.normal(0, 0.001, FRAME_SAMPLES).astype(np.float32))
        self.assertAlmostEqual(tr.floor_db, -60.0, delta=3.0)
        base = tr.floor_db
        for _ in range(40):
            tr.update(sine(-26))
        self.assertLessEqual(tr.floor_db - base, 0.3 * 10.5 + 0.5)

    def test_3_gate(self):
        tr = LevelTracker(SR, LiveTuning())
        tr.floor_db = -60.0
        self.assertFalse(tr.update(sine(-56)).gated)
        tr.floor_db = -60.0
        self.assertTrue(tr.update(sine(-50)).gated)
        tr.floor_db = -60.0
        lv = tr.update(np.zeros(FRAME_SAMPLES, np.float32))
        self.assertTrue(lv.zero)
        self.assertFalse(lv.gated)
        self.assertEqual(tr.floor_db, -60.0)

    def test_4_gain(self):
        tr = LevelTracker(SR, LiveTuning())
        for _ in range(40):          # 約 10 秒
            tr.update(sine(-38))
        self.assertAlmostEqual(tr.gain_db, 12.0, delta=1.5)
        tr2 = LevelTracker(SR, LiveTuning())
        for _ in range(40):
            tr2.update(sine(-10))
        self.assertEqual(tr2.gain_db, 0.0)
        tr3 = LevelTracker(SR, LiveTuning())
        tr3.floor_db = -90.0
        for _ in range(200):
            tr3.update(sine(-70))
        self.assertLessEqual(tr3.gain_db, 24.0)

    def test_5_normalize_for_asr(self):
        t = LiveTuning()
        x = sine(-38, SR)
        y = normalize_for_asr(x, [-38.0, -38.0, -38.0], t)
        self.assertAlmostEqual(frame_rms_db(y), -23.0, delta=1.5)
        self.assertTrue(np.all(np.abs(y) < 1.0))
        t2 = LiveTuning(asr_normalize=False)
        self.assertIs(normalize_for_asr(x, [-38.0], t2), x)
        z = np.concatenate([sine(-38, SR), np.ones(800, np.float32)])
        yz = normalize_for_asr(z, [-38.0], t)
        self.assertTrue(np.all(np.abs(yz) < 1.0))


if __name__ == "__main__":
    unittest.main()
