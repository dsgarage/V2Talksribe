# -*- coding: utf-8 -*-
import unittest

from settings import AppSettings, LiveTuning, apply_tuning_overrides


class LiveTuningTest(unittest.TestCase):
    def test_1_from_dict(self):
        s = AppSettings.from_dict({"live_tuning": {
            "silence_end_sec": "1.4", "asr_normalize": 0, "temperature": [0, 0.4],
            "unknown": 1, "repeat_min": "x"}})
        t = s.live_tuning
        self.assertEqual(t.silence_end_sec, 1.4)
        self.assertIs(t.asr_normalize, False)
        self.assertEqual(t.temperature, (0.0, 0.4))
        self.assertFalse(hasattr(t, "unknown"))
        self.assertEqual(t.repeat_min, 3)

    def test_2_overrides(self):
        t = apply_tuning_overrides(LiveTuning(), ["silence_end_sec=1.4", "halluc_filter=off",
                                                   "temperature=0.0,0.4"])
        self.assertEqual(t.silence_end_sec, 1.4)
        self.assertIs(t.halluc_filter, False)
        self.assertEqual(t.temperature, (0.0, 0.4))
        with self.assertRaises(ValueError) as cm:
            apply_tuning_overrides(LiveTuning(), ["nope=1"])
        self.assertIn("nope", str(cm.exception))
        with self.assertRaises(ValueError):
            apply_tuning_overrides(LiveTuning(), ["repeat_min=x"])

    def test_3_roundtrip(self):
        s = AppSettings()
        s.live_tuning.silence_end_sec = 1.5
        d = s.to_dict()
        self.assertIn("live_tuning", d)
        s2 = AppSettings.from_dict(d)
        self.assertEqual(s2.live_tuning, s.live_tuning)


if __name__ == "__main__":
    unittest.main()
