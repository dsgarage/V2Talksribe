# -*- coding: utf-8 -*-
import unittest

from devices import normalize_device_name, resolve_input_device

IPHONE = "‎DaisukeのiPhoneのマイク"
DEVS = [
    {"index": 0, "name": "MacBook Proのスピーカー", "max_input_channels": 0},
    {"index": 1, "name": "MacBook Proのマイク", "max_input_channels": 1},
    {"index": 2, "name": IPHONE, "max_input_channels": 1},
    {"index": 3, "name": "BlackHole 2ch", "max_input_channels": 2},
    {"index": 4, "name": "BlackHole 2ch Extra", "max_input_channels": 2},
]


class DevicesTest(unittest.TestCase):
    def test_normalize_strips_marks(self):
        self.assertEqual(normalize_device_name(" " + IPHONE + "‏ "), "DaisukeのiPhoneのマイク")

    def test_exact_match_ignoring_mark(self):
        self.assertEqual(resolve_input_device(DEVS, "DaisukeのiPhoneのマイク", 1),
                         (2, IPHONE, False))

    def test_exact_beats_partial(self):
        devs = [DEVS[4], DEVS[3]]
        self.assertEqual(resolve_input_device(devs, "blackhole 2ch", 4), (3, "BlackHole 2ch", False))

    def test_partial_match(self):
        self.assertEqual(resolve_input_device(DEVS, "iphone", 1), (2, IPHONE, False))

    def test_fallback_to_default(self):
        self.assertEqual(resolve_input_device(DEVS, "存在しない", 1),
                         (1, "MacBook Proのマイク", True))

    def test_default_minus_one(self):
        self.assertEqual(resolve_input_device(DEVS, "存在しない", -1), (None, None, True))
        self.assertEqual(resolve_input_device(DEVS, None, -1), (None, None, True))

    def test_default_without_input(self):
        self.assertEqual(resolve_input_device(DEVS, None, 0), (None, None, True))

    def test_wanted_none(self):
        self.assertEqual(resolve_input_device(DEVS, None, 1), (1, "MacBook Proのマイク", False))

    def test_output_only_not_matched(self):
        self.assertEqual(resolve_input_device(DEVS, "スピーカー", 1),
                         (1, "MacBook Proのマイク", True))


if __name__ == "__main__":
    unittest.main()
