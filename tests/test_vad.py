# -*- coding: utf-8 -*-
import unittest

import numpy as np

from vad import VoiceActivityDetector


class VadTest(unittest.TestCase):
    def test_webrtc_silence(self):
        v = VoiceActivityDetector("webrtc", 16000)
        v.load()
        self.assertFalse(v.is_speech(np.zeros(8000, dtype=np.float32)))

    def test_silero_load_and_silence(self):
        v = VoiceActivityDetector("silero", 16000)
        v.load()
        self.assertFalse(v.is_speech(np.zeros(8000, dtype=np.float32)))

    def test_silero_non_multiple_of_512(self):
        v = VoiceActivityDetector("silero", 16000)
        v.load()
        noise = (np.random.RandomState(0).randn(8000) * 0.02).astype(np.float32)
        v.is_speech(noise)  # 真偽は問わない。例外が出ないこと

    def test_silero_reset(self):
        v = VoiceActivityDetector("silero", 16000)
        v.load()
        v.reset()

    def test_unknown_backend(self):
        with self.assertRaises(ValueError):
            VoiceActivityDetector("nope", 16000).load()


if __name__ == "__main__":
    unittest.main()
