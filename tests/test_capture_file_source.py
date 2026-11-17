# -*- coding: utf-8 -*-
import os
import tempfile
import unittest

import numpy as np
import soundfile as sf

import capture

SR = 16000


def _write_wav(path):
    t = np.arange(SR * 3) / SR
    sf.write(path, (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32), SR)


class FileCaptureTest(unittest.TestCase):
    def test_stream_chunks_and_eof(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "tone.wav")
            _write_wav(path)
            h = capture.start_file_capture(SR, 0.5, path, speed=0)
            try:
                self.assertTrue(h.ctrl.poll(10))
                msg = h.ctrl.recv()
                self.assertEqual(msg["type"], "started")
                self.assertTrue(msg["device_name"].startswith("file:"))
                for _ in range(6):
                    self.assertTrue(h.audio.poll(5))
                    x = np.frombuffer(h.audio.recv_bytes(), np.float32)
                    self.assertEqual(len(x), 8000)
                self.assertTrue(h.ctrl.poll(5))
                self.assertEqual(h.ctrl.recv()["type"], "eof")
            finally:
                h.stop()
            self.assertFalse(h.is_alive())

    def test_child_killed(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "tone.wav")
            _write_wav(path)
            h = capture.start_file_capture(SR, 0.5, path, speed=1.0)
            try:
                self.assertTrue(h.ctrl.poll(10))
                self.assertEqual(h.ctrl.recv()["type"], "started")
                h.proc.kill()
                h.proc.join(3)
                while h.audio.poll(2):
                    with self.assertRaises((EOFError, OSError)):
                        # データが残っていれば読み切り、尽きたら EOFError
                        for _ in range(100):
                            h.audio.recv_bytes()
                        self.fail("EOF にならなかった")
                    break
            finally:
                h.stop()
            h.stop()  # 2 回呼んでも安全


if __name__ == "__main__":
    unittest.main()
