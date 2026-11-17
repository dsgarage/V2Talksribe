# -*- coding: utf-8 -*-
import multiprocessing
import os
import tempfile
import time
import unittest

import numpy as np
import soundfile as sf

from audio_store import WavRecorder, prune_recordings, session_part_path

SR = 16000


def _child_record(path):
    rec = WavRecorder(path, SR)
    chunk = (np.random.rand(SR // 10).astype(np.float32) - 0.5) * 0.2
    for _ in range(30):
        rec.write(chunk)
        time.sleep(0.1)
    rec.close()


class WavRecorderTest(unittest.TestCase):
    def test_killed_child_wav_readable(self):
        with tempfile.TemporaryDirectory() as d:
            path = session_part_path(d, "20260101_000000", 1)
            self.assertTrue(path.endswith("20260101_000000_01.wav"))
            proc = multiprocessing.get_context("spawn").Process(target=_child_record, args=(path,))
            proc.start()
            time.sleep(2.6)
            proc.kill()
            proc.join()
            self.assertGreaterEqual(sf.info(path).duration, 2.0)


class PruneTest(unittest.TestCase):
    def _make(self, d, name, size, age_days, now):
        p = os.path.join(d, name)
        with open(p, "wb") as f:
            f.write(b"0" * size)
        t = now - age_days * 86400
        os.utime(p, (t, t))
        return p

    def test_age_size_protect(self):
        now = time.time()
        with tempfile.TemporaryDirectory() as d:
            old = self._make(d, "a_old.wav", 10, 20, now)
            mid = self._make(d, "b_mid.wav", 100, 5, now)
            new = self._make(d, "c_new.wav", 100, 1, now)
            prot = self._make(d, "cur_old.wav", 10, 30, now)
            calls = []

            def fake(p):
                calls.append(p)
                return True

            # 期限のみ(容量は十分)
            r = prune_recordings(d, max_age_days=14, max_total_bytes=10 ** 9,
                                 protect_prefix="cur", now=now, trash=fake)
            self.assertEqual(r, [old])
            # 容量超過: 残り(mid 100 + new 100 + prot 10)=210 > 150 → 古い順に mid
            r = prune_recordings(d, max_age_days=14, max_total_bytes=150,
                                 protect_prefix="cur", now=now, trash=fake)
            self.assertEqual(r, [old, mid])
            self.assertNotIn(prot, r)
            self.assertNotIn(new, r)
            # trash が False なら一覧に入らない
            self.assertEqual(prune_recordings(d, now=now, trash=lambda p: False), [])
            self.assertTrue(all(os.path.exists(p) for p in (old, mid, new, prot)))

    def test_missing_dir(self):
        self.assertEqual(prune_recordings("/nonexistent_dir_xyz"), [])


if __name__ == "__main__":
    unittest.main()
