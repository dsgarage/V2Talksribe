# -*- coding: utf-8 -*-
import unittest

from segmenter import SegInfo, dedupe_prefix, plan_window_commit


def S(a, b, t):
    return SegInfo(a, b, t)


class SegmenterTest(unittest.TestCase):
    def test_1_empty(self):
        p = plan_window_commit([], 10.0, force=False)
        self.assertEqual(p.segs, [])
        self.assertAlmostEqual(p.cut_sec, 8.0)

    def test_2_force(self):
        p = plan_window_commit([S(0, 3, "a"), S(3.5, 7, "b")], 20.0, force=True)
        self.assertEqual(p.texts, ["a", "b"])
        self.assertEqual(p.cut_sec, 20.0)

    def test_3_keep_last(self):
        p = plan_window_commit([S(0, 3, "a"), S(3.5, 7, "b"), S(8, 19.5, "c")], 20.0, force=False)
        self.assertEqual(p.texts, ["a", "b"])
        self.assertAlmostEqual(p.cut_sec, 7.8)

    def test_4_last_stable(self):
        p = plan_window_commit([S(0, 3, "a"), S(4, 15, "b")], 20.0, force=False)
        self.assertEqual(p.texts, ["a", "b"])
        self.assertAlmostEqual(p.cut_sec, 15.0)

    def test_5_single_to_end(self):
        p = plan_window_commit([S(0, 19.8, "a")], 20.0, force=False)
        self.assertEqual(p.texts, [])
        self.assertEqual(p.cut_sec, 0.0)

    def test_6_cut_bounds(self):
        p = plan_window_commit([S(0, 0.05, "a"), S(0.1, 19.9, "b")], 20.0, force=False)
        self.assertTrue(0.0 <= p.cut_sec <= 20.0)
        p = plan_window_commit([S(0.1, 19.9, "b")], 20.0, force=False)
        self.assertTrue(0.0 <= p.cut_sec <= 20.0)

    def test_7_dedupe(self):
        self.assertEqual(dedupe_prefix("今日は晴れです", "晴れです。明日は"), "。明日は")

    def test_8_dedupe_none(self):
        self.assertEqual(dedupe_prefix("abc", "xyz"), "xyz")
        self.assertEqual(dedupe_prefix("", "xyz"), "xyz")
        self.assertEqual(dedupe_prefix("abc", "cxyz"), "cxyz")   # 1 文字だけの一致は取り除かない


class DedupeBoundaryTest(unittest.TestCase):
    def test_short_match_not_removed(self):
        self.assertEqual(dedupe_prefix("I think so", "so I went"), "so I went")

    def test_long_match_removed(self):
        self.assertEqual(dedupe_prefix("ありがとう", "ありがとうございました"), "ございました")
        self.assertEqual(dedupe_prefix("... we can see", "we can see the result"), "the result")

    def test_mid_word_cut_not_adopted(self):
        self.assertEqual(dedupe_prefix("this is the test", "this is the testing"), "this is the testing")


if __name__ == "__main__":
    unittest.main()
