# -*- coding: utf-8 -*-
import unittest

import numpy as np

from eval_metrics import char_errors, normalize_ja, percentile


class EvalMetricsTest(unittest.TestCase):
    def test_normalize_ja(self):
        self.assertEqual(normalize_ja("[12:00:01] こんにちは、 世界。\n# 見出し"), "こんにちは世界")

    def test_identical(self):
        self.assertEqual(char_errors("abcdef", "abcdef")["cer"], 0)

    def test_substitution(self):
        e = char_errors("abcdef", "abxdef")
        self.assertEqual((e["S"], e["D"], e["I"]), (1, 0, 0))

    def test_deletion(self):
        e = char_errors("abcdef", "abef")
        self.assertEqual(e["D"], 2)
        self.assertEqual(e["longest_deletion"], 2)

    def test_insertion(self):
        self.assertEqual(char_errors("abc", "abxyc")["I"], 2)

    def test_percentile(self):
        self.assertAlmostEqual(percentile([1, 2, 3, 4], 50), float(np.percentile([1, 2, 3, 4], 50)))
        self.assertAlmostEqual(percentile([1, 2, 3, 4], 95), float(np.percentile([1, 2, 3, 4], 95)))
        self.assertEqual(percentile([], 50), 0.0)


if __name__ == "__main__":
    unittest.main()
