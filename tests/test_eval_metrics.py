# -*- coding: utf-8 -*-
import unittest

import numpy as np

from eval_metrics import char_errors, line_stats, normalize_ja, percentile, word_errors


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

    def test_word_errors(self):
        self.assertEqual(word_errors("the cat sat", "the cat sat")["wer"], 0)
        self.assertEqual(word_errors("the cat sat", "the sat")["D"], 1)
        self.assertEqual(word_errors("The cat, sat.", "the cat sat")["wer"], 0)

    def test_line_stats(self):
        st = line_stats(["to detect.", "before it fires.", "Thank you.", "Bye. Bye. Bye."], "en")
        self.assertEqual(st["lines"], 4)
        self.assertEqual(st["short_ratio"], 1.0)
        self.assertEqual(st["stock_lines"], 2)
        self.assertEqual(st["repeat_lines"], 1)


if __name__ == "__main__":
    unittest.main()
