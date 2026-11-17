# -*- coding: utf-8 -*-
import unittest

from hallucination import collapse_repeats, is_stock_phrase, norm_phrase


class HallucinationTest(unittest.TestCase):
    def test_1_collapse_en(self):
        self.assertEqual(
            collapse_repeats("Thanks for watching. Bye. Bye. Bye. Bye. Bye. Bye. Bye."),
            ("Thanks for watching. Bye.", True))

    def test_2_collapse_ja_and_min(self):
        self.assertEqual(collapse_repeats("ありがとうありがとうありがとう"), ("ありがとう", True))
        self.assertEqual(collapse_repeats("Bye. Bye."), ("Bye. Bye.", False))

    def test_3_stock_true(self):
        self.assertTrue(is_stock_phrase("Thank you.", "en"))
        self.assertTrue(is_stock_phrase("Thanks for watching. Bye.", "en"))
        self.assertTrue(is_stock_phrase("ご視聴ありがとうございました", "ja"))

    def test_4_stock_false(self):
        self.assertFalse(is_stock_phrase("Thank you for the detailed explanation.", "en"))
        self.assertFalse(is_stock_phrase("UGUIのフィーチャルを説明します", "ja"))
        self.assertFalse(is_stock_phrase("おめでとうございます", "ja"))

    def test_5_norm_phrase(self):
        self.assertEqual(norm_phrase("Thank you!"), norm_phrase("thank you"))


class CollapseKeepsRealSpeechTest(unittest.TestCase):
    def test_not_weak_min5_keeps_original(self):
        for text in ["1000000円です", "2222年", "ははは、そうですね", "いいいいい", "www", "ladadada",
                     "very very very good", "ねえねえねえ聞いて"]:
            with self.subTest(text=text):
                self.assertEqual(collapse_repeats(text, 5), (text, False))

    def test_digits_and_single_chars_kept_even_at_min3(self):
        for text in ["1000000円です", "2222年", "www", "いいいいい"]:
            with self.subTest(text=text):
                self.assertEqual(collapse_repeats(text, 3), (text, False))

    def test_weak_collapses(self):
        self.assertEqual(collapse_repeats("Bye. Bye. Bye. Bye.", 3), ("Bye.", True))
        self.assertEqual(collapse_repeats("ありがとうございました。" * 3, 3), ("ありがとうございました。", True))
        self.assertEqual(collapse_repeats("なるほど" * 5, 5), ("なるほど", True))


if __name__ == "__main__":
    unittest.main()
