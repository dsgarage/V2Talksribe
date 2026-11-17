# -*- coding: utf-8 -*-
import glob
import os
import re
import tempfile
import unittest

import applog
from settings import AppSettings


class SettingsTest(unittest.TestCase):
    def test_migrate_with_index(self):
        s = AppSettings(input_device_id=2)
        self.assertTrue(s.migrate_input_device_name({2: "‎iPhoneのマイク"}))
        self.assertEqual(s.input_device_name, "iPhoneのマイク")
        self.assertIsNone(s.input_device_id)

    def test_migrate_unknown_index(self):
        s = AppSettings(input_device_id=9)
        self.assertTrue(s.migrate_input_device_name({2: "x"}))
        self.assertIsNone(s.input_device_name)
        self.assertIsNone(s.input_device_id)

    def test_migrate_no_index(self):
        s = AppSettings()
        self.assertFalse(s.migrate_input_device_name({0: "x"}))

    def test_migrate_name_already_set(self):
        s = AppSettings(input_device_id=1, input_device_name="A")
        self.assertFalse(s.migrate_input_device_name({1: "B"}))
        self.assertEqual(s.input_device_name, "A")
        self.assertEqual(s.input_device_id, 1)

    def test_save_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "sub", "config.json")
            s = AppSettings(input_device_name="マイク", language="ja")
            s.save(path)
            r = AppSettings.load(path)
            self.assertEqual(r.input_device_name, "マイク")
            self.assertEqual(r.language, "ja")


class ApplogTest(unittest.TestCase):
    def test_rotation_and_format(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "debug.log")
            old = (applog.LOG_PATH, applog.MAX_BYTES, applog.BACKUP_COUNT)
            applog._reset_logger()
            applog.LOG_PATH, applog.MAX_BYTES, applog.BACKUP_COUNT = path, 500, 2
            try:
                for i in range(100):
                    applog.log(f"テスト行 {i}")
                applog._reset_logger()
                self.assertTrue(os.path.exists(path + ".1"))
                with open(path, encoding="utf-8") as f:
                    lines = f.read().splitlines()
                self.assertTrue(lines)
                for ln in lines:
                    self.assertRegex(ln, r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} \[\d+\] ")
            finally:
                applog._reset_logger()
                applog.LOG_PATH, applog.MAX_BYTES, applog.BACKUP_COUNT = old


if __name__ == "__main__":
    unittest.main()
