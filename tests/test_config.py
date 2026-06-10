"""config.py — JSON config, deep merge, dotted-key get/set, atomic save."""
from __future__ import annotations

import json
import os
import stat
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from collama import config


class MergeTests(unittest.TestCase):
    def test_override_replaces_scalar(self):
        merged = config._merge({"a": 1}, {"a": 2})
        self.assertEqual(merged, {"a": 2})

    def test_recursive_merges_nested_dict(self):
        base = {"github": {"token": None, "host": "github.com"}}
        override = {"github": {"token": "abc"}}
        merged = config._merge(base, override)
        self.assertEqual(merged, {"github": {"token": "abc", "host": "github.com"}})

    def test_non_dict_override_replaces_dict(self):
        merged = config._merge({"github": {"token": None}}, {"github": None})
        self.assertEqual(merged, {"github": None})

    def test_merge_does_not_mutate_base(self):
        base = {"github": {"token": None}}
        config._merge(base, {"github": {"token": "abc"}})
        self.assertEqual(base, {"github": {"token": None}})


class DottedKeyTests(unittest.TestCase):
    def test_set_value_creates_intermediate_dicts(self):
        cfg: dict = {}
        config.set_value(cfg, "github.token", "abc")
        self.assertEqual(cfg, {"github": {"token": "abc"}})

    def test_set_value_overwrites_non_dict_intermediate(self):
        """If the intermediate exists but isn't a dict, replace it — otherwise
        a later get would crash on stale junk."""
        cfg = {"github": "stringly"}
        config.set_value(cfg, "github.token", "abc")
        self.assertEqual(cfg, {"github": {"token": "abc"}})

    def test_set_value_deep_path(self):
        cfg: dict = {}
        config.set_value(cfg, "a.b.c.d", 42)
        self.assertEqual(cfg, {"a": {"b": {"c": {"d": 42}}}})

    def test_get_value_returns_nested_value(self):
        cfg = {"github": {"token": "abc"}}
        self.assertEqual(config.get_value(cfg, "github.token"), "abc")

    def test_get_value_missing_returns_default(self):
        self.assertIsNone(config.get_value({}, "github.token"))
        self.assertEqual(config.get_value({}, "github.token", "fallback"), "fallback")

    def test_get_value_traverses_through_non_dict_returns_default(self):
        """If a midpoint is a non-dict (e.g. set to a string), get_value
        must not crash — it returns the default."""
        cfg = {"github": "stringly"}
        self.assertIsNone(config.get_value(cfg, "github.token"))


class SaveLoadTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        # Force config_dir to point inside the temp dir for this test.
        self._patcher = mock.patch.dict(
            os.environ, {"XDG_CONFIG_HOME": self.tmp.name}, clear=False
        )
        self._patcher.start()

    def tearDown(self):
        self._patcher.stop()
        self.tmp.cleanup()

    def test_load_returns_defaults_when_no_file(self):
        cfg = config.load()
        self.assertEqual(cfg.get("host"), "http://localhost:11434")
        self.assertEqual(cfg.get("temperature"), 0.2)
        self.assertEqual(cfg.get("effort"), "medium")
        self.assertFalse(cfg.get("yolo"))

    def test_save_then_load_roundtrip(self):
        cfg = config.load()
        cfg["model"] = "qwen2.5-coder:14b"
        config.set_value(cfg, "github.token", "ghp_abc")
        config.save(cfg)

        loaded = config.load()
        self.assertEqual(loaded["model"], "qwen2.5-coder:14b")
        self.assertEqual(loaded["github"]["token"], "ghp_abc")

    def test_save_file_is_chmod_600(self):
        cfg = config.load()
        cfg["model"] = "x"
        config.save(cfg)
        mode = stat.S_IMODE(config.config_path().stat().st_mode)
        self.assertEqual(mode, 0o600,
                         "config holds a GitHub PAT — must be user-only")

    def test_load_recovers_from_corrupt_json(self):
        config.config_dir().mkdir(parents=True, exist_ok=True)
        config.config_path().write_text("{not valid json")
        cfg = config.load()
        self.assertEqual(cfg.get("host"), "http://localhost:11434")

    def test_save_is_atomic(self):
        """save() writes to .tmp then replaces — a partial file should never
        be visible at the real path."""
        cfg = config.load()
        cfg["model"] = "first"
        config.save(cfg)
        # Sanity: file ends with } (complete JSON), not the start of {"...
        text = config.config_path().read_text()
        self.assertTrue(text.rstrip().endswith("}"))
        # Reload + resave with new value; old value should not bleed through.
        cfg = config.load()
        cfg["model"] = "second"
        config.save(cfg)
        self.assertEqual(json.loads(config.config_path().read_text())["model"],
                         "second")


if __name__ == "__main__":
    unittest.main()
