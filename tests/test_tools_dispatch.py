"""tools.dispatch — the entrypoint every tool call funnels through.

Covers: alias resolution, unknown-tool suggestion, error suppression.
This exercises the new per-group package layout (collama.tools.*).
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from collama.tools import ToolContext, dispatch
from collama.tools.registry import TOOL_ALIASES


class AliasTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / "hello.txt").write_text("hello world\n")
        self.ctx = ToolContext(root=self.root, yolo=True)

    def tearDown(self):
        self.tmp.cleanup()

    def test_alias_resolves_to_canonical_tool(self):
        """`cat` → `read_file`. Result is annotated with the alias note."""
        out = dispatch("cat", {"path": "hello.txt"}, self.ctx)
        self.assertIn("hello world", out)
        self.assertIn("'cat' is an alias for 'read_file'", out)

    def test_every_alias_target_exists(self):
        """Every entry in TOOL_ALIASES must point to a real tool — otherwise
        we silently send the model down a dead-end."""
        from collama.tools import _all_tools
        all_t = _all_tools()
        missing = sorted(canon for canon in set(TOOL_ALIASES.values())
                         if canon not in all_t)
        self.assertEqual(missing, [], f"aliases pointing to missing tools: {missing}")


class UnknownToolTests(unittest.TestCase):
    def setUp(self):
        self.ctx = ToolContext(root=Path("/tmp"))

    def test_unknown_returns_error_with_suggestions(self):
        out = dispatch("reed_file", {}, self.ctx)
        self.assertTrue(out.startswith("ERROR: unknown tool"))
        # 'reed_file' is close to 'read_file' — it must be suggested.
        self.assertIn("read_file", out)

    def test_unknown_tool_does_not_raise(self):
        # dispatch must wrap all errors so the model gets a string back.
        out = dispatch("totally_made_up_xyz", {}, self.ctx)
        self.assertTrue(out.startswith("ERROR:"))


class ExceptionWrappingTests(unittest.TestCase):
    def test_missing_required_argument_returns_error_string(self):
        """read_file requires `path` — calling without it must yield an ERROR
        line, not a raised KeyError that crashes the loop."""
        ctx = ToolContext(root=Path("/tmp"))
        out = dispatch("read_file", {}, ctx)
        self.assertTrue(out.startswith("ERROR"))


if __name__ == "__main__":
    unittest.main()
