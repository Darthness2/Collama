"""diff.py — addition/deletion counts and short colorized render."""
from __future__ import annotations

import unittest

from collama.diff import render, stats


class StatsTests(unittest.TestCase):
    def test_identical_strings_have_no_changes(self):
        self.assertEqual(stats("abc\n", "abc\n"), (0, 0))

    def test_pure_addition(self):
        old = "line1\nline2\n"
        new = "line1\nline2\nline3\n"
        self.assertEqual(stats(old, new), (1, 0))

    def test_pure_deletion(self):
        old = "line1\nline2\nline3\n"
        new = "line1\n"
        self.assertEqual(stats(old, new), (0, 2))

    def test_replace_counts_both_sides(self):
        old = "alpha\nbeta\n"
        new = "alpha\nGAMMA\n"
        adds, dels = stats(old, new)
        self.assertEqual((adds, dels), (1, 1))


class RenderTests(unittest.TestCase):
    def test_identical_renders_empty(self):
        self.assertEqual(render("abc", "abc", "f.py"), "")

    def test_diff_truncates_with_more_marker(self):
        old = "\n".join(f"line{i}" for i in range(40))
        new = "\n".join(f"changed{i}" for i in range(40))
        out = render(old, new, "f.py", max_lines=4)
        self.assertIn("more diff line(s)", out)
        # All shown lines should be a tight prefix of the full diff.
        self.assertLessEqual(out.count("\n"), 5)


if __name__ == "__main__":
    unittest.main()
