"""services.compact — conversation buffer trimming and the compact boundary."""
from __future__ import annotations

import unittest

from collama.services.compact import (
    BOUNDARY_MARKER,
    find_compact_boundary,
    messages_after_boundary,
    snip_compact,
)


class BoundaryTests(unittest.TestCase):
    def test_absent_returns_minus_one(self):
        msgs = [
            {"role": "system", "content": "you are a coding agent"},
            {"role": "user", "content": "hello"},
        ]
        self.assertEqual(find_compact_boundary(msgs), -1)

    def test_finds_last_boundary_when_multiple(self):
        msgs = [
            {"role": "system", "content": f"x {BOUNDARY_MARKER}"},
            {"role": "user", "content": "intervening"},
            {"role": "system", "content": f"y {BOUNDARY_MARKER}"},
            {"role": "user", "content": "post"},
        ]
        self.assertEqual(find_compact_boundary(msgs), 2)

    def test_after_boundary_returns_only_post_messages(self):
        msgs = [
            {"role": "user", "content": "old"},
            {"role": "system", "content": BOUNDARY_MARKER},
            {"role": "user", "content": "new1"},
            {"role": "assistant", "content": "new2"},
        ]
        after = messages_after_boundary(msgs)
        self.assertEqual([m["content"] for m in after], ["new1", "new2"])

    def test_after_boundary_with_no_boundary_returns_copy_of_all(self):
        msgs = [{"role": "user", "content": "a"}, {"role": "user", "content": "b"}]
        after = messages_after_boundary(msgs)
        self.assertEqual(after, msgs)
        # Must be a copy so callers can mutate without aliasing.
        after.append({"role": "user", "content": "c"})
        self.assertEqual(len(msgs), 2)


class SnipCompactTests(unittest.TestCase):
    def test_drops_empty_assistant_zombies(self):
        msgs = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "", "tool_calls": []},
        ]
        n = snip_compact(msgs)
        self.assertEqual(n, 1)
        self.assertEqual(len(msgs), 1)

    def test_keeps_assistant_with_tool_calls(self):
        msgs = [
            {"role": "assistant", "content": "",
             "tool_calls": [{"function": {"name": "read_file"}}]},
        ]
        n = snip_compact(msgs)
        self.assertEqual(n, 0, "assistant with tool_calls is not a zombie")
        self.assertEqual(len(msgs), 1)

    def test_dedupes_consecutive_identical_user_messages(self):
        msgs = [
            {"role": "user", "content": "same"},
            {"role": "user", "content": "same"},
        ]
        n = snip_compact(msgs)
        self.assertEqual(n, 1)
        self.assertEqual(len(msgs), 1)

    def test_strips_old_thinking_blocks_but_keeps_recent(self):
        """Thinking is reasoning, not decisions — keep only the last
        keep_recent_think assistants' <think> blocks."""
        msgs = [
            {"role": "assistant",
             "content": "<thinking>old reasoning</thinking>I will A"},
            {"role": "user", "content": "ok"},
            {"role": "assistant",
             "content": "<thinking>new reasoning</thinking>I will B"},
        ]
        snip_compact(msgs, keep_recent_think=1)
        self.assertNotIn("<thinking>", msgs[0]["content"])
        self.assertIn("I will A", msgs[0]["content"])
        self.assertIn("<thinking>", msgs[-1]["content"],
                      "the most recent thinking block must be preserved")


if __name__ == "__main__":
    unittest.main()
