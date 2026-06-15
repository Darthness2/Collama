"""StatusBar must never strand the cursor at the home position.

The bottom status bar reserves the last terminal row with DECSTBM
(`ESC [ top;bottom r`). Per the VT spec that escape — and its reset
(`ESC [ r`) — homes the cursor to (1,1). If a scroll-region change isn't
bracketed by a cursor save (DECSC, `ESC 7`) / restore (DECRC, `ESC 8`),
the cursor is left at the top of the screen and the model's streamed
answer writes right over the banner and previous output.

These tests assert the structural invariant directly: *every* scroll-region
escape the StatusBar emits lies between a save and a later restore.
"""
from __future__ import annotations

import io
import os
import re
import sys
import time
import unittest

from collama import ui


_REGION_RX = re.compile(r"\x1b\[[0-9;]*r")   # DECSTBM set/reset


class _FakeTTY(io.StringIO):
    def isatty(self) -> bool:  # StatusBar no-ops unless stdout is a tty
        return True


def _region_escapes_are_bracketed(s: str) -> bool:
    """True iff every scroll-region escape sits inside an open DECSC/DECRC
    pair — i.e. the cursor is always saved before and restored after any
    region change, so it can never be stranded at home."""
    depth = 0
    i = 0
    n = len(s)
    while i < n:
        two = s[i : i + 2]
        if two == "\x1b7":          # DECSC — save
            depth += 1
            i += 2
            continue
        if two == "\x1b8":          # DECRC — restore
            depth = max(0, depth - 1)
            i += 2
            continue
        m = _REGION_RX.match(s, i)
        if m:
            if depth <= 0:
                return False        # region change with no save in effect
            i = m.end()
            continue
        i += 1
    return True


class StatusBarCursorSafetyTests(unittest.TestCase):
    def setUp(self) -> None:
        self._real_stdout = sys.stdout
        self._fake = _FakeTTY()
        sys.stdout = self._fake
        # Force the bar on regardless of the environment running the tests.
        self._prev_env = os.environ.pop("COLLAMA_STATUS_BAR", None)

    def tearDown(self) -> None:
        sys.stdout = self._real_stdout
        if self._prev_env is not None:
            os.environ["COLLAMA_STATUS_BAR"] = self._prev_env

    def _run_lifecycle(self) -> str:
        bar = ui.StatusBar()
        bar.start(ctx_tokens=1234)
        # Let the background painter run a few frames (incl. the resize path
        # the first frame may exercise).
        time.sleep(0.3)
        bar.stop()
        return self._fake.getvalue()

    def test_region_changes_are_always_bracketed(self) -> None:
        out = self._run_lifecycle()
        self.assertIn("\x1b[", out, "status bar produced no output")
        self.assertTrue(
            _REGION_RX.search(out),
            "status bar never set a scroll region — test would be vacuous",
        )
        self.assertTrue(
            _region_escapes_are_bracketed(out),
            "a scroll-region escape was emitted outside a cursor save/restore "
            "bracket — the cursor can be stranded at home and the model will "
            "write over the screen",
        )

    def test_cursor_is_restored_last(self) -> None:
        # The final cursor-affecting escape must be a restore, never a bare
        # region reset — otherwise the next prompt is drawn at the top.
        out = self._run_lifecycle()
        tail = out.rsplit("\x1b8", 1)[-1]
        self.assertFalse(
            _REGION_RX.search(tail),
            "scroll region was changed AFTER the last cursor restore — the "
            "cursor ends up homed and output overwrites the screen",
        )


if __name__ == "__main__":
    unittest.main()
