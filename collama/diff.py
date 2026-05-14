"""Colorized unified diffs for terminal display."""
from __future__ import annotations

import difflib

from . import ui


# Keep file-edit diffs short in the terminal — show a preview and cut off.
DEFAULT_MAX_LINES = 12


def render(old: str, new: str, path: str, context: int = 3,
           max_lines: int = DEFAULT_MAX_LINES) -> str:
    """Return a colored unified-diff string, truncated to `max_lines`.

    Empty if nothing changed. When the diff is longer than `max_lines`, the
    preview is cut off and a '… +N more lines' marker is appended so you can
    see something changed without flooding the terminal.
    """
    old_lines = old.splitlines(keepends=False)
    new_lines = new.splitlines(keepends=False)
    if old == new:
        return ""

    diff_lines = list(difflib.unified_diff(
        old_lines, new_lines,
        fromfile=f"a/{path}", tofile=f"b/{path}",
        n=context, lineterm="",
    ))

    out = []
    shown = diff_lines[:max_lines]
    for line in shown:
        if line.startswith("+++") or line.startswith("---"):
            out.append(ui.color(line, ui.BOLD))
        elif line.startswith("@@"):
            out.append(ui.color(line, ui.CYAN))
        elif line.startswith("+"):
            out.append(ui.color(line, ui.GREEN))
        elif line.startswith("-"):
            out.append(ui.color(line, ui.RED))
        else:
            out.append(ui.color(line, ui.GRAY))

    remaining = len(diff_lines) - len(shown)
    if remaining > 0:
        out.append(ui.color(f"  … +{remaining} more diff line(s) (truncated)", ui.GRAY))
    return "\n".join(out)


def stats(old: str, new: str) -> tuple[int, int]:
    """(additions, deletions) line counts."""
    old_lines = old.splitlines()
    new_lines = new.splitlines()
    sm = difflib.SequenceMatcher(a=old_lines, b=new_lines, autojunk=False)
    adds = dels = 0
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "replace":
            dels += i2 - i1
            adds += j2 - j1
        elif tag == "delete":
            dels += i2 - i1
        elif tag == "insert":
            adds += j2 - j1
    return adds, dels


def stats_line(adds: int, dels: int) -> str:
    return ui.color(f"+{adds}", ui.GREEN) + " " + ui.color(f"-{dels}", ui.RED)
