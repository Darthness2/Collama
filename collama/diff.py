"""Colorized unified diffs for terminal display."""
from __future__ import annotations

import difflib

from . import ui


def render(old: str, new: str, path: str, context: int = 3, max_lines: int = 200) -> str:
    """Return a colored unified-diff string. Empty if nothing changed."""
    old_lines = old.splitlines(keepends=False)
    new_lines = new.splitlines(keepends=False)
    if old == new:
        return ""

    diff = difflib.unified_diff(
        old_lines, new_lines,
        fromfile=f"a/{path}", tofile=f"b/{path}",
        n=context, lineterm="",
    )

    out = []
    count = 0
    for line in diff:
        count += 1
        if count > max_lines:
            out.append(ui.color(f"  …diff truncated after {max_lines} lines", ui.GRAY))
            break
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
