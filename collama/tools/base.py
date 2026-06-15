"""Shared primitives for every tool module: ToolContext, output truncation,
path resolution, the edit-history recorder, and the failed-command analyzer.
"""
from __future__ import annotations

import logging
import os
import re as _re_err
from dataclasses import dataclass
from pathlib import Path

from .. import ui


logger = logging.getLogger(__name__)

MAX_OUTPUT_CHARS = 16000
MAX_EDIT_HISTORY = 50


def _truncate(s: str, limit: int = MAX_OUTPUT_CHARS) -> str:
    if len(s) <= limit:
        return s
    return s[:limit] + f"\n…[truncated, {len(s) - limit} more chars]"


def _resolve(path: str, root: Path) -> Path:
    expanded = os.path.expanduser(os.path.expandvars(path))
    p = Path(expanded)
    if not p.is_absolute():
        p = root / p
    return p


class PathEscapeError(Exception):
    """Raised when a resolved path would escape the workspace root.

    Carries a ready-to-return ``ERROR: path escapes workspace ...`` message so
    callers can simply ``return exc.message``.
    """

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


def _is_within(child_real: str, root_real: str) -> bool:
    """True if ``child_real`` is ``root_real`` or lives underneath it.

    Both arguments must already be real (symlink-resolved, absolute) paths.
    Uses a normalized ``commonpath`` comparison and a ``startswith(root + sep)``
    fallback so it behaves correctly on Python 3.9 and across drive roots.
    """
    if child_real == root_real:
        return True
    try:
        if os.path.commonpath([child_real, root_real]) == root_real:
            return True
    except ValueError:
        # Different drives / mix of absolute+relative — definitely not within.
        return False
    # Defensive fallback (also guards against commonpath edge cases).
    root_with_sep = root_real.rstrip(os.sep) + os.sep
    return child_real.startswith(root_with_sep)


def _resolve_contained(path: str, root: Path) -> Path:
    """Resolve ``path`` like :func:`_resolve` but REQUIRE the real result to
    stay inside ``root``. Expands ``~`` / ``$VARS`` and accepts absolute paths,
    yet refuses any result (including via symlinks) that lands above ``root``.

    Relative paths and in-workspace absolute paths keep working; only traversal
    ABOVE the workspace is blocked.

    Raises :class:`PathEscapeError` (whose ``.message`` is a user-facing
    ``ERROR: path escapes workspace ...`` string) when containment fails.
    """
    p = _resolve(path, root)
    # os.path.realpath resolves symlinks and ".." for both existing and
    # not-yet-existing paths, so a symlink that points outside the workspace —
    # or a "../" traversal — is caught even when the target doesn't exist.
    child_real = os.path.realpath(str(p))
    root_real = os.path.realpath(str(root))
    if not _is_within(child_real, root_real):
        raise PathEscapeError(
            f"ERROR: path escapes workspace: {path} resolves to {child_real}, "
            f"which is outside {root_real}. Only paths inside the workspace are "
            f"allowed; use set_workspace to change the workspace root."
        )
    return Path(child_real)


@dataclass
class ToolContext:
    root: Path
    yolo: bool = False
    github_token: str | None = None
    insecure_ssl: bool = False
    # Optional plumbing for engine-aware tools (subagent, worktree,
    # background, tasks). These are populated by the QueryEngine when it
    # builds the ctx; pure-FS tools ignore them.
    state: object | None = None
    engine: object | None = None
    background: object | None = None
    tasks: object | None = None
    teams: object | None = None
    # Per-turn read cache: {resolved_path -> last result}. Plumbed in by the
    # engine; read_file populates it, edit_file/write_file invalidate it.
    read_cache: dict | None = None

    def confirm(self, action: str, detail: str) -> bool:
        if self.yolo:
            return True
        ui.warn(f"\nApprove {action}? {detail}")
        try:
            ans = input("  [y]es / [n]o / [a]lways: ").strip().lower()
        except EOFError:
            return False
        if ans in ("a", "always"):
            self.yolo = True
            return True
        return ans in ("y", "yes")


def _record_edit(ctx: ToolContext, path: Path, before: str, after: str, op: str) -> None:
    """Push an edit to state.edit_history and invalidate any read cache for
    this path. Powers /undo and /diff and prevents the read cache from
    serving stale content after a write."""
    state = getattr(ctx, "state", None)
    if state is not None:
        import time as _t
        hist: list = getattr(state, "edit_history", None) or []
        hist.append({
            "ts": _t.time(),
            "path": str(path.resolve()),
            "before": before,
            "after": after,
            "op": op,
        })
        del hist[:-MAX_EDIT_HISTORY]
        state.update(edit_history=hist)
    rp = str(path.resolve())
    cache = getattr(ctx, "read_cache", None)
    if isinstance(cache, dict):
        cache.pop(rp, None)
        # also drop any legacy tuple-keyed entries (older versions of the cache)
        for k in list(cache):
            if isinstance(k, tuple) and k and k[0] == rp:
                del cache[k]
    if state is not None:
        files_read = set(getattr(state, "files_read", set()) or set())
        if rp in files_read:
            files_read.discard(rp)
            state.update(files_read=files_read)


# Error-location patterns, most-specific first. Used to point the model
# straight at the failing file:line when a command exits non-zero.
_ERR_LOC_PATTERNS = [
    _re_err.compile(r'File "([^"]+)", line (\d+)'),          # Python traceback
    _re_err.compile(r'-->\s+([^\s:]+):(\d+):\d+'),           # Rust
    _re_err.compile(r'\(([^()\s]+):(\d+):\d+\)'),            # Node "at (file:line:col)"
    _re_err.compile(r'([\w./\\+-]+\.\w+):(\d+):\d+'),        # tsc / gcc / go / eslint
    _re_err.compile(r'([\w./\\+-]+\.\w+):(\d+)\b'),          # generic file:line
]
_ERR_MSG_RX = _re_err.compile(
    r'^\s*([A-Z]\w*(?:Error|Exception|Warning|Fault)): ?(.*)$', _re_err.M
)


def _analyze_failure(stdout: str, stderr: str) -> str:
    """Scan failed-command output for an error message + file:line so the
    model can jump straight to the bug. Returns a one-line hint or ''."""
    blob = (stderr or "") + "\n" + (stdout or "")
    hints: list[str] = []
    msgs = _ERR_MSG_RX.findall(blob)
    if msgs:
        kind, detail = msgs[-1]
        hints.append(f"{kind}: {detail.strip()[:200]}")
    loc = None
    for pat in _ERR_LOC_PATTERNS:
        found = pat.findall(blob)
        if found:
            loc = found[-1]  # last match = deepest / actual error frame
            break
    if loc:
        hints.append(f"likely at {loc[0]}:{loc[1]}")
    return ("  ↳ " + "  ·  ".join(hints)) if hints else ""
