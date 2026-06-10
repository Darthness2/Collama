"""Core file tools: read / write / edit / multi_edit / replace_lines /
list_dir / grep / run_bash / set_workspace. The bulk of every session
runs through these; everything else is sugar on top.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

from .. import diff as _diff
from .. import ui
from .base import (
    MAX_OUTPUT_CHARS,
    ToolContext,
    _analyze_failure,
    _record_edit,
    _resolve,
    _truncate,
)


def t_read_file(args: dict, ctx: ToolContext) -> str:
    path = args["path"]
    start = int(args.get("start_line", 1))
    end = args.get("end_line")
    p = _resolve(path, ctx.root)
    if not p.exists():
        return f"ERROR: file not found: {path}"
    if not p.is_file():
        return f"ERROR: not a file: {path}"

    # Per-turn cache — keyed on PATH ONLY so the model can't bypass it by
    # passing slightly different line ranges each call. The cached content
    # is the full file (or whatever was read first); the model has it in
    # context already and should scroll back, not re-read.
    abs_path = str(p.resolve())
    cache = ctx.read_cache if isinstance(ctx.read_cache, dict) else None
    state = getattr(ctx, "state", None)
    already_read = (
        (cache is not None and abs_path in cache)
        or (state is not None and abs_path in getattr(state, "files_read", set()))
    )
    if already_read:
        return (
            f"[CACHED — you already read {path} earlier in this turn. "
            f"The full content is in your context above; SCROLL BACK and use it. "
            f"Do NOT re-read this file with any args. Act now (edit_file, "
            f"write_file, or give the user a final answer).]"
        )

    try:
        text = p.read_text(errors="replace")
    except Exception as e:
        return f"ERROR: {e}"
    lines = text.splitlines()
    s = max(1, start) - 1
    e = len(lines) if end is None else min(len(lines), int(end))
    selected = lines[s:e]
    numbered = "\n".join(f"{i + s + 1:>5}  {ln}" for i, ln in enumerate(selected))
    header = f"{path}  ({len(lines)} lines)"
    out = _truncate(f"{header}\n{numbered}")
    if cache is not None:
        cache[abs_path] = out
    if state is not None:
        files_read = set(getattr(state, "files_read", set()) or set())
        files_read.add(abs_path)
        state.update(files_read=files_read)
    return out


def t_write_file(args: dict, ctx: ToolContext) -> str:
    path = args["path"]
    content = args.get("content")
    if content is None:
        return "ERROR: missing argument 'content'"
    if not isinstance(content, str):
        return f"ERROR: 'content' must be a string, got {type(content).__name__}"
    p = _resolve(path, ctx.root)
    existed = p.exists()
    old_text = p.read_text(errors="replace") if existed else ""
    # SAFETY: refuse to silently empty an existing non-empty file. Small
    # models sometimes call write_file with content="" (forgot to fill it
    # in, or thought it'd 'reset' the file). Require explicit allow_empty.
    if existed and old_text.strip() and not content.strip() and not args.get("allow_empty"):
        return (
            f"ERROR: refusing to write empty content to existing file {path} "
            f"({len(old_text)} bytes). If you really meant to empty it, pass "
            f"allow_empty=true. Otherwise you probably forgot to include the "
            f"new content."
        )
    detail = f"{'overwrite' if existed else 'create'} {path} ({len(content)} bytes)"
    if not ctx.confirm("file write", detail):
        return "ERROR: user denied write"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    _record_edit(ctx, p, old_text, content, "write")
    adds, dels = _diff.stats(old_text, content)
    return f"OK: wrote {path} +{adds} -{dels}"


def _norm_eol(s: str) -> str:
    return s.replace("\r\n", "\n").replace("\r", "\n")


def _fuzzy_span(text: str, old: str):
    """Locate `old` in `text` tolerating per-line trailing-whitespace and
    blank-line drift (the #1 cause of edit_file misses). Returns a unique
    (start_line, end_line) into text.split('\\n'), or None if 0 / ambiguous.
    """
    text_lines = text.split("\n")
    old_lines = old.split("\n")
    while len(old_lines) > 1 and old_lines[-1] == "":
        old_lines = old_lines[:-1]
    while len(old_lines) > 1 and old_lines[0] == "":
        old_lines = old_lines[1:]
    n = len(old_lines)
    if n == 0:
        return None
    norm_old = [ln.rstrip() for ln in old_lines]
    hits = [
        i for i in range(len(text_lines) - n + 1)
        if [ln.rstrip() for ln in text_lines[i:i + n]] == norm_old
    ]
    return (hits[0], hits[0] + n) if len(hits) == 1 else None


def _indent_insensitive_span(text: str, old: str):
    """Last-resort matcher: compare lines after stripping ALL leading and
    trailing whitespace. The model often remembers code at the wrong indent
    (e.g. it copied a snippet that was inside a block but pastes it
    top-level). Returns (start, end, indent_delta) or None. indent_delta is
    how much to re-indent the replacement to match the file's actual
    indentation at the match site (so we don't dedent the new content).
    """
    text_lines = text.split("\n")
    old_lines = old.split("\n")
    while len(old_lines) > 1 and old_lines[-1] == "":
        old_lines = old_lines[:-1]
    while len(old_lines) > 1 and old_lines[0] == "":
        old_lines = old_lines[1:]
    n = len(old_lines)
    if n == 0:
        return None
    strip_old = [ln.strip() for ln in old_lines]
    hits = [
        i for i in range(len(text_lines) - n + 1)
        if [ln.strip() for ln in text_lines[i:i + n]] == strip_old
    ]
    if len(hits) != 1:
        return None
    i = hits[0]

    def leading(s: str) -> str:
        return s[:len(s) - len(s.lstrip())]
    # Use the first non-blank line on each side to gauge the indent delta.
    first_old_idx = next((k for k, ln in enumerate(old_lines) if ln.strip()), 0)
    file_indent = leading(text_lines[i + first_old_idx])
    old_indent = leading(old_lines[first_old_idx])
    return (i, i + n, file_indent, old_indent)


def _closest_region(text: str, old: str) -> str:
    """Show the file region most similar to `old`'s first non-blank line so
    the model can self-correct after a failed match."""
    import difflib
    first = ""
    for ln in old.split("\n"):
        if ln.strip():
            first = ln.strip()
            break
    if not first:
        return ""
    text_lines = text.split("\n")
    best_i, best = 0, 0.0
    for i, ln in enumerate(text_lines):
        score = difflib.SequenceMatcher(None, ln.strip(), first).ratio()
        if score > best:
            best, best_i = score, i
    if best < 0.4:
        return ""
    lo, hi = max(0, best_i - 2), min(len(text_lines), best_i + 6)
    return "\n".join(f"{j + 1:>5}  {text_lines[j]}" for j in range(lo, hi))


def _read_text_robust(p: Path) -> str:
    """Read a text file tolerating BOM and Windows code pages.

    Tries UTF-8 (including BOM) first, then falls back to the platform
    default. Strips a leading BOM if present so byte-for-byte comparison
    against `old_string` from a model (which won't include the BOM) works.
    """
    try:
        text = p.read_text(encoding="utf-8", errors="replace")
    except (UnicodeDecodeError, OSError):
        text = p.read_text(errors="replace")
    if text and text[0] == "﻿":
        text = text[1:]
    return text


def t_edit_file(args: dict, ctx: ToolContext) -> str:
    path = args["path"]
    old = args["old_string"]
    new = args["new_string"]
    replace_all = bool(args.get("replace_all", False))
    p = _resolve(path, ctx.root)
    if not p.exists():
        return f"ERROR: file not found: {path}"
    raw = _read_text_robust(p)

    # 1. Exact match on the raw file — byte-perfect, preserves everything.
    count = raw.count(old)
    if count == 1 or (count > 1 and replace_all):
        if not ctx.confirm("file edit", f"{path}: replace {count} occurrence(s)"):
            return "ERROR: user denied edit"
        new_text = raw.replace(old, new) if replace_all else raw.replace(old, new, 1)
        # SAFETY: refuse to silently empty a non-empty file.
        if raw.strip() and not new_text.strip() and not args.get("allow_empty"):
            return (
                f"ERROR: refusing to empty {path}. The replacement would leave "
                f"the file with no content. If that's intentional, pass "
                f"allow_empty=true. Otherwise double-check your new_string."
            )
        p.write_text(new_text, encoding="utf-8")
        _record_edit(ctx, p, raw, new_text, "edit")
        adds, dels = _diff.stats(raw, new_text)
        return f"OK: edited {path} +{adds} -{dels}"
    if count > 1:
        return f"ERROR: old_string matches {count} times — pass replace_all=true or supply more context"

    # 2. Recovery: normalize line endings + tolerate trailing-whitespace drift.
    text = _norm_eol(raw)
    old_n = _norm_eol(old)
    new_n = _norm_eol(new)
    if text.count(old_n) == 1:
        if not ctx.confirm("file edit", f"{path}: replace 1 occurrence (line-ending normalized)"):
            return "ERROR: user denied edit"
        new_text = text.replace(old_n, new_n, 1)
    else:
        span = _fuzzy_span(text, old_n)
        if span is None:
            # Last-resort recovery: indent-insensitive line match. Re-indents
            # the replacement to match the file's actual indentation so we
            # don't break Python / YAML scoping.
            ii = _indent_insensitive_span(text, old_n)
            if ii is not None:
                i, j, file_indent, old_indent = ii
                if not ctx.confirm("file edit", f"{path}: replace lines {i + 1}-{j} (indent-tolerant match)"):
                    return "ERROR: user denied edit"
                file_lines = text.split("\n")
                new_split = new_n.split("\n")
                # Re-base every replacement line: strip the old common indent,
                # add the file's indent. Blank lines stay blank.
                rebased = []
                for ln in new_split:
                    if not ln.strip():
                        rebased.append("")
                        continue
                    if old_indent and ln.startswith(old_indent):
                        ln = ln[len(old_indent):]
                    rebased.append(file_indent + ln)
                new_text = "\n".join(file_lines[:i] + rebased + file_lines[j:])
                if raw.strip() and not new_text.strip() and not args.get("allow_empty"):
                    return (
                        f"ERROR: refusing to empty {path}. The replacement would leave the "
                        f"file with no content. If that's intentional, pass allow_empty=true."
                    )
                p.write_text(new_text, encoding="utf-8")
                _record_edit(ctx, p, raw, new_text, "edit")
                adds, dels = _diff.stats(text, new_text)
                return f"OK: edited {path} +{adds} -{dels} (indent-recovered)"
            lines = len(text.splitlines())
            # Count consecutive failures on this path this turn — after the
            # second one, escalate hard: keep retrying edit_file won't work,
            # the model needs to switch strategy to write_file.
            state = getattr(ctx, "state", None)
            fail_count = 1
            if state is not None:
                fails = dict(getattr(state, "edit_fails", {}) or {})
                rp = str(p.resolve())
                fails[rp] = fails.get(rp, 0) + 1
                fail_count = fails[rp]
                state.update(edit_fails=fails)
            msg = (
                f"ERROR: old_string not found in {path} (file has {lines} lines). "
                f"old_string must match the file EXACTLY, including indentation and "
                f"whitespace."
            )
            hint = _closest_region(text, old_n)
            if hint:
                msg += f"\n\nClosest region in the file (copy from here, NOT from memory):\n{hint}"
            if fail_count >= 2:
                msg += (
                    f"\n\nESCALATE: this is failure #{fail_count} on {path}. STOP using "
                    f"edit_file on this file. You have TWO good options:\n"
                    f"  (a) replace_lines(path, start_line, end_line, new_content) — "
                    f"surgical line-range edit that doesn't depend on exact string "
                    f"matching. Use grep first to find the line numbers, then call this. "
                    f"This is the right answer when you keep failing on encoding or "
                    f"whitespace mismatch.\n"
                    f"  (b) write_file with the full new content if the change spans "
                    f"too much of the file."
                )
            else:
                msg += (
                    f"\n\nIf this fails again, switch to replace_lines(path, start_line, "
                    f"end_line, new_content) — it does a surgical line-range edit and "
                    f"sidesteps every string-matching problem."
                )
            return msg
        i, j = span
        if not ctx.confirm("file edit", f"{path}: replace lines {i + 1}-{j} (whitespace-tolerant match)"):
            return "ERROR: user denied edit"
        file_lines = text.split("\n")
        new_text = "\n".join(file_lines[:i] + new_n.split("\n") + file_lines[j:])

    # SAFETY: refuse to silently empty a non-empty file (catches both the
    # exact-match and fuzzy-match branches).
    if raw.strip() and not new_text.strip() and not args.get("allow_empty"):
        return (
            f"ERROR: refusing to empty {path}. The replacement would leave the "
            f"file with no content. If that's intentional, pass allow_empty=true."
        )
    p.write_text(new_text, encoding="utf-8")
    _record_edit(ctx, p, raw, new_text, "edit")
    adds, dels = _diff.stats(text, new_text)
    return f"OK: edited {path} +{adds} -{dels}"


def _resolve_edit(
    text: str, old: str, new: str, replace_all: bool
) -> tuple[str | None, str]:
    """Apply ONE old->new substitution to `text`, trying progressively
    looser matches. Returns (new_text, note) on success or (None, reason)
    on failure. Pure — does no IO, so multi_edit can chain it in memory and
    only touch disk once the whole batch resolves."""
    count = text.count(old)
    if count == 1 or (count > 1 and replace_all):
        out = text.replace(old, new) if replace_all else text.replace(old, new, 1)
        return out, "exact"
    if count > 1:
        return None, f"matches {count}× — set replace_all or add surrounding context"

    t, o, n = _norm_eol(text), _norm_eol(old), _norm_eol(new)
    c = t.count(o)
    if c == 1 or (c > 1 and replace_all):
        out = t.replace(o, n) if replace_all else t.replace(o, n, 1)
        return out, "eol-normalized"
    if c > 1:
        return None, f"matches {c}× — set replace_all or add surrounding context"

    span = _fuzzy_span(t, o)
    if span is not None:
        i, j = span
        lines = t.split("\n")
        return "\n".join(lines[:i] + n.split("\n") + lines[j:]), "whitespace-tolerant"

    ii = _indent_insensitive_span(t, o)
    if ii is not None:
        i, j, file_indent, old_indent = ii
        lines = t.split("\n")
        rebased = []
        for ln in n.split("\n"):
            if not ln.strip():
                rebased.append("")
                continue
            if old_indent and ln.startswith(old_indent):
                ln = ln[len(old_indent):]
            rebased.append(file_indent + ln)
        return "\n".join(lines[:i] + rebased + lines[j:]), "indent-recovered"

    return None, "old_string not found"


def t_multi_edit(args: dict, ctx: ToolContext) -> str:
    """Apply MANY edits to a single file in one call. Edits apply in order,
    each against the result of the previous one. The batch is atomic: if any
    edit fails to match, NOTHING is written and the failing edit is named."""
    path = args["path"]
    edits = args.get("edits") or []
    if not isinstance(edits, list) or not edits:
        return "ERROR: 'edits' must be a non-empty list of {old_string, new_string} objects."
    p = _resolve(path, ctx.root)
    if not p.exists():
        return f"ERROR: file not found: {path}"
    raw = _read_text_robust(p)

    text = raw
    notes: list[str] = []
    for idx, e in enumerate(edits, start=1):
        if not isinstance(e, dict):
            return f"ERROR: edit #{idx} is not an object. Nothing written."
        old = e.get("old_string")
        new = e.get("new_string")
        if old is None or new is None:
            return (f"ERROR: edit #{idx} is missing old_string or new_string. "
                    f"Nothing written.")
        new_text, note = _resolve_edit(text, old, new, bool(e.get("replace_all", False)))
        if new_text is None:
            return (
                f"ERROR: edit #{idx} of {len(edits)} failed: {note}.\n"
                f"The batch is ATOMIC — nothing was written. Fix edit #{idx} "
                f"(copy old_string exactly from a recent read_file) and resend the "
                f"whole batch, or drop that edit and apply it separately with "
                f"replace_lines."
            )
        text = new_text
        notes.append(f"#{idx}:{note}")

    if text == raw:
        return f"OK: no-op — all {len(edits)} edits left {path} unchanged."
    if raw.strip() and not text.strip() and not args.get("allow_empty"):
        return (f"ERROR: refusing to empty {path}. If intentional, pass "
                f"allow_empty=true.")
    if not ctx.confirm("file edit", f"{path}: apply {len(edits)} edits in one batch"):
        return "ERROR: user denied edit"
    p.write_text(text, encoding="utf-8")
    _record_edit(ctx, p, raw, text, "edit")
    adds, dels = _diff.stats(raw, text)
    recovered = sum(1 for n in notes if not n.endswith(":exact"))
    tail = f"  ({recovered} fuzzy-matched)" if recovered else ""
    return f"OK: applied {len(edits)} edits to {path} +{adds} -{dels}{tail}"


def t_replace_lines(args: dict, ctx: ToolContext) -> str:
    """Surgical line-range replacement — bypasses string matching entirely.

    Use when edit_file fails because of encoding / quote / indentation
    drift. Reads the file, replaces lines [start_line, end_line] (1-indexed,
    inclusive) with `new_content`, writes it back. Records to /undo history.
    """
    path = args["path"]
    start = args.get("start_line")
    end = args.get("end_line")
    new_content = args.get("new_content", "")
    if start is None or end is None:
        return "ERROR: replace_lines requires start_line and end_line (1-indexed, inclusive)"
    p = _resolve(path, ctx.root)
    if not p.exists():
        return f"ERROR: file not found: {path}"
    raw = _read_text_robust(p)
    lines = raw.splitlines(keepends=True)
    s = max(1, int(start)) - 1
    e = min(len(lines), int(end))
    if s >= len(lines):
        return f"ERROR: start_line {start} is past end of file ({len(lines)} lines)"
    if e < s + 1:
        return f"ERROR: end_line ({end}) must be >= start_line ({start})"
    if not ctx.confirm("file edit (replace_lines)", f"{path}: lines {s + 1}-{e}"):
        return "ERROR: user denied edit"
    # Preserve the file's original line ending if possible.
    eol = "\r\n" if (raw and "\r\n" in raw and "\n" in raw and raw.count("\r\n") >= raw.count("\n") / 2) else "\n"
    new_chunk = new_content
    if not new_chunk.endswith(("\n", "\r")):
        new_chunk += eol
    # Splice
    new_lines = lines[:s] + [new_chunk] + lines[e:]
    new_text = "".join(new_lines)
    # SAFETY: refuse to silently empty a non-empty file.
    if raw.strip() and not new_text.strip() and not args.get("allow_empty"):
        return (
            f"ERROR: refusing to empty {path}. The replacement would leave the "
            f"file with no content. If that's intentional, pass allow_empty=true."
        )
    p.write_text(new_text, encoding="utf-8")
    _record_edit(ctx, p, raw, new_text, "replace_lines")
    adds, dels = _diff.stats(raw, new_text)
    return f"OK: replaced {path}:{s + 1}-{e} +{adds} -{dels}"


def t_list_dir(args: dict, ctx: ToolContext) -> str:
    path = args.get("path", ".")
    p = _resolve(path, ctx.root).resolve()
    if not p.exists():
        return f"ERROR: not found: {path}"
    if not p.is_dir():
        return f"ERROR: not a directory: {path}"
    rows = []
    for entry in sorted(p.iterdir(), key=lambda e: (not e.is_dir(), e.name)):
        if entry.name.startswith("."):
            continue
        kind = "dir " if entry.is_dir() else "file"
        try:
            size = entry.stat().st_size if entry.is_file() else 0
        except OSError:
            size = 0
        rows.append(f"  {kind}  {size:>9}  {entry.name}")
    header = f"{path}  ({len(rows)} entries)"
    body = "\n".join(rows) if rows else "(empty or only dotfiles)"
    hint = ""
    try:
        outside = (p != ctx.root) and (ctx.root not in p.parents) and (p not in ctx.root.parents)
    except Exception:
        outside = False
    if outside:
        hint = (
            f"\n\nNOTE: this directory ({p}) is OUTSIDE the current workspace ({ctx.root}). "
            f"To read or edit files inside it with relative paths, first call "
            f"set_workspace with path={p}. Otherwise use absolute paths like {p}/<file>."
        )
    return _truncate(f"{header}\n{body}{hint}")


def t_grep(args: dict, ctx: ToolContext) -> str:
    pattern = args.get("pattern") or args.get("query") or args.get("regex")
    if not pattern:
        return "ERROR: missing argument 'pattern'"
    path = args.get("path", ".")
    case_insensitive = bool(args.get("case_insensitive", False))
    p = _resolve(path, ctx.root)

    # Refuse absurdly broad targets — a grep across the user's whole home dir
    # eats minutes and floods the context with irrelevant matches (e.g. node
    # node_modules, ~/.cache/huggingface). The model must narrow the search.
    home = Path.home().resolve()
    target = p.resolve()
    if target == home or target == Path(home.anchor):
        return (
            f"ERROR: grep target {p} is your home directory — too broad. "
            f"Specify a narrower path (a project subdirectory). For example: "
            f"grep pattern='{pattern}' path='{home}/<project>'."
        )

    # Fast path: if ripgrep is installed, use it. Way faster than Python regex
    # walks on big trees, and it respects .gitignore by default.
    import shutil as _sh
    rg = _sh.which("rg")
    if rg and p.exists():
        cmd = [rg, "-n", "--no-heading", "--color=never", "-m", "200"]
        if case_insensitive:
            cmd.append("-i")
        cmd.extend(["--", pattern, str(p)])
        try:
            proc = subprocess.run(cmd, capture_output=True,
                                  encoding="utf-8", errors="replace",
                                  timeout=30)
        except subprocess.TimeoutExpired:
            return "ERROR: rg timed out after 30s"
        if proc.returncode in (0, 1):  # 0=hits, 1=no hits (not an error)
            out = proc.stdout.strip()
            return _truncate(out) if out else "(no matches)"
        # Unexpected rg failure — fall through to the Python implementation.

    flags = re.IGNORECASE if case_insensitive else 0
    try:
        rx = re.compile(pattern, flags)
    except re.error as e:
        return f"ERROR: bad regex: {e}"
    skip_dirs = {".git", "node_modules", "__pycache__", ".venv", "venv", "dist", "build"}
    matches: list[str] = []
    targets = [p] if p.is_file() else list(p.rglob("*")) if p.is_dir() else []
    scanned = 0
    MAX_SCAN = 2000  # files to crack open before giving up; protects against home-dir-wide greps
    for f in targets:
        if not f.is_file():
            continue
        if any(part in skip_dirs for part in f.parts):
            continue
        # Skip noisy cache/data dirs the model rarely cares about.
        if any(s in f.parts for s in (".cache", "site-packages", ".tox", ".pytest_cache")):
            continue
        scanned += 1
        if scanned > MAX_SCAN:
            matches.append(f"…[scanned {MAX_SCAN}+ files, stopping — narrow the path]")
            return _truncate("\n".join(matches))
        try:
            with f.open("r", errors="replace") as fp:
                for i, line in enumerate(fp, 1):
                    if rx.search(line):
                        rel = f.relative_to(ctx.root) if f.is_relative_to(ctx.root) else f
                        matches.append(f"{rel}:{i}: {line.rstrip()}")
                        if len(matches) >= 200:
                            matches.append("…[200 match cap]")
                            return _truncate("\n".join(matches))
        except (OSError, UnicodeDecodeError):
            continue
    return _truncate("\n".join(matches) if matches else "(no matches)")


def t_set_workspace(args: dict, ctx: ToolContext) -> str:
    path = args["path"]
    create = bool(args.get("create", False))
    p = _resolve(path, ctx.root)
    if not p.exists():
        if not create:
            return f"ERROR: directory does not exist: {p}. Pass create=true to mkdir it."
        p.mkdir(parents=True, exist_ok=True)
    elif not p.is_dir():
        return f"ERROR: not a directory: {p}"
    ctx.root = p.resolve()
    return f"OK: workspace set to {ctx.root}"


def t_run_bash(args: dict, ctx: ToolContext) -> str:
    cmd = args["command"]
    timeout = int(args.get("timeout", 60))
    if not ctx.confirm("shell command", cmd):
        return "ERROR: user denied command"
    try:
        with ui.Spinner(f"running: {cmd[:40] + ('…' if len(cmd) > 40 else '')}"):
            proc = subprocess.run(
                cmd, shell=True, cwd=str(ctx.root),
                capture_output=True,
                encoding="utf-8", errors="replace", timeout=timeout,
            )
    except subprocess.TimeoutExpired:
        return f"ERROR: timed out after {timeout}s — the command was still running."
    out = proc.stdout
    err = proc.stderr
    status = "PASS" if proc.returncode == 0 else "FAIL"
    parts = [f"{status} (exit code {proc.returncode})"]
    if out:
        parts.append(f"--- stdout ---\n{out}")
    if err:
        parts.append(f"--- stderr ---\n{err}")
    if proc.returncode != 0:
        hint = _analyze_failure(out, err)
        if hint:
            parts.append(hint)
    return _truncate("\n".join(parts))


TOOLS = {
    "read_file":     t_read_file,
    "write_file":    t_write_file,
    "edit_file":     t_edit_file,
    "multi_edit":    t_multi_edit,
    "replace_lines": t_replace_lines,
    "list_dir":      t_list_dir,
    "grep":          t_grep,
    "set_workspace": t_set_workspace,
    "run_bash":      t_run_bash,
}


TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a UTF-8 text file from the workspace. Returns line-numbered content.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path (absolute or relative to cwd)."},
                    "start_line": {"type": "integer", "description": "1-indexed start line. Default 1."},
                    "end_line": {"type": "integer", "description": "Inclusive end line. Default end of file."},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Create or overwrite a file with the given content. Requires user approval.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": "Replace an exact string in a file. By default old_string must be unique; set replace_all to replace every occurrence. For 2+ edits to the SAME file, use multi_edit instead — one call, not many.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "old_string": {"type": "string"},
                    "new_string": {"type": "string"},
                    "replace_all": {"type": "boolean"},
                },
                "required": ["path", "old_string", "new_string"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "multi_edit",
            "description": (
                "Apply MANY edits to ONE file in a single call. Strongly "
                "preferred over repeated edit_file calls when a task touches "
                "several spots in the same file (renames, palette swaps, "
                "multi-site refactors). Edits apply in order, each against the "
                "result of the previous one. Atomic: if any edit fails to "
                "match, nothing is written and the failing edit is named so "
                "you can fix just that one. Collect ALL the changes you intend "
                "for a file and send them together."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File to edit."},
                    "edits": {
                        "type": "array",
                        "description": "Ordered list of edits to apply.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "old_string": {"type": "string"},
                                "new_string": {"type": "string"},
                                "replace_all": {"type": "boolean"},
                            },
                            "required": ["old_string", "new_string"],
                        },
                    },
                },
                "required": ["path", "edits"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "replace_lines",
            "description": "Surgical line-range replacement. Use when edit_file fails on string matching (encoding / quote / indent drift). Lines are 1-indexed, inclusive on both ends. Use grep to find the lines first, then call this.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "start_line": {"type": "integer"},
                    "end_line": {"type": "integer"},
                    "new_content": {"type": "string"},
                },
                "required": ["path", "start_line", "end_line", "new_content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_dir",
            "description": "List entries in a directory (skips dotfiles).",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string", "description": "Default '.'"}},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "grep",
            "description": "Recursive regex search over the workspace. Skips .git, node_modules, build artifacts.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string"},
                    "path": {"type": "string", "description": "Default '.'"},
                    "case_insensitive": {"type": "boolean"},
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_workspace",
            "description": "Change the workspace directory used to resolve relative paths. Use this immediately after creating a new project folder so subsequent file writes go inside it.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Directory to switch to. Absolute or ~-path recommended."},
                    "create": {"type": "boolean", "description": "Create the directory if it does not exist. Default false."},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_bash",
            "description": "Run a shell command in the workspace. Requires user approval.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                    "timeout": {"type": "integer", "description": "Seconds. Default 60."},
                },
                "required": ["command"],
            },
        },
    },
]
