"""Tool implementations the model can call.

Every tool receives a dict of arguments and returns a string result that gets
fed back to the model as the tool's output.
"""
from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from . import ui

MAX_OUTPUT_CHARS = 16000


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


@dataclass
class ToolContext:
    root: Path
    yolo: bool = False
    github_token: str | None = None

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


# ---------- tool implementations ----------

def t_read_file(args: dict, ctx: ToolContext) -> str:
    path = args["path"]
    start = int(args.get("start_line", 1))
    end = args.get("end_line")
    p = _resolve(path, ctx.root)
    if not p.exists():
        return f"ERROR: file not found: {path}"
    if not p.is_file():
        return f"ERROR: not a file: {path}"
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
    return _truncate(f"{header}\n{numbered}")


def t_write_file(args: dict, ctx: ToolContext) -> str:
    from . import diff as _diff
    path = args["path"]
    content = args["content"]
    p = _resolve(path, ctx.root)
    existed = p.exists()
    old_text = p.read_text(errors="replace") if existed else ""
    detail = f"{'overwrite' if existed else 'create'} {path} ({len(content)} bytes)"
    if not ctx.confirm("file write", detail):
        return "ERROR: user denied write"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)
    rendered = _diff.render(old_text, content, path)
    if rendered:
        print(rendered)
    adds, dels = _diff.stats(old_text, content)
    return f"OK: wrote {path} ({'overwrote' if existed else 'created'}, +{adds} -{dels} lines)"


def t_edit_file(args: dict, ctx: ToolContext) -> str:
    path = args["path"]
    old = args["old_string"]
    new = args["new_string"]
    replace_all = bool(args.get("replace_all", False))
    p = _resolve(path, ctx.root)
    if not p.exists():
        return f"ERROR: file not found: {path}"
    text = p.read_text(errors="replace")
    count = text.count(old)
    if count == 0:
        return "ERROR: old_string not found"
    if count > 1 and not replace_all:
        return f"ERROR: old_string matches {count} times — pass replace_all=true or supply more context"
    if not ctx.confirm("file edit", f"{path}: replace {count} occurrence(s)"):
        return "ERROR: user denied edit"
    new_text = text.replace(old, new) if replace_all else text.replace(old, new, 1)
    p.write_text(new_text)
    from . import diff as _diff
    rendered = _diff.render(text, new_text, path)
    if rendered:
        print(rendered)
    adds, dels = _diff.stats(text, new_text)
    return f"OK: edited {path} ({count} replacement(s), +{adds} -{dels} lines)"


def t_list_dir(args: dict, ctx: ToolContext) -> str:
    path = args.get("path", ".")
    p = _resolve(path, ctx.root)
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
    return _truncate(f"{path}\n" + "\n".join(rows) if rows else f"{path} (empty)")


def t_grep(args: dict, ctx: ToolContext) -> str:
    pattern = args["pattern"]
    path = args.get("path", ".")
    case_insensitive = bool(args.get("case_insensitive", False))
    p = _resolve(path, ctx.root)
    flags = re.IGNORECASE if case_insensitive else 0
    try:
        rx = re.compile(pattern, flags)
    except re.error as e:
        return f"ERROR: bad regex: {e}"
    skip_dirs = {".git", "node_modules", "__pycache__", ".venv", "venv", "dist", "build"}
    matches: list[str] = []
    targets = [p] if p.is_file() else list(p.rglob("*")) if p.is_dir() else []
    for f in targets:
        if not f.is_file():
            continue
        if any(part in skip_dirs for part in f.parts):
            continue
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


def t_run_bash(args: dict, ctx: ToolContext) -> str:
    cmd = args["command"]
    timeout = int(args.get("timeout", 60))
    if not ctx.confirm("shell command", cmd):
        return "ERROR: user denied command"
    try:
        proc = subprocess.run(
            cmd, shell=True, cwd=str(ctx.root),
            capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return f"ERROR: timed out after {timeout}s"
    out = proc.stdout
    err = proc.stderr
    parts = [f"exit code: {proc.returncode}"]
    if out:
        parts.append(f"--- stdout ---\n{out}")
    if err:
        parts.append(f"--- stderr ---\n{err}")
    return _truncate("\n".join(parts))


# ---------- registry ----------

ToolFn = Callable[[dict, ToolContext], str]

TOOLS: dict[str, ToolFn] = {
    "read_file": t_read_file,
    "write_file": t_write_file,
    "edit_file": t_edit_file,
    "list_dir": t_list_dir,
    "grep": t_grep,
    "run_bash": t_run_bash,
}


# Schema sent to Ollama (OpenAI-style function definitions).
TOOL_SCHEMAS: list[dict[str, Any]] = [
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
            "description": "Replace an exact string in a file. By default old_string must be unique; set replace_all to replace every occurrence.",
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


def _all_tools() -> dict[str, ToolFn]:
    from .github import GITHUB_TOOLS
    return {**TOOLS, **GITHUB_TOOLS}


def all_tool_schemas() -> list[dict]:
    from .github import GITHUB_TOOL_SCHEMAS
    return TOOL_SCHEMAS + GITHUB_TOOL_SCHEMAS


def dispatch(name: str, args: dict, ctx: ToolContext) -> str:
    fn = _all_tools().get(name)
    if fn is None:
        return f"ERROR: unknown tool '{name}'"
    try:
        return fn(args, ctx)
    except KeyError as e:
        return f"ERROR: missing argument {e}"
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}"
