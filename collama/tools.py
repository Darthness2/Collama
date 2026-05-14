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
    insecure_ssl: bool = False
    # Optional plumbing for engine-aware tools (subagent, worktree,
    # background, tasks). These are populated by the QueryEngine when it
    # builds the ctx; pure-FS tools ignore them.
    state: object | None = None
    engine: object | None = None
    background: object | None = None
    tasks: object | None = None
    teams: object | None = None

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


def t_edit_file(args: dict, ctx: ToolContext) -> str:
    from . import diff as _diff
    path = args["path"]
    old = args["old_string"]
    new = args["new_string"]
    replace_all = bool(args.get("replace_all", False))
    p = _resolve(path, ctx.root)
    if not p.exists():
        return f"ERROR: file not found: {path}"
    raw = p.read_text(errors="replace")

    # 1. Exact match on the raw file — byte-perfect, preserves everything.
    count = raw.count(old)
    if count == 1 or (count > 1 and replace_all):
        if not ctx.confirm("file edit", f"{path}: replace {count} occurrence(s)"):
            return "ERROR: user denied edit"
        new_text = raw.replace(old, new) if replace_all else raw.replace(old, new, 1)
        p.write_text(new_text)
        rendered = _diff.render(raw, new_text, path)
        if rendered:
            print(rendered)
        adds, dels = _diff.stats(raw, new_text)
        return f"OK: edited {path} ({count} replacement(s), +{adds} -{dels} lines)"
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
            lines = len(text.splitlines())
            msg = (
                f"ERROR: old_string not found in {path} (file has {lines} lines). "
                f"old_string must match the file EXACTLY, including indentation and "
                f"whitespace. Re-read the file with read_file and copy the exact text, "
                f"or use write_file to replace the whole file."
            )
            hint = _closest_region(text, old_n)
            if hint:
                msg += f"\n\nClosest region in the file:\n{hint}"
            return msg
        i, j = span
        if not ctx.confirm("file edit", f"{path}: replace lines {i + 1}-{j} (whitespace-tolerant match)"):
            return "ERROR: user denied edit"
        file_lines = text.split("\n")
        new_text = "\n".join(file_lines[:i] + new_n.split("\n") + file_lines[j:])

    p.write_text(new_text)
    rendered = _diff.render(text, new_text, path)
    if rendered:
        print(rendered)
    adds, dels = _diff.stats(text, new_text)
    return f"OK: edited {path} (+{adds} -{dels} lines, recovered via fuzzy match)"


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


# -------------------------------------------------------------- s12 worktree

def t_enter_worktree(args: dict, ctx: ToolContext) -> str:
    path = args["path"]
    create = bool(args.get("create", False))
    p = _resolve(path, ctx.root)
    if not p.exists():
        if not create:
            return f"ERROR: worktree dir not found: {p}. Pass create=true to mkdir it."
        p.mkdir(parents=True, exist_ok=True)
    elif not p.is_dir():
        return f"ERROR: not a directory: {p}"
    state = getattr(ctx, "state", None)
    if state is None:
        ctx.root = p.resolve()
        return f"OK: workspace set to {p} (no state for stack)"
    stack = list(getattr(state, "worktree_stack", []) or [])
    stack.append(str(ctx.root))
    state.update(worktree_stack=stack, workspace=p.resolve())
    ctx.root = p.resolve()
    return f"OK: entered worktree {p}  (stack depth {len(stack)})"


def t_exit_worktree(args: dict, ctx: ToolContext) -> str:
    state = getattr(ctx, "state", None)
    if state is None:
        return "ERROR: worktree stack not available"
    stack = list(getattr(state, "worktree_stack", []) or [])
    if not stack:
        return "ERROR: worktree stack is empty (no enter_worktree to pop)"
    prev = stack.pop()
    state.update(worktree_stack=stack, workspace=Path(prev))
    ctx.root = Path(prev)
    return f"OK: exited worktree, back to {prev}"


# -------------------------------------------------------------- s07 tasks

def _tasks(ctx: ToolContext):
    return getattr(ctx, "tasks", None)


def t_task_create(args: dict, ctx: ToolContext) -> str:
    tg = _tasks(ctx)
    if tg is None:
        return "ERROR: task graph not available"
    title = args["title"]
    task = tg.create(
        title=title,
        description=args.get("description", ""),
        deps=args.get("deps") or [],
        parent_id=args.get("parent_id"),
        worktree=args.get("worktree"),
    )
    return f"OK: created {task.id}  {task.short()}"


def t_task_update(args: dict, ctx: ToolContext) -> str:
    tg = _tasks(ctx)
    if tg is None:
        return "ERROR: task graph not available"
    tid = args["id"]
    changes = {k: v for k, v in args.items() if k != "id"}
    task = tg.update(tid, **changes)
    if not task:
        return f"ERROR: no task with id {tid}"
    return f"OK: updated {task.short()}"


def t_task_get(args: dict, ctx: ToolContext) -> str:
    tg = _tasks(ctx)
    if tg is None:
        return "ERROR: task graph not available"
    task = tg.get(args["id"])
    if not task:
        return f"ERROR: no task with id {args['id']}"
    import json as _json
    return _json.dumps(task.to_dict(), indent=2)


def t_task_list(args: dict, ctx: ToolContext) -> str:
    tg = _tasks(ctx)
    if tg is None:
        return "ERROR: task graph not available"
    tasks = tg.list(status=args.get("status"), parent_id=args.get("parent_id"))
    if not tasks:
        return "(no tasks)"
    return "\n".join(t.short() for t in tasks[:60])


def t_task_delete(args: dict, ctx: ToolContext) -> str:
    tg = _tasks(ctx)
    if tg is None:
        return "ERROR: task graph not available"
    return "OK: deleted" if tg.delete(args["id"]) else f"ERROR: no task {args['id']}"


# -------------------------------------------------------------- s08 background

def t_bash_async(args: dict, ctx: ToolContext) -> str:
    bg = getattr(ctx, "background", None)
    if bg is None:
        return "ERROR: background executor not available"
    cmd = args["command"]
    timeout = int(args.get("timeout", 600))
    if not ctx.confirm("background shell command", cmd):
        return "ERROR: user denied command"
    job_id = bg.submit_bash(cmd, ctx.root, timeout=timeout)
    return f"OK: queued {job_id}  (poll with task_status, or wait with task_wait)"


def t_task_status(args: dict, ctx: ToolContext) -> str:
    bg = getattr(ctx, "background", None)
    if bg is None:
        return "ERROR: background executor not available"
    job = bg.status(args["task_id"])
    if not job:
        return f"ERROR: no background job {args['task_id']}"
    summary = job.result.splitlines()[0][:160] if job.result else ""
    return f"{job.id}  status={job.status}  kind={job.kind}  label={job.label[:80]}\n{summary}"


def t_task_wait(args: dict, ctx: ToolContext) -> str:
    bg = getattr(ctx, "background", None)
    if bg is None:
        return "ERROR: background executor not available"
    timeout = float(args.get("timeout", 60))
    job = bg.wait(args["task_id"], timeout=timeout)
    if not job:
        return f"ERROR: no background job {args['task_id']}"
    if job.status == "running":
        return f"still running after {timeout}s — try again with longer timeout"
    return _truncate(f"{job.id} finished {job.status}\n\n{job.result}")


# -------------------------------------------------------------- s04 sub-agent

def t_agent_call(args: dict, ctx: ToolContext) -> str:
    """Fork a sub-agent on a fresh messages[]; return its final answer."""
    engine = getattr(ctx, "engine", None)
    if engine is None:
        return "ERROR: engine not available for sub-agent"
    from .subagent import fork_subagent
    prompt = args["prompt"]
    model = args.get("model")
    answer = fork_subagent(engine, prompt, model=model, title=args.get("title", "subagent"))
    return _truncate(f"[sub-agent answer]\n{answer}")


def t_agent_call_async(args: dict, ctx: ToolContext) -> str:
    """Fork a sub-agent in the BACKGROUND; result arrives later as a notification."""
    engine = getattr(ctx, "engine", None)
    bg = getattr(ctx, "background", None)
    if engine is None or bg is None:
        return "ERROR: engine/background not available"
    from .subagent import fork_subagent
    prompt = args["prompt"]
    model = args.get("model")

    def _run(p):
        return fork_subagent(engine, p, model=model)

    job_id = bg.submit_dream(prompt, _run)
    return f"OK: dream {job_id} dispatched (will surface on completion)"


# -------------------------------------------------------------- s09 teams

def _teams(ctx: ToolContext):
    return getattr(ctx, "teams", None)


def t_team_create(args: dict, ctx: ToolContext) -> str:
    reg = _teams(ctx)
    if reg is None:
        return "ERROR: team registry not available"
    name = args["name"]
    reg.create_team(name)
    return f"OK: team '{name}' ready at {reg.root / name}"


def t_team_delete(args: dict, ctx: ToolContext) -> str:
    reg = _teams(ctx)
    if reg is None:
        return "ERROR: team registry not available"
    name = args["name"]
    if not ctx.confirm("delete team", f"team {name} and all its teammates"):
        return "ERROR: user denied"
    return "OK: deleted" if reg.delete_team(name) else f"ERROR: no team {name}"


def t_team_list(args: dict, ctx: ToolContext) -> str:
    reg = _teams(ctx)
    if reg is None:
        return "ERROR: team registry not available"
    teams = reg.list_teams()
    if not teams:
        return "(no teams)"
    out: list[str] = []
    for t in teams:
        members = reg.list_teammates(t)
        out.append(f"{t}  ({len(members)} member{'s' if len(members) != 1 else ''})")
        for m in members:
            out.append(f"  - {m.short()}")
    return "\n".join(out)


def t_teammate_create(args: dict, ctx: ToolContext) -> str:
    reg = _teams(ctx)
    if reg is None:
        return "ERROR: team registry not available"
    tm = reg.add_teammate(
        team=args["team"],
        name=args["name"],
        role=args.get("role", ""),
        skills=args.get("skills") or [],
    )
    return f"OK: created teammate {tm.short()}"


def t_teammate_delete(args: dict, ctx: ToolContext) -> str:
    reg = _teams(ctx)
    if reg is None:
        return "ERROR: team registry not available"
    return ("OK: deleted"
            if reg.delete_teammate(args["team"], args["name"])
            else f"ERROR: no teammate {args['name']} on {args['team']}")


def t_teammate_list(args: dict, ctx: ToolContext) -> str:
    reg = _teams(ctx)
    if reg is None:
        return "ERROR: team registry not available"
    members = reg.list_teammates(args.get("team"))
    if not members:
        return "(no teammates)"
    return "\n".join(m.short() for m in members)


# -------------------------------------------------------------- s10 protocols

def t_send_message(args: dict, ctx: ToolContext) -> str:
    """SendMessageTool — request/response across teammates via mailboxes."""
    reg = _teams(ctx)
    if reg is None:
        return "ERROR: team registry not available"
    team = args["team"]
    to = args["to"]
    sender = args.get("from", "lead")
    content = args["content"]
    kind = args.get("kind", "msg")
    tm = reg.deliver(team, to, sender, content, kind=kind)
    if tm is None:
        return f"ERROR: no teammate {to} on team {team}"
    return f"OK: delivered to {team}/{tm.name}  (inbox now {len(tm.mailbox)})"


def t_inbox(args: dict, ctx: ToolContext) -> str:
    reg = _teams(ctx)
    if reg is None:
        return "ERROR: team registry not available"
    tm = reg.get_teammate(args["team"], args["name"])
    if not tm:
        return f"ERROR: no teammate {args['name']} on {args['team']}"
    if not tm.mailbox:
        return f"{tm.team}/{tm.name}: inbox empty"
    out = [f"{tm.team}/{tm.name}: {len(tm.mailbox)} message(s)"]
    for i, m in enumerate(tm.mailbox, 1):
        head = (m.get("content") or "").splitlines()[0][:160]
        out.append(f"  {i}. [{m.get('kind','msg')}] from {m.get('from','?')}: {head}")
    return "\n".join(out)


# -------------------------------------------------------------- s11 coordinator

def t_coordinator_tick(args: dict, ctx: ToolContext) -> str:
    engine = getattr(ctx, "engine", None)
    if engine is None:
        return "ERROR: engine not available"
    from .coordinator import tick
    results = tick(
        engine,
        team=args.get("team"),
        auto_claim=bool(args.get("auto_claim", False)),
        max_per_teammate=int(args.get("max_per_teammate", 1)),
    )
    if not results:
        return "(no teammates with pending mail or claimable tasks)"
    out = [f"processed {len(results)} teammate(s):"]
    for r in results:
        first = r.answer.splitlines()[0][:140] if r.answer else ""
        claimed = f"  claimed={r.claimed_task_id}" if r.claimed_task_id else ""
        out.append(f"  - {r.teammate}  inbox={r.inbox_count}{claimed}")
        if first:
            out.append(f"      → {first}")
    return "\n".join(out)


def t_coordinator_run(args: dict, ctx: ToolContext) -> str:
    """Tick repeatedly until everyone is idle (or `max_rounds` reached)."""
    engine = getattr(ctx, "engine", None)
    if engine is None:
        return "ERROR: engine not available"
    from .coordinator import tick
    max_rounds = int(args.get("max_rounds", 5))
    auto_claim = bool(args.get("auto_claim", True))
    team = args.get("team")
    rounds: list[str] = []
    for r in range(1, max_rounds + 1):
        results = tick(engine, team=team, auto_claim=auto_claim)
        if not results:
            break
        rounds.append(f"round {r}: processed {len(results)} teammate(s)")
    return "\n".join(rounds) if rounds else "(idle — nothing to do)"


# ============================================================================
# Extended toolset: file/search/web/system/skills/planning/interaction
# ============================================================================

# -------------------------------------------------------------- search/glob

def t_glob(args: dict, ctx: ToolContext) -> str:
    """GlobTool — file pattern matching (supports **)."""
    import fnmatch
    pattern = args["pattern"]
    base = _resolve(args.get("path", "."), ctx.root)
    if not base.exists() or not base.is_dir():
        return f"ERROR: not a directory: {base}"
    skip_dirs = {".git", "node_modules", "__pycache__", ".venv", "venv", "dist", "build"}
    matches: list[str] = []
    if "**" in pattern or "/" in pattern:
        for p in base.rglob("*"):
            if any(part in skip_dirs for part in p.parts):
                continue
            rel = p.relative_to(base)
            if fnmatch.fnmatch(str(rel), pattern):
                matches.append(str(rel))
    else:
        for p in base.iterdir():
            if fnmatch.fnmatch(p.name, pattern):
                matches.append(p.name)
    matches.sort()
    return _truncate("\n".join(matches[:500]) if matches else "(no matches)")


def t_tool_search(args: dict, ctx: ToolContext) -> str:
    """ToolSearchTool — find tools by keyword in name or description."""
    q = (args.get("query") or "").lower().strip()
    schemas = TOOL_SCHEMAS
    out: list[str] = []
    for s in schemas:
        fn = s.get("function") or {}
        name = fn.get("name", "")
        desc = fn.get("description", "")
        hay = f"{name} {desc}".lower()
        if not q or q in hay:
            out.append(f"{name}  —  {desc.splitlines()[0][:140] if desc else ''}")
    return _truncate("\n".join(out) if out else "(no matching tools)")


# -------------------------------------------------------------- powershell

def t_powershell(args: dict, ctx: ToolContext) -> str:
    """PowerShellTool — run a command via pwsh / powershell.exe."""
    import shutil as _shutil
    pwsh = _shutil.which("pwsh") or _shutil.which("powershell")
    if not pwsh:
        return "ERROR: PowerShell not installed (pwsh / powershell.exe not on PATH)"
    cmd = args["command"]
    timeout = int(args.get("timeout", 60))
    if not ctx.confirm("PowerShell command", cmd):
        return "ERROR: user denied command"
    try:
        proc = subprocess.run(
            [pwsh, "-NoProfile", "-NonInteractive", "-Command", cmd],
            cwd=str(ctx.root), capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return f"ERROR: timed out after {timeout}s"
    parts = [f"exit code: {proc.returncode}"]
    if proc.stdout:
        parts.append(f"--- stdout ---\n{proc.stdout}")
    if proc.stderr:
        parts.append(f"--- stderr ---\n{proc.stderr}")
    return _truncate("\n".join(parts))


# -------------------------------------------------------------- web

def t_web_fetch(args: dict, ctx: ToolContext) -> str:
    """WebFetchTool — fetch a URL and return text (HTML or JSON)."""
    import requests
    url = args["url"]
    if not url.startswith(("http://", "https://")):
        return "ERROR: url must be http:// or https://"
    timeout = int(args.get("timeout", 20))
    max_bytes = int(args.get("max_bytes", 200_000))
    headers = {"User-Agent": "collama/0.1 (+https://github.com/Darthness2/Collama)"}
    try:
        r = requests.get(url, timeout=timeout, headers=headers,
                         verify=not ctx.insecure_ssl, stream=True)
    except requests.RequestException as e:
        return f"ERROR: fetch failed: {e}"
    chunks: list[bytes] = []
    seen = 0
    for chunk in r.iter_content(8192):
        chunks.append(chunk)
        seen += len(chunk)
        if seen >= max_bytes:
            break
    raw = b"".join(chunks)
    try:
        body = raw.decode(r.encoding or "utf-8", errors="replace")
    except Exception:
        body = raw.decode("utf-8", errors="replace")
    return _truncate(f"HTTP {r.status_code}  {url}\n{body}")


def t_web_search(args: dict, ctx: ToolContext) -> str:
    """WebSearchTool — DuckDuckGo HTML-frontend scrape (no API key needed)."""
    import re as _re
    import requests
    q = args["query"]
    n = int(args.get("limit", 10))
    try:
        r = requests.get(
            "https://duckduckgo.com/html/", params={"q": q},
            headers={"User-Agent": "Mozilla/5.0 collama/0.1"},
            timeout=15, verify=not ctx.insecure_ssl,
        )
    except requests.RequestException as e:
        return f"ERROR: search failed: {e}"
    if r.status_code != 200:
        return f"ERROR: HTTP {r.status_code}"
    # Lightweight parse — DuckDuckGo HTML wraps results in <a class="result__a">.
    rx = _re.compile(r'<a[^>]*class="[^"]*result__a[^"]*"[^>]*href="([^"]+)"[^>]*>(.*?)</a>', _re.DOTALL)
    items = rx.findall(r.text)
    out: list[str] = []
    for href, title in items[:n]:
        title_text = _re.sub(r"<[^>]+>", "", title).strip()
        out.append(f"- {title_text}\n  {href}")
    return _truncate("\n".join(out) if out else "(no results)")


# -------------------------------------------------------------- notebook

def t_notebook_edit(args: dict, ctx: ToolContext) -> str:
    """NotebookEditTool — insert/replace/delete cells in a Jupyter .ipynb file."""
    import json as _json
    path = args["path"]
    op = args.get("op", "replace")  # replace | insert | delete | get
    cell_index = args.get("cell_index")
    new_source = args.get("source", "")
    cell_type = args.get("cell_type", "code")  # code | markdown | raw
    p = _resolve(path, ctx.root)
    if not p.exists():
        if op == "insert" and not cell_index:
            p.parent.mkdir(parents=True, exist_ok=True)
            nb = {"cells": [], "metadata": {}, "nbformat": 4, "nbformat_minor": 5}
        else:
            return f"ERROR: file not found: {path}"
    else:
        try:
            nb = _json.loads(p.read_text())
        except _json.JSONDecodeError as e:
            return f"ERROR: invalid notebook JSON: {e}"
    cells = nb.setdefault("cells", [])

    if op == "get":
        if cell_index is None:
            return _truncate("\n\n".join(
                f"# cell {i} [{c.get('cell_type','?')}]\n" +
                ("".join(c.get("source", [])) if isinstance(c.get("source"), list) else (c.get("source") or ""))
                for i, c in enumerate(cells)
            ))
        i = int(cell_index)
        if not 0 <= i < len(cells):
            return f"ERROR: cell_index {i} out of range"
        c = cells[i]
        return f"cell {i} [{c.get('cell_type','?')}]\n" + (
            "".join(c.get("source", [])) if isinstance(c.get("source"), list) else (c.get("source") or ""))

    if not ctx.confirm("notebook edit", f"{op} {path} [cell {cell_index}]"):
        return "ERROR: user denied"

    if op == "replace":
        i = int(cell_index)
        if not 0 <= i < len(cells):
            return f"ERROR: cell_index {i} out of range"
        cells[i]["source"] = new_source
        if cell_type:
            cells[i]["cell_type"] = cell_type
    elif op == "insert":
        i = len(cells) if cell_index is None else int(cell_index)
        new_cell = {"cell_type": cell_type, "source": new_source, "metadata": {}}
        if cell_type == "code":
            new_cell["execution_count"] = None
            new_cell["outputs"] = []
        cells.insert(max(0, min(i, len(cells))), new_cell)
    elif op == "delete":
        i = int(cell_index)
        if not 0 <= i < len(cells):
            return f"ERROR: cell_index {i} out of range"
        del cells[i]
    else:
        return f"ERROR: unknown op '{op}'"

    p.write_text(_json.dumps(nb, indent=1))
    return f"OK: {op} on {path} (now {len(cells)} cells)"


# -------------------------------------------------------------- interaction

def t_ask_user_question(args: dict, ctx: ToolContext) -> str:
    """AskUserQuestionTool — pause and ask the user."""
    question = args["question"]
    options = args.get("options") or []
    if ctx.yolo:
        return "ERROR: cannot ask user in yolo mode"
    print()
    from . import ui
    ui.warn("? " + question)
    for i, opt in enumerate(options, 1):
        print(f"  {i}. {opt}")
    try:
        ans = input("  > ").strip()
    except EOFError:
        return "ERROR: no tty"
    if options and ans.isdigit():
        idx = int(ans) - 1
        if 0 <= idx < len(options):
            return options[idx]
    return ans or "(empty)"


def t_brief(args: dict, ctx: ToolContext) -> str:
    """BriefTool — store/retrieve a short markdown brief on the session state."""
    state = getattr(ctx, "state", None)
    if state is None:
        return "ERROR: state not available"
    op = args.get("op", "set")
    name = args["name"]
    if op == "get":
        return state.briefs.get(name, f"(no brief named '{name}')")
    if op == "list":
        return "\n".join(f"- {k}  ({len(v)} chars)" for k, v in state.briefs.items()) or "(no briefs)"
    if op == "delete":
        state.briefs.pop(name, None)
        state.update(briefs=dict(state.briefs))
        return f"OK: deleted brief '{name}'"
    # set
    text = args.get("content") or ""
    state.briefs[name] = text
    state.update(briefs=dict(state.briefs))
    return f"OK: brief '{name}' saved ({len(text)} chars)"


# -------------------------------------------------------------- planning

def t_enter_plan_mode(args: dict, ctx: ToolContext) -> str:
    """EnterPlanModeTool — read-only mode for safe exploration before execution."""
    state = getattr(ctx, "state", None)
    if state is None:
        return "ERROR: state not available"
    state.update(plan_mode=True)
    engine = getattr(ctx, "engine", None)
    if engine is not None:
        try:
            engine.refresh_system_prompt()
        except Exception:
            pass
    return "OK: PLAN MODE entered. Mutating tools (write_file, edit_file, run_bash, etc.) will be denied. Use exit_plan_mode to resume."


def t_exit_plan_mode(args: dict, ctx: ToolContext) -> str:
    state = getattr(ctx, "state", None)
    if state is None:
        return "ERROR: state not available"
    state.update(plan_mode=False)
    engine = getattr(ctx, "engine", None)
    if engine is not None:
        try:
            engine.refresh_system_prompt()
        except Exception:
            pass
    return "OK: plan mode OFF. Mutating tools allowed again."


def t_todo_write(args: dict, ctx: ToolContext) -> str:
    """TodoWriteTool — write or update the per-session todo list."""
    state = getattr(ctx, "state", None)
    if state is None:
        return "ERROR: state not available"
    items = args.get("items")
    if not isinstance(items, list):
        return "ERROR: items must be a list of {text, status?}"
    todos: list[dict] = []
    for it in items:
        if isinstance(it, str):
            todos.append({"text": it, "status": "pending"})
        elif isinstance(it, dict) and "text" in it:
            todos.append({"text": it["text"], "status": it.get("status", "pending")})
    state.update(todos=todos)
    out = "\n".join(f"  [{t['status'][0]}] {t['text']}" for t in todos)
    return f"OK: {len(todos)} todo(s):\n{out}"


# -------------------------------------------------------------- system

def t_config_get(args: dict, ctx: ToolContext) -> str:
    """ConfigTool (get) — read a value from the persistent config."""
    engine = getattr(ctx, "engine", None)
    if engine is None or engine.config is None:
        return "ERROR: config not available"
    from .config import get_value
    key = args["key"]
    v = get_value(engine.config, key, None)
    if v is None:
        return f"(unset: {key})"
    return f"{key} = {v}"


def t_config_set(args: dict, ctx: ToolContext) -> str:
    engine = getattr(ctx, "engine", None)
    if engine is None or engine.config is None:
        return "ERROR: config not available"
    from .config import set_value, save
    key = args["key"]
    val = args["value"]
    if not ctx.confirm("config set", f"{key} = {val}"):
        return "ERROR: user denied"
    set_value(engine.config, key, val)
    save(engine.config)
    return f"OK: {key} = {val} (saved)"


def t_sleep(args: dict, ctx: ToolContext) -> str:
    """SleepTool — pause for N seconds. Capped at 60s."""
    import time as _time
    secs = float(args.get("seconds", 1))
    secs = max(0.0, min(60.0, secs))
    _time.sleep(secs)
    return f"OK: slept {secs}s"


def t_schedule_cron(args: dict, ctx: ToolContext) -> str:
    """ScheduleCronTool — register a recurring prompt (in-memory only this session)."""
    state = getattr(ctx, "state", None)
    if state is None:
        return "ERROR: state not available"
    import time as _time
    sched = list(state.schedules or [])
    sid = f"s{len(sched):04d}"
    entry = {
        "id": sid,
        "every_seconds": int(args.get("every_seconds", 0)),
        "expr": args.get("expr", ""),
        "prompt": args["prompt"],
        "last_run": 0.0,
        "registered_at": _time.time(),
    }
    sched.append(entry)
    state.update(schedules=sched)
    return f"OK: scheduled {sid} (run every {entry['every_seconds']}s, when due will inject the prompt on the next turn)"


# -------------------------------------------------------------- task control

def t_task_stop(args: dict, ctx: ToolContext) -> str:
    """TaskStopTool — mark a task cancelled. (Background threads can't be killed
    cleanly in pure-Python; we mark cancelled and ignore the result.)"""
    bg = getattr(ctx, "background", None)
    tid = args["id"]
    if bg is not None:
        job = bg.status(tid)
        if job and job.status == "running":
            job.status = "cancelled"
            return f"OK: marked background job {tid} cancelled (the thread may still finish)"
    tasks = getattr(ctx, "tasks", None)
    if tasks is not None:
        if tasks.update(tid, status="cancelled"):
            return f"OK: task {tid} marked cancelled"
    return f"ERROR: no task/job with id {tid}"


def t_task_output(args: dict, ctx: ToolContext) -> str:
    """TaskOutputTool — read the result/output of a task or background job."""
    bg = getattr(ctx, "background", None)
    tasks = getattr(ctx, "tasks", None)
    tid = args["id"]
    if bg is not None:
        job = bg.status(tid)
        if job:
            return _truncate(f"[bg {job.id} status={job.status}]\n{job.result}")
    if tasks is not None:
        t = tasks.get(tid)
        if t:
            return _truncate(f"[task {t.id} status={t.status}]\n{t.result}")
    return f"ERROR: no task/job with id {tid}"


# -------------------------------------------------------------- skills

def t_skill(args: dict, ctx: ToolContext) -> str:
    """SkillTool — append a named skill's instructions onto the engine for the
    rest of the session. Skills live at ~/.config/collama/skills/<name>.md
    (or .txt). The contents become a system-prompt addendum.
    """
    from .config import config_dir
    op = args.get("op", "use")
    name = args.get("name", "")
    skills_dir = config_dir() / "skills"
    skills_dir.mkdir(parents=True, exist_ok=True)

    if op == "list":
        files = sorted(skills_dir.glob("*"))
        if not files:
            return "(no skills installed; drop *.md files in " + str(skills_dir) + ")"
        return "\n".join(f"- {f.stem}  ({f.stat().st_size} bytes)" for f in files)

    if not name:
        return "ERROR: skill name required"
    candidate = None
    for ext in (".md", ".txt", ""):
        c = skills_dir / f"{name}{ext}"
        if c.exists():
            candidate = c
            break
    if op == "get":
        if not candidate:
            return f"ERROR: no skill '{name}'"
        return _truncate(candidate.read_text(errors="replace"))

    if op == "use":
        if not candidate:
            return f"ERROR: no skill '{name}'"
        engine = getattr(ctx, "engine", None)
        if engine is None:
            return "ERROR: engine not available"
        body = candidate.read_text(errors="replace")
        if engine.messages and engine.messages[0].get("role") == "system":
            engine.messages[0]["content"] += f"\n\n=== SKILL: {name} ===\n{body}"
        return f"OK: skill '{name}' attached to system prompt ({len(body)} chars)"

    return f"ERROR: unknown op '{op}'"


# -------------------------------------------------------------- mcp / lsp / tungsten (stubs)

def t_mcp(args: dict, ctx: ToolContext) -> str:
    return ("ERROR: MCP server not configured. To enable, install an MCP "
            "server, set its URL in ~/.config/collama/config.json under "
            "mcp.servers.<name>, and re-run.")


def t_mcp_list_resources(args: dict, ctx: ToolContext) -> str:
    return t_mcp(args, ctx)


def t_mcp_read_resource(args: dict, ctx: ToolContext) -> str:
    return t_mcp(args, ctx)


def t_lsp(args: dict, ctx: ToolContext) -> str:
    return ("ERROR: LSP not configured. Configure a language server (e.g. "
            "pyright, gopls) under lsp.servers in config.json. Stub.")


def t_tungsten(args: dict, ctx: ToolContext) -> str:
    return "ERROR: TungstenTool is a placeholder; not implemented in Collama."


def t_run_bash(args: dict, ctx: ToolContext) -> str:
    cmd = args["command"]
    timeout = int(args.get("timeout", 60))
    if not ctx.confirm("shell command", cmd):
        return "ERROR: user denied command"
    try:
        with ui.Spinner(f"running: {cmd[:40] + ('…' if len(cmd) > 40 else '')}"):
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
    "set_workspace": t_set_workspace,
    # s12 worktree
    "enter_worktree": t_enter_worktree,
    "exit_worktree": t_exit_worktree,
    # s07 tasks
    "task_create": t_task_create,
    "task_update": t_task_update,
    "task_get": t_task_get,
    "task_list": t_task_list,
    "task_delete": t_task_delete,
    # s08 background
    "bash_async": t_bash_async,
    "task_status": t_task_status,
    "task_wait": t_task_wait,
    # s04 sub-agent
    "agent_call": t_agent_call,
    "agent_call_async": t_agent_call_async,
    # s09 teams
    "team_create": t_team_create,
    "team_delete": t_team_delete,
    "team_list": t_team_list,
    "teammate_create": t_teammate_create,
    "teammate_delete": t_teammate_delete,
    "teammate_list": t_teammate_list,
    # s10 protocols
    "send_message": t_send_message,
    "inbox": t_inbox,
    # s11 coordinator
    "coordinator_tick": t_coordinator_tick,
    "coordinator_run": t_coordinator_run,
    # extended tools
    "glob": t_glob,
    "tool_search": t_tool_search,
    "powershell": t_powershell,
    "web_fetch": t_web_fetch,
    "web_search": t_web_search,
    "notebook_edit": t_notebook_edit,
    "ask_user_question": t_ask_user_question,
    "brief": t_brief,
    "enter_plan_mode": t_enter_plan_mode,
    "exit_plan_mode": t_exit_plan_mode,
    "todo_write": t_todo_write,
    "config_get": t_config_get,
    "config_set": t_config_set,
    "sleep": t_sleep,
    "schedule_cron": t_schedule_cron,
    "task_stop": t_task_stop,
    "task_output": t_task_output,
    "skill": t_skill,
    "mcp": t_mcp,
    "mcp_list_resources": t_mcp_list_resources,
    "mcp_read_resource": t_mcp_read_resource,
    "lsp": t_lsp,
    "tungsten": t_tungsten,
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
    # ---------- s12: worktrees ----------
    {"type": "function", "function": {
        "name": "enter_worktree",
        "description": "Push the current workspace onto a stack and switch to `path`. Use when working on a sub-task in its own directory. Pair with exit_worktree.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"},
            "create": {"type": "boolean"},
        }, "required": ["path"]},
    }},
    {"type": "function", "function": {
        "name": "exit_worktree",
        "description": "Pop the worktree stack and restore the previous workspace.",
        "parameters": {"type": "object", "properties": {}},
    }},

    # ---------- s07: persistent task graph ----------
    {"type": "function", "function": {
        "name": "task_create",
        "description": "Create a persistent task with status tracking and optional dependencies. Returns the new task id (e.g. t9f3e2c1).",
        "parameters": {"type": "object", "properties": {
            "title": {"type": "string"},
            "description": {"type": "string"},
            "deps": {"type": "array", "items": {"type": "string"}},
            "parent_id": {"type": "string"},
            "worktree": {"type": "string", "description": "Optional worktree directory bound to this task."},
        }, "required": ["title"]},
    }},
    {"type": "function", "function": {
        "name": "task_update",
        "description": "Update a task. Common: status (pending|active|done|blocked|failed|cancelled) and result.",
        "parameters": {"type": "object", "properties": {
            "id": {"type": "string"},
            "status": {"type": "string"},
            "title": {"type": "string"},
            "description": {"type": "string"},
            "result": {"type": "string"},
            "deps": {"type": "array", "items": {"type": "string"}},
            "worktree": {"type": "string"},
        }, "required": ["id"]},
    }},
    {"type": "function", "function": {
        "name": "task_get",
        "description": "Get one task by id (prefix match accepted).",
        "parameters": {"type": "object", "properties": {"id": {"type": "string"}}, "required": ["id"]},
    }},
    {"type": "function", "function": {
        "name": "task_list",
        "description": "List tasks, optionally filtered by status and/or parent_id.",
        "parameters": {"type": "object", "properties": {
            "status": {"type": "string"},
            "parent_id": {"type": "string"},
        }},
    }},
    {"type": "function", "function": {
        "name": "task_delete",
        "description": "Delete a task by id.",
        "parameters": {"type": "object", "properties": {"id": {"type": "string"}}, "required": ["id"]},
    }},

    # ---------- s08: background ----------
    {"type": "function", "function": {
        "name": "bash_async",
        "description": "Run a shell command IN THE BACKGROUND. Returns a job id immediately; the result will be auto-injected on completion. Use task_status / task_wait to check sooner.",
        "parameters": {"type": "object", "properties": {
            "command": {"type": "string"},
            "timeout": {"type": "integer"},
        }, "required": ["command"]},
    }},
    {"type": "function", "function": {
        "name": "task_status",
        "description": "Check the status of a background job (bash_async / agent_call_async).",
        "parameters": {"type": "object", "properties": {"task_id": {"type": "string"}}, "required": ["task_id"]},
    }},
    {"type": "function", "function": {
        "name": "task_wait",
        "description": "Block until a background job finishes or `timeout` seconds elapse.",
        "parameters": {"type": "object", "properties": {
            "task_id": {"type": "string"},
            "timeout": {"type": "number"},
        }, "required": ["task_id"]},
    }},

    # ---------- s04: sub-agents ----------
    {"type": "function", "function": {
        "name": "agent_call",
        "description": "Fork a sub-agent on a FRESH conversation to handle a focused subtask (e.g. 'find every file that imports requests and summarize'). Returns the sub-agent's final answer. Inherits workspace, github_token, etc., but its messages are isolated so the main context stays clean.",
        "parameters": {"type": "object", "properties": {
            "prompt": {"type": "string"},
            "model": {"type": "string", "description": "Override model for this sub-agent. Default: same as parent."},
            "title": {"type": "string"},
        }, "required": ["prompt"]},
    }},
    {"type": "function", "function": {
        "name": "agent_call_async",
        "description": "Like agent_call but runs in the background; result is auto-injected on completion. Useful for long research while you keep working.",
        "parameters": {"type": "object", "properties": {
            "prompt": {"type": "string"},
            "model": {"type": "string"},
        }, "required": ["prompt"]},
    }},

    # ---------- s09 teams ----------
    {"type": "function", "function": {
        "name": "team_create",
        "description": "Create a persistent team (a directory of long-lived teammate personas).",
        "parameters": {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]},
    }},
    {"type": "function", "function": {
        "name": "team_delete",
        "description": "Delete a team and all its teammates. Requires user approval.",
        "parameters": {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]},
    }},
    {"type": "function", "function": {
        "name": "team_list",
        "description": "List all teams and their teammates.",
        "parameters": {"type": "object", "properties": {}},
    }},
    {"type": "function", "function": {
        "name": "teammate_create",
        "description": "Add a teammate to a team. `role` is appended to the system prompt for that teammate; `skills` are tags used by the coordinator's auto-claim.",
        "parameters": {"type": "object", "properties": {
            "team": {"type": "string"},
            "name": {"type": "string"},
            "role": {"type": "string"},
            "skills": {"type": "array", "items": {"type": "string"}},
        }, "required": ["team", "name"]},
    }},
    {"type": "function", "function": {
        "name": "teammate_delete",
        "description": "Remove a teammate from a team.",
        "parameters": {"type": "object", "properties": {
            "team": {"type": "string"}, "name": {"type": "string"},
        }, "required": ["team", "name"]},
    }},
    {"type": "function", "function": {
        "name": "teammate_list",
        "description": "List teammates, optionally filtered to one team.",
        "parameters": {"type": "object", "properties": {"team": {"type": "string"}}},
    }},

    # ---------- s10 protocols ----------
    {"type": "function", "function": {
        "name": "send_message",
        "description": "Send a message to a teammate's inbox (request-response protocol). Recipient processes mail next coordinator_tick / coordinator_run.",
        "parameters": {"type": "object", "properties": {
            "team": {"type": "string"},
            "to": {"type": "string", "description": "Recipient teammate name or id."},
            "content": {"type": "string"},
            "from": {"type": "string", "description": "Sender label. Default 'lead'."},
            "kind": {"type": "string", "description": "msg | task | question | reply"},
        }, "required": ["team", "to", "content"]},
    }},
    {"type": "function", "function": {
        "name": "inbox",
        "description": "Read a teammate's pending mailbox.",
        "parameters": {"type": "object", "properties": {
            "team": {"type": "string"}, "name": {"type": "string"},
        }, "required": ["team", "name"]},
    }},

    # ---------- s11 coordinator ----------
    {"type": "function", "function": {
        "name": "coordinator_tick",
        "description": "One coordinator tick: process every teammate's mailbox by spawning a sub-agent. With auto_claim=true, idle teammates also pick up matching pending tasks from the task graph.",
        "parameters": {"type": "object", "properties": {
            "team": {"type": "string", "description": "Limit to one team. Default: all teams."},
            "auto_claim": {"type": "boolean"},
            "max_per_teammate": {"type": "integer"},
        }},
    }},
    {"type": "function", "function": {
        "name": "coordinator_run",
        "description": "Tick the coordinator repeatedly until no teammate has work (or max_rounds is reached).",
        "parameters": {"type": "object", "properties": {
            "team": {"type": "string"},
            "max_rounds": {"type": "integer"},
            "auto_claim": {"type": "boolean"},
        }},
    }},

    # ---------- extended file / search ----------
    {"type": "function", "function": {
        "name": "glob",
        "description": "File pattern matching. Supports ** for recursive globs (e.g. **/*.py, src/*.ts).",
        "parameters": {"type": "object", "properties": {
            "pattern": {"type": "string"}, "path": {"type": "string"},
        }, "required": ["pattern"]},
    }},
    {"type": "function", "function": {
        "name": "tool_search",
        "description": "Search the registered tools by keyword in their name or description.",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
    }},

    # ---------- shell variants ----------
    {"type": "function", "function": {
        "name": "powershell",
        "description": "Run a command via pwsh / powershell.exe. Asks for approval.",
        "parameters": {"type": "object", "properties": {
            "command": {"type": "string"}, "timeout": {"type": "integer"},
        }, "required": ["command"]},
    }},

    # ---------- web ----------
    {"type": "function", "function": {
        "name": "web_fetch",
        "description": "Fetch a URL (HTTP/HTTPS) and return the body text. Caps at ~200 KB.",
        "parameters": {"type": "object", "properties": {
            "url": {"type": "string"},
            "timeout": {"type": "integer"},
            "max_bytes": {"type": "integer"},
        }, "required": ["url"]},
    }},
    {"type": "function", "function": {
        "name": "web_search",
        "description": "Search the web (DuckDuckGo HTML frontend). Returns title + URL pairs.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string"}, "limit": {"type": "integer"},
        }, "required": ["query"]},
    }},

    # ---------- notebook ----------
    {"type": "function", "function": {
        "name": "notebook_edit",
        "description": "Get/insert/replace/delete a cell in a Jupyter .ipynb file.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"},
            "op": {"type": "string", "enum": ["get", "insert", "replace", "delete"]},
            "cell_index": {"type": "integer"},
            "source": {"type": "string"},
            "cell_type": {"type": "string", "enum": ["code", "markdown", "raw"]},
        }, "required": ["path"]},
    }},

    # ---------- interaction ----------
    {"type": "function", "function": {
        "name": "ask_user_question",
        "description": "Pause and ask the user a question. Optionally offer multiple-choice options.",
        "parameters": {"type": "object", "properties": {
            "question": {"type": "string"},
            "options": {"type": "array", "items": {"type": "string"}},
        }, "required": ["question"]},
    }},
    {"type": "function", "function": {
        "name": "brief",
        "description": "Store / retrieve / list a named markdown brief in this session.",
        "parameters": {"type": "object", "properties": {
            "op": {"type": "string", "enum": ["set", "get", "list", "delete"]},
            "name": {"type": "string"},
            "content": {"type": "string"},
        }, "required": ["name"]},
    }},

    # ---------- planning / workflow ----------
    {"type": "function", "function": {
        "name": "enter_plan_mode",
        "description": "Enter PLAN MODE: read-only, no mutating tools. Use to safely explore before committing to changes.",
        "parameters": {"type": "object", "properties": {}},
    }},
    {"type": "function", "function": {
        "name": "exit_plan_mode",
        "description": "Leave plan mode; mutating tools allowed again.",
        "parameters": {"type": "object", "properties": {}},
    }},
    {"type": "function", "function": {
        "name": "todo_write",
        "description": "Replace this session's todo list. Items are strings or {text, status}.",
        "parameters": {"type": "object", "properties": {
            "items": {"type": "array", "items": {}},
        }, "required": ["items"]},
    }},

    # ---------- system ----------
    {"type": "function", "function": {
        "name": "config_get",
        "description": "Read a value from the persistent config (dotted key, e.g. 'github.token').",
        "parameters": {"type": "object", "properties": {"key": {"type": "string"}}, "required": ["key"]},
    }},
    {"type": "function", "function": {
        "name": "config_set",
        "description": "Set a config value. Asks for approval.",
        "parameters": {"type": "object", "properties": {
            "key": {"type": "string"}, "value": {},
        }, "required": ["key", "value"]},
    }},
    {"type": "function", "function": {
        "name": "sleep",
        "description": "Pause execution for `seconds` (capped at 60).",
        "parameters": {"type": "object", "properties": {"seconds": {"type": "number"}}},
    }},
    {"type": "function", "function": {
        "name": "schedule_cron",
        "description": "Register a recurring prompt; on each turn the engine checks if any schedule is due and re-runs the prompt. (In-memory for now.)",
        "parameters": {"type": "object", "properties": {
            "prompt": {"type": "string"},
            "every_seconds": {"type": "integer"},
            "expr": {"type": "string"},
        }, "required": ["prompt"]},
    }},

    # ---------- task control ----------
    {"type": "function", "function": {
        "name": "task_stop",
        "description": "Mark a task or background job cancelled.",
        "parameters": {"type": "object", "properties": {"id": {"type": "string"}}, "required": ["id"]},
    }},
    {"type": "function", "function": {
        "name": "task_output",
        "description": "Read the result/output of a task or background job by id.",
        "parameters": {"type": "object", "properties": {"id": {"type": "string"}}, "required": ["id"]},
    }},

    # ---------- skills ----------
    {"type": "function", "function": {
        "name": "skill",
        "description": "Manage and apply skills (markdown bundles in ~/.config/collama/skills/). op: list | get | use.",
        "parameters": {"type": "object", "properties": {
            "op": {"type": "string", "enum": ["list", "get", "use"]},
            "name": {"type": "string"},
        }},
    }},

    # ---------- MCP / LSP / tungsten (stubs) ----------
    {"type": "function", "function": {
        "name": "mcp",
        "description": "Generic MCP tool call. Stubbed — returns ERROR until an MCP server is configured.",
        "parameters": {"type": "object", "properties": {
            "server": {"type": "string"}, "tool": {"type": "string"}, "arguments": {"type": "object"},
        }},
    }},
    {"type": "function", "function": {
        "name": "mcp_list_resources",
        "description": "List resources exposed by an MCP server. Stub.",
        "parameters": {"type": "object", "properties": {"server": {"type": "string"}}},
    }},
    {"type": "function", "function": {
        "name": "mcp_read_resource",
        "description": "Read an MCP resource by URI. Stub.",
        "parameters": {"type": "object", "properties": {
            "server": {"type": "string"}, "uri": {"type": "string"},
        }},
    }},
    {"type": "function", "function": {
        "name": "lsp",
        "description": "Call into a language server (definitions / references / hover). Stub.",
        "parameters": {"type": "object", "properties": {
            "method": {"type": "string"}, "params": {"type": "object"},
        }},
    }},
    {"type": "function", "function": {
        "name": "tungsten",
        "description": "Reserved placeholder — not implemented.",
        "parameters": {"type": "object", "properties": {}},
    }},
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
