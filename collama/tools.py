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
