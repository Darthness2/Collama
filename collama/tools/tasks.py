"""Persistent task graph + todo/brief + task_stop/task_output.

The underlying registry is `collama.tasks.TaskGraph` (a different module —
this one is just the model-facing wrappers grouped by tool category).
"""
from __future__ import annotations

from .base import ToolContext, _truncate


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


TOOLS = {
    "task_create": t_task_create,
    "task_update": t_task_update,
    "task_get":    t_task_get,
    "task_list":   t_task_list,
    "task_delete": t_task_delete,
    "todo_write":  t_todo_write,
    "brief":       t_brief,
    "task_stop":   t_task_stop,
    "task_output": t_task_output,
}


TOOL_SCHEMAS = [
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
    {"type": "function", "function": {
        "name": "todo_write",
        "description": "Replace this session's todo list. Items are strings or {text, status}.",
        "parameters": {"type": "object", "properties": {
            "items": {"type": "array", "items": {}},
        }, "required": ["items"]},
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
]
