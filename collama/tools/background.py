"""bash_async / task_status / task_wait — run shell jobs in the background."""
from __future__ import annotations

from .base import ToolContext, _truncate


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


TOOLS = {
    "bash_async":  t_bash_async,
    "task_status": t_task_status,
    "task_wait":   t_task_wait,
}


TOOL_SCHEMAS = [
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
]
