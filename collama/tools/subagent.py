"""agent_call / agent_call_async — fork a sub-agent on a fresh conversation."""
from __future__ import annotations

from .base import ToolContext, _truncate


def t_agent_call(args: dict, ctx: ToolContext) -> str:
    """Fork a sub-agent on a fresh messages[]; return its final answer."""
    engine = getattr(ctx, "engine", None)
    if engine is None:
        return "ERROR: engine not available for sub-agent"
    from ..subagent import fork_subagent
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
    from ..subagent import fork_subagent
    prompt = args["prompt"]
    model = args.get("model")

    def _run(p):
        return fork_subagent(engine, p, model=model)

    job_id = bg.submit_dream(prompt, _run)
    return f"OK: dream {job_id} dispatched (will surface on completion)"


TOOLS = {
    "agent_call":       t_agent_call,
    "agent_call_async": t_agent_call_async,
}


TOOL_SCHEMAS = [
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
