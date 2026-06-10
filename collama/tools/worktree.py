"""enter_worktree / exit_worktree — push/pop the workspace stack."""
from __future__ import annotations

from pathlib import Path

from .base import ToolContext, _resolve


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


TOOLS = {
    "enter_worktree": t_enter_worktree,
    "exit_worktree":  t_exit_worktree,
}


TOOL_SCHEMAS = [
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
]
