"""enter_plan_mode / exit_plan_mode — gate the agent into read-only exploration."""
from __future__ import annotations

from .base import ToolContext


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
    return ("OK: PLAN MODE entered (approve-then-execute). Mutating tools "
            "(write_file, edit_file, run_bash, etc.) are denied. Investigate "
            "read-only, then call exit_plan_mode with a `plan` for the user to "
            "approve — only after approval can you make changes.")


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


TOOLS = {
    "enter_plan_mode": t_enter_plan_mode,
    "exit_plan_mode":  t_exit_plan_mode,
}


TOOL_SCHEMAS = [
    {"type": "function", "function": {
        "name": "enter_plan_mode",
        "description": "Enter PLAN MODE: read-only, no mutating tools. Use to safely explore before committing to changes.",
        "parameters": {"type": "object", "properties": {}},
    }},
    {"type": "function", "function": {
        "name": "exit_plan_mode",
        "description": "Signal that your plan is ready for the user to approve. Pass the full plan in `plan`. The user is asked to approve it; only on approval does plan mode turn off and let you make the actual changes.",
        "parameters": {"type": "object", "properties": {
            "plan": {"type": "string", "description": "The numbered plan of changes to present to the user for approval."},
        }},
    }},
]
