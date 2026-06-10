"""ask_user_question — pause execution and prompt the human."""
from __future__ import annotations

from .base import ToolContext


def t_ask_user_question(args: dict, ctx: ToolContext) -> str:
    """AskUserQuestionTool — pause and ask the user."""
    question = args.get("question") or args.get("prompt") or args.get("q")
    if not question:
        return "ERROR: missing argument 'question'"
    options = args.get("options") or []
    if ctx.yolo:
        return "ERROR: cannot ask user in yolo mode"
    from .. import ui
    # CRITICAL: stop any active spinner before reading input — otherwise the
    # spinner thread keeps redrawing the line and clobbers what the user is
    # typing. Same reason terminal_resolver does this for approval prompts.
    ui.stop_all_spinners()
    import sys as _sys
    if _sys.stdout.isatty():
        _sys.stdout.write("\033[?25h")  # show cursor
        _sys.stdout.flush()
    print()
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


TOOLS = {"ask_user_question": t_ask_user_question}


TOOL_SCHEMAS = [
    {"type": "function", "function": {
        "name": "ask_user_question",
        "description": "Pause and ask the user a question. Optionally offer multiple-choice options.",
        "parameters": {"type": "object", "properties": {
            "question": {"type": "string"},
            "options": {"type": "array", "items": {"type": "string"}},
        }, "required": ["question"]},
    }},
]
