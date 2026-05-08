"""canUseTool(): formal permission gate.

Decision sources, in order:
    1. Per-tool cache in AppState.permissions ('always'/'never').
    2. Read-only tools — always allowed without prompting.
    3. yolo mode — always allowed.
    4. Resolver callback — interactive y/n/a prompt for the REPL,
       or auto-decide for the SDK / headless caller.
"""
from __future__ import annotations

from typing import Callable

from .state import AppState


# Tools that don't mutate anything we care about — always allowed.
READ_ONLY: set[str] = {
    "read_file", "list_dir", "grep", "glob", "tool_search",
    "web_fetch", "web_search",
    "task_get", "task_list", "task_output", "task_status",
    "inbox", "team_list", "teammate_list",
    "config_get", "sleep",
    "gh_whoami", "gh_list_repos", "gh_get_repo", "gh_get_file",
    "gh_list_issues", "gh_list_pulls", "gh_get_pull", "gh_search_code",
}


# Tools that are safe to run concurrently (no shared mutable state).
CONCURRENT_SAFE: set[str] = READ_ONLY | {"set_workspace"}


# Resolver: (tool_name, args, state) -> 'yes' | 'always' | 'no' | 'never'
Resolver = Callable[[str, dict, AppState], str]


def auto_deny_resolver(name: str, args: dict, state: AppState) -> str:
    """Default for headless / SDK use: never prompt, deny mutating ops."""
    return "no"


def can_use_tool(
    name: str,
    args: dict,
    state: AppState,
    resolver: Resolver = auto_deny_resolver,
) -> tuple[bool, str]:
    """Returns (allowed, reason)."""
    cached = state.permissions.get(name)
    if cached == "always":
        return True, "permission cache: always"
    if cached == "never":
        return False, "permission cache: never"

    # Plan-mode gate: only read-only tools are allowed.
    if getattr(state, "plan_mode", False) and name not in READ_ONLY and name not in {
        "enter_plan_mode", "exit_plan_mode", "ask_user_question", "brief",
        "todo_write", "tool_search", "task_get", "task_list", "task_output",
        "team_list", "teammate_list", "inbox", "config_get",
    }:
        return False, "plan mode active — mutating tools blocked"

    if name in READ_ONLY:
        return True, "read-only tool"

    if state.yolo:
        return True, "yolo mode"

    answer = resolver(name, args, state) or "no"
    if answer == "always":
        state.permissions[name] = "always"
        return True, "user approved (always)"
    if answer == "never":
        state.permissions[name] = "never"
        return False, "user denied (never)"
    if answer == "yes":
        return True, "user approved"
    return False, "user denied"


def terminal_resolver(name: str, args: dict, state: AppState) -> str:
    """Interactive REPL resolver: ask the user [y]es / [n]o / [a]lways / never."""
    from . import ui
    detail = ""
    if name == "run_bash":
        detail = f": {args.get('command', '')}"
    elif name == "write_file":
        detail = f": write {args.get('path', '')}"
    elif name == "edit_file":
        detail = f": edit {args.get('path', '')}"
    elif name.startswith("gh_") or name == "github_api":
        detail = f": {name} {args}"
    ui.warn(f"\nApprove {name}{detail}?")
    try:
        ans = input("  [y]es / [n]o / [a]lways / [N]ever: ").strip().lower()
    except EOFError:
        return "no"
    if ans in ("a", "always"):
        return "always"
    if ans == "never":
        return "never"
    if ans in ("y", "yes"):
        return "yes"
    return "no"
