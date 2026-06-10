"""Aggregates every per-group submodule into the single set of registries
the engine uses: TOOLS (dispatch table), TOOL_SCHEMAS (sent to the model),
TOOL_GROUPS (group → tool names) and DEFAULT_GROUPS.

Also home to dispatch(), the alias map, and the schema compactor.
"""
from __future__ import annotations

from typing import Any, Callable

from .base import ToolContext
from . import (
    background,
    core,
    interaction,
    mcp as mcp_tools,
    notebook,
    planning,
    search,
    subagent,
    syntax,
    system,
    tasks,
    teams,
    worktree,
)


ToolFn = Callable[[dict, ToolContext], str]


# Master tool dispatch table. Built by merging every submodule's local TOOLS.
TOOLS: dict[str, ToolFn] = {
    **core.TOOLS,
    **syntax.TOOLS,
    **worktree.TOOLS,
    **tasks.TOOLS,
    **background.TOOLS,
    **subagent.TOOLS,
    **teams.TOOLS,
    **search.TOOLS,
    **notebook.TOOLS,
    **interaction.TOOLS,
    **planning.TOOLS,
    **system.TOOLS,
    **mcp_tools.TOOLS,
}


# Master schema list. Order matters for tool_search output, so this mirrors
# the original ordering: core → worktree → tasks/background/subagent/teams →
# extended search/web/notebook/interaction/planning/system/skills/mcp.
TOOL_SCHEMAS: list[dict[str, Any]] = (
    core.TOOL_SCHEMAS
    + worktree.TOOL_SCHEMAS
    + tasks.TOOL_SCHEMAS[:5]     # task_create..task_delete
    + background.TOOL_SCHEMAS
    + subagent.TOOL_SCHEMAS
    + teams.TOOL_SCHEMAS
    + search.TOOL_SCHEMAS[:2]    # glob, tool_search
    + system.TOOL_SCHEMAS[-1:]   # powershell
    + search.TOOL_SCHEMAS[2:]    # web_fetch, web_search
    + notebook.TOOL_SCHEMAS
    + interaction.TOOL_SCHEMAS
    + tasks.TOOL_SCHEMAS[6:7]    # brief
    + planning.TOOL_SCHEMAS
    + tasks.TOOL_SCHEMAS[5:6]    # todo_write
    + system.TOOL_SCHEMAS[:3]    # config_get, config_set, sleep
    + tasks.TOOL_SCHEMAS[7:]     # task_stop, task_output
    + system.TOOL_SCHEMAS[3:4]   # skill
    + mcp_tools.TOOL_SCHEMAS
    + syntax.TOOL_SCHEMAS
)


def _all_tools() -> dict[str, ToolFn]:
    from ..github import GITHUB_TOOLS
    return {**TOOLS, **GITHUB_TOOLS}


# Tool groups — sending all ~50 schemas every request is a heavy prompt-eval
# cost on a local model. Groups let the heavy/rarely-used ones be opt-in.
TOOL_GROUPS: dict[str, list[str]] = {
    "core": [
        "read_file", "write_file", "edit_file", "multi_edit", "replace_lines",
        "list_dir", "grep", "run_bash", "set_workspace", "check_syntax",
    ],
    "search": ["glob", "tool_search", "web_fetch", "web_search"],
    "tasks": [
        "task_create", "task_update", "task_get", "task_list", "task_delete",
        "todo_write", "brief",
    ],
    "background": ["bash_async", "task_status", "task_wait"],
    "planning": ["enter_plan_mode", "exit_plan_mode"],
    "notebook": ["notebook_edit"],
    "worktree": ["enter_worktree", "exit_worktree"],
    "interaction": ["ask_user_question"],
    "system": ["config_get", "config_set", "sleep", "skill", "powershell"],
    # heavy / specialized — OFF by default
    "subagent": ["agent_call", "agent_call_async"],
    "github": [
        "gh_whoami", "gh_list_repos", "gh_get_repo", "gh_get_file",
        "gh_list_issues", "gh_create_issue", "gh_list_pulls", "gh_get_pull",
        "gh_search_code", "github_api",
    ],
    "teams": [
        "team_create", "team_delete", "team_list", "teammate_create",
        "teammate_delete", "teammate_list", "send_message", "inbox",
        "coordinator_tick", "coordinator_run",
    ],
    # MCP: the two control tools live here; auto-discovered third-party
    # tools (mcp__<server>__<tool>) are injected at all_tool_schemas() time.
    "mcp": ["mcp_servers", "mcp_restart"],
}

DEFAULT_GROUPS: set[str] = {
    "core", "search", "tasks", "background", "planning",
    "notebook", "worktree", "interaction", "system",
}
# off by default: subagent, github, teams, mcp


def _compact_schema(schema: dict) -> dict:
    """Return a copy of a tool schema with descriptions trimmed to one line.

    Most tool/param descriptions are paragraphs explaining edge cases — the
    model only needs the first line to know what the tool does and how to
    call it. Trimming saves ~40% of the per-request tool prompt overhead,
    which is one of the biggest wins on slow local hardware.
    """
    fn = schema.get("function", {}) or {}
    desc = (fn.get("description") or "").strip()
    short_desc = desc.splitlines()[0].strip() if desc else ""
    if len(short_desc) > 140:
        short_desc = short_desc[:137] + "…"
    new_fn = {**fn, "description": short_desc}
    params = new_fn.get("parameters") or {}
    if isinstance(params, dict) and isinstance(params.get("properties"), dict):
        # Drop parameter `description` fields entirely. Param names like
        # 'path', 'pattern', 'command' are self-explanatory; the model
        # knows what to put there. Saves the bulk of the prompt tokens.
        new_props = {}
        for pname, pspec in params["properties"].items():
            if isinstance(pspec, dict):
                new_props[pname] = {k: v for k, v in pspec.items() if k != "description"}
            else:
                new_props[pname] = pspec
        new_fn["parameters"] = {**params, "properties": new_props}
    return {**schema, "function": new_fn}


def all_tool_schemas(
    enabled_groups: set[str] | None = None,
    compact: bool = True,
) -> list[dict]:
    """Return tool schemas for the enabled groups (DEFAULT_GROUPS if None).

    When the ``mcp`` group is enabled the configured MCP servers spawn (if
    they haven't already) and their auto-discovered tools join the schema
    list — namespaced as ``mcp__<server>__<tool>``.

    ``compact=True`` (default) trims every description to its first line —
    saves substantial prompt-eval cost on local models. Pass False to keep
    the verbose docstrings if a model is struggling without them.
    """
    from ..github import GITHUB_TOOL_SCHEMAS
    groups = DEFAULT_GROUPS if enabled_groups is None else set(enabled_groups)
    allowed = {n for g in groups for n in TOOL_GROUPS.get(g, ())}
    schemas = TOOL_SCHEMAS + GITHUB_TOOL_SCHEMAS
    filtered = [s for s in schemas if s.get("function", {}).get("name") in allowed]
    if "mcp" in groups:
        # Lazy: this spawns any configured MCP servers and pulls their
        # tools/list. ensure_started() is a no-op once everything is ready.
        from .. import mcp as _mcp_mod
        reg = _mcp_mod.registry()
        reg.ensure_started()
        filtered.extend(reg.all_tool_schemas())
    return [_compact_schema(s) for s in filtered] if compact else filtered


# Common shortened names that small models reach for. Keep this map tight —
# only well-known synonyms; ambiguous abbreviations should NOT be aliased.
TOOL_ALIASES: dict[str, str] = {
    "read":         "read_file",
    "open":         "read_file",
    "view":         "read_file",
    "cat":          "read_file",
    "write":        "write_file",
    "create":       "write_file",
    "edit":         "edit_file",
    "patch":        "edit_file",
    "replace":      "edit_file",
    "ls":           "list_dir",
    "list":         "list_dir",
    "dir":          "list_dir",
    "search":       "grep",
    "find":         "glob",
    "bash":         "run_bash",
    "shell":        "run_bash",
    "exec":         "run_bash",
    "run":          "run_bash",
    "ps":           "run_bash",
    "cd":           "set_workspace",
    "fetch":        "web_fetch",
    "curl":         "web_fetch",
    "wget":         "web_fetch",
    "search_web":   "web_search",
    "todo":         "todo_write",
    "todos":        "todo_write",
}


def dispatch(name: str, args: dict, ctx: ToolContext) -> str:
    # MCP-namespaced tools route to the MCP registry instead of TOOLS.
    if name.startswith("mcp__"):
        try:
            from .. import mcp as _mcp_mod
            return _mcp_mod.registry().dispatch(name, args)
        except Exception as e:
            return f"ERROR: {type(e).__name__}: {e}"

    all_tools = _all_tools()
    fn = all_tools.get(name)
    if fn is None:
        # Try a direct alias mapping first — silent fix for common shortened
        # names like 'read', 'cat', 'ls'. Logs a faint note in the result so
        # the model learns the canonical name.
        canonical = TOOL_ALIASES.get(name.lower())
        if canonical and canonical in all_tools:
            result = all_tools[canonical](args, ctx)
            if isinstance(result, str) and not result.startswith("ERROR"):
                result = f"[note: '{name}' is an alias for '{canonical}' — use '{canonical}' next time]\n{result}"
            return result
        # No alias — suggest the closest matching name(s) so the model can
        # self-correct on the next call instead of looping on the same typo.
        # Pool includes alias keys so 'reed' can land on 'read'/'read_file'.
        import difflib
        pool = list(all_tools.keys()) + list(TOOL_ALIASES.keys())
        suggestions = difflib.get_close_matches(name.lower(), pool, n=3, cutoff=0.4)
        # Resolve any alias matches to their canonical names + dedupe.
        canonical_suggestions: list[str] = []
        for s in suggestions:
            c = TOOL_ALIASES.get(s, s)
            if c in all_tools and c not in canonical_suggestions:
                canonical_suggestions.append(c)
        hint = f"  Did you mean: {', '.join(canonical_suggestions)}?" if canonical_suggestions else ""
        return f"ERROR: unknown tool '{name}'.{hint}  Use /tools to see the full list."
    try:
        return fn(args, ctx)
    except KeyError as e:
        return f"ERROR: missing argument {e}"
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}"
