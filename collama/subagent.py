"""Sub-agents (forkSubagent).

Forks a child QueryEngine with a FRESH messages[] but inherited state
(workspace, github_token, yolo, …). The child runs an isolated turn —
its own plan, tools, loop — and returns a single string answer to the
parent. The parent's main conversation stays focused; the sub-agent
acts as a black-box research/specialist call.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from .state import AppState

if TYPE_CHECKING:  # pragma: no cover
    from .engine import QueryEngine

_log = logging.getLogger(__name__)


def fork_subagent(
    parent: "QueryEngine",
    prompt: str,
    *,
    model: str | None = None,
    title: str = "subagent",
    role: str | None = None,
) -> str:
    """Run a sub-agent and return its final answer.

    `role` (optional) is appended to the system prompt — used by the team
    coordinator to give each teammate a persistent persona without
    rewriting the engine's system-prompt builder.
    """
    from .engine import QueryEngine  # local import — avoids circular at module load
    from .permissions import auto_deny_resolver, terminal_resolver

    sub_state = AppState(
        workspace=parent.state.workspace,
        home=parent.state.home,
        github_token=parent.state.github_token,
        yolo=parent.state.yolo,
        insecure_ssl=parent.state.insecure_ssl,
        tools_enabled=parent.state.tools_enabled,
        permissions=dict(parent.state.permissions),
    )
    # A sub-agent runs as a black-box call with no terminal of its own. If it
    # inherited the parent's interactive resolver (terminal_resolver), a
    # mutating tool would block forever on input() from a thread/background
    # context. Use a non-interactive resolver instead: yolo still
    # auto-approves, the permissions cache still applies, and anything else is
    # denied rather than hanging. If the parent itself is non-interactive,
    # inherit its resolver.
    parent_resolver = getattr(parent, "permission_resolver", None)
    child_resolver = (
        auto_deny_resolver
        if parent_resolver is terminal_resolver or parent_resolver is None
        else parent_resolver
    )

    sub = QueryEngine(
        client=parent.client,
        state=sub_state,
        model=model or parent.model,
        temperature=parent.temperature,
        config=parent.config,
        permission_resolver=child_resolver,
        stream=False,
    )

    if role and sub.messages and sub.messages[0].get("role") == "system":
        sub.messages[0]["content"] = sub.messages[0]["content"] + "\n\n=== ROLE ===\n" + role

    final = ""
    try:
        for ev in sub.submit_message(prompt):
            if ev.kind in ("assistant", "done"):
                text = ev.data.get("text") or ""
                if text:
                    final = text
            elif ev.kind == "error":
                return f"sub-agent error: {ev.data.get('text', '')}"
    except Exception as e:
        # A raised provider/tool error inside the child must not crash the
        # parent turn (or the coordinator). Surface it as an error result.
        _log.warning("sub-agent '%s' raised: %s", title, e, exc_info=True)
        return f"ERROR: sub-agent '{title}' failed: {type(e).__name__}: {e}"
    return final or "(sub-agent returned no answer)"
