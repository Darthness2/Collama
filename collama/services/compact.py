"""Auto-compact: keep the message history from blowing past the model's context.

Strategy (simple, no second LLM call needed for a baseline):
- Always keep the system prompt and the last `keep_recent` messages verbatim.
- For everything in between, keep just a synthetic "[older context summary]"
  user message that lists the assistant's *final answers* and the tool calls
  that ran. That preserves the gist while shrinking length dramatically.

A more sophisticated version would call the model to summarize. We'll add
that as `summarize_with_model=True` later.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CompactReport:
    triggered: bool
    before: int
    after: int
    summary_added: bool


def _approx_tokens(messages: list[dict]) -> int:
    """Cheap token estimator: 1 token ≈ 4 chars of content."""
    total = 0
    for m in messages:
        total += len(str(m.get("content") or "")) // 4
    return total


def auto_compact(
    messages: list[dict],
    *,
    max_tokens: int = 12000,
    keep_recent: int = 12,
) -> CompactReport:
    """Compact `messages` in place if it crosses `max_tokens`.

    Returns a report so the caller can surface the event.
    """
    before = _approx_tokens(messages)
    if before <= max_tokens or len(messages) <= keep_recent + 2:
        return CompactReport(False, before, before, False)

    head: list[dict] = []
    if messages and messages[0].get("role") == "system":
        head.append(messages[0])

    tail = messages[-keep_recent:]
    middle = messages[len(head):-keep_recent]

    # Build a tiny summary from `middle`.
    bullets: list[str] = []
    for m in middle:
        role = m.get("role")
        content = (m.get("content") or "").strip()
        if role == "assistant":
            # last lines are usually the final answer / decision
            snippet = content.splitlines()[-1][:160] if content else ""
            if snippet:
                bullets.append(f"- assistant: {snippet}")
            for tc in m.get("tool_calls") or []:
                fn = tc.get("function", {}) or {}
                bullets.append(f"  - called {fn.get('name', '?')}")
        elif role == "tool":
            bullets.append(f"- tool {m.get('name', '?')}: {content.splitlines()[0][:160] if content else ''}")
        elif role == "user":
            first = content.splitlines()[0][:120]
            if first:
                bullets.append(f"- user: {first}")

    summary = "[older context, compacted]\n" + "\n".join(bullets[-80:])
    summary_msg = {"role": "user", "content": summary}

    new_messages = head + [summary_msg] + tail
    messages.clear()
    messages.extend(new_messages)
    after = _approx_tokens(messages)
    return CompactReport(True, before, after, True)
