"""Context management.

Layout of the conversation buffer:

    [system prompt]
    [compacted summary of older messages]   ← optional
    [<<COMPACT_BOUNDARY>>] (system marker)  ← inserted by auto_compact
    [recent messages — full fidelity]
    [current turn]

Three compression strategies:
    autoCompact      — LLM-summarize messages before the boundary
    snipCompact      — remove zombie messages (empty / duplicate)
    contextCollapse  — merge consecutive same-role messages

Helpers:
    find_compact_boundary(messages)   index of last boundary or -1
    messages_after_boundary(messages) slice after the boundary
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass

log = logging.getLogger(__name__)

BOUNDARY_MARKER = "<<COMPACT_BOUNDARY>>"


@dataclass
class CompactReport:
    triggered: bool
    before: int
    after: int
    strategy: str
    summary_added: bool = False


def approx_message_tokens(m: dict) -> int:
    """Rough chars/4 token estimate for a single message.

    Counts not just `content` but also the assistant `tool_calls` payload
    (function name + serialized arguments) and any attached `images` — both
    occupy real context budget. Ignoring them under-counts large turns and
    lets the buffer blow past the limit before compaction ever triggers.
    """
    total = len(str(m.get("content") or ""))
    for tc in m.get("tool_calls") or []:
        try:
            total += len(json.dumps(tc, default=str))
        except (TypeError, ValueError):
            total += len(str(tc))
    for img in m.get("images") or []:
        # base64 image strings are the bulk of a multimodal turn.
        total += len(str(img))
    return total // 4


def _approx_tokens(messages: list[dict]) -> int:
    return sum(approx_message_tokens(m) for m in messages)


def find_compact_boundary(messages: list[dict]) -> int:
    """Index of the last boundary marker, or -1 if absent."""
    for i in range(len(messages) - 1, -1, -1):
        m = messages[i]
        if m.get("role") == "system" and BOUNDARY_MARKER in (m.get("content") or ""):
            return i
    return -1


def messages_after_boundary(messages: list[dict]) -> list[dict]:
    idx = find_compact_boundary(messages)
    return messages[idx + 1:] if idx >= 0 else list(messages)


# --------- snipCompact ---------------------------------------------------

_SNIP_THINK_RX = re.compile(r"<think(?:ing)?>.*?</think(?:ing)?>", re.DOTALL | re.IGNORECASE)


def snip_compact(messages: list[dict], keep_recent_think: int = 2) -> int:
    """Drop empty / duplicate-consecutive messages AND strip <think>…</think>
    blocks from assistant messages older than the last `keep_recent_think`
    assistant turns. Thinking is reasoning, not decisions — keeping it forever
    just bloats the context. Returns the number of changes made.
    """
    kept: list[dict] = []
    removed = 0
    for m in messages:
        role = m.get("role")
        content = (m.get("content") or "").strip()
        tool_calls = m.get("tool_calls") or []
        # Empty assistant messages with no tool_calls are zombies.
        if role == "assistant" and not content and not tool_calls:
            removed += 1
            continue
        # Drop exact-duplicate consecutive user/system frames.
        if kept and kept[-1].get("role") == role and (kept[-1].get("content") or "") == m.get("content"):
            removed += 1
            continue
        kept.append(m)
    # Strip <think> blocks from older assistant messages.
    asst_idxs = [i for i, m in enumerate(kept) if m.get("role") == "assistant"]
    cutoff = len(asst_idxs) - keep_recent_think
    if cutoff > 0:
        for i in asst_idxs[:cutoff]:
            c = kept[i].get("content") or ""
            if "<think" in c.lower():
                new_c = _SNIP_THINK_RX.sub("", c).strip()
                if new_c != c:
                    kept[i]["content"] = new_c
                    removed += 1
    if removed:
        messages.clear()
        messages.extend(kept)
    return removed


# --------- contextCollapse -----------------------------------------------

def context_collapse(messages: list[dict]) -> int:
    """Merge consecutive same-role plain-text messages. Returns count merged."""
    if len(messages) < 2:
        return 0
    out: list[dict] = []
    merged = 0
    for m in messages:
        role = m.get("role")
        # Don't collapse messages that carry tool_calls or names — they have structure.
        plain = role in ("user", "assistant") and not m.get("tool_calls") and not m.get("name")
        if (
            plain and out
            and out[-1].get("role") == role
            and not out[-1].get("tool_calls")
            and not out[-1].get("name")
        ):
            prev = out[-1]
            prev["content"] = ((prev.get("content") or "") + "\n\n" + (m.get("content") or "")).strip()
            merged += 1
            continue
        out.append(m)
    if merged:
        messages.clear()
        messages.extend(out)
    return merged


# --------- autoCompact ---------------------------------------------------

_FALLBACK_BULLETS_LIMIT = 80


def _bulletize(messages: list[dict]) -> str:
    bullets: list[str] = []
    for m in messages:
        role = m.get("role")
        content = (m.get("content") or "").strip()
        if role == "assistant":
            tail = content.splitlines()[-1][:160] if content else ""
            if tail:
                bullets.append(f"- assistant: {tail}")
            for tc in m.get("tool_calls") or []:
                fn = tc.get("function", {}) or {}
                bullets.append(f"  - called {fn.get('name', '?')}")
        elif role == "tool":
            head = content.splitlines()[0][:160] if content else ""
            bullets.append(f"- tool {m.get('name', '?')}: {head}")
        elif role == "user":
            head = content.splitlines()[0][:120] if content else ""
            if head:
                bullets.append(f"- user: {head}")
    return "\n".join(bullets[-_FALLBACK_BULLETS_LIMIT:])


def auto_compact(
    messages: list[dict],
    *,
    max_tokens: int = 12000,
    keep_recent: int = 12,
    summarize_with_model: callable | None = None,
) -> CompactReport:
    """Compact `messages` in place if it crosses `max_tokens`.

    If `summarize_with_model` is provided, calls it with the older slice and
    expects a string summary back (e.g. an Ollama chat call); otherwise falls
    back to a deterministic bulleted summary.
    """
    before = _approx_tokens(messages)
    if before <= max_tokens or len(messages) <= keep_recent + 2:
        return CompactReport(False, before, before, "noop")

    head: list[dict] = []
    if messages and messages[0].get("role") == "system":
        head.append(messages[0])

    boundary_idx = find_compact_boundary(messages)
    # >= 0 (not > 0): when the boundary marker sits at index 0 it's still a
    # real boundary, so the middle must start AFTER it. With the old `> 0`
    # check, a boundary at index 0 fell through to `len(head)`, pulling the
    # already-compacted summary back into the slice and re-summarizing it
    # every cycle — compounding information loss.
    middle_start = boundary_idx + 1 if boundary_idx >= 0 else len(head)

    tail = messages[-keep_recent:]
    middle = messages[middle_start:-keep_recent] if len(messages) > middle_start + keep_recent else []

    if not middle:
        return CompactReport(False, before, before, "noop")

    summary_text: str | None = None
    if summarize_with_model is not None:
        try:
            summary_text = summarize_with_model(middle)
        except Exception:
            # Deliberate fallback to the deterministic bulletize summary, but
            # log it — a silently-failing LLM summarizer would otherwise be
            # invisible and degrade summary quality with no signal.
            log.warning("LLM summarizer failed; falling back to bulletize", exc_info=True)
            summary_text = None
    if not summary_text:
        summary_text = _bulletize(middle)

    summary_msg = {
        "role": "user",
        "content": "[older context, compacted]\n" + summary_text,
    }
    boundary_msg = {"role": "system", "content": BOUNDARY_MARKER}

    new_messages = head + [summary_msg, boundary_msg] + tail
    messages.clear()
    messages.extend(new_messages)
    after = _approx_tokens(messages)
    return CompactReport(
        triggered=True,
        before=before,
        after=after,
        strategy="autoCompact",
        summary_added=True,
    )


# --------- one-stop ------------------------------------------------------

def manage_context(
    messages: list[dict],
    *,
    max_tokens: int = 12000,
    keep_recent: int = 12,
    summarize_with_model: callable | None = None,
) -> list[CompactReport]:
    """Run the three strategies in order: snip → collapse → auto."""
    reports: list[CompactReport] = []
    # Capture the REAL token count before each mutation rather than fabricating
    # `before` from `after + count * constant` — the old form reported a made-up
    # number that drifted from the actual saving.
    before_snip = _approx_tokens(messages)
    snipped = snip_compact(messages)
    if snipped:
        reports.append(CompactReport(True, before_snip,
                                     _approx_tokens(messages), "snipCompact"))
    before_collapse = _approx_tokens(messages)
    merged = context_collapse(messages)
    if merged:
        reports.append(CompactReport(True, before_collapse,
                                     _approx_tokens(messages), "contextCollapse"))
    r = auto_compact(
        messages,
        max_tokens=max_tokens,
        keep_recent=keep_recent,
        summarize_with_model=summarize_with_model,
    )
    if r.triggered:
        reports.append(r)
    return reports
