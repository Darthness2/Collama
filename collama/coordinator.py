"""Autonomous Agents (Coordinator).

The Coordinator drives long-lived teammates: when ticked, every teammate
with non-empty mailbox is processed by spawning a sub-agent with that
teammate's role + accumulated mail as the prompt. Replies land in the
teammate's transcript and (optionally) are surfaced as parent
notifications. With auto_claim=True, idle teammates also pick up
matching pending tasks from the TaskGraph and execute them.

Implemented as a synchronous tick — the parent calls Coordinator.tick()
or the model calls coordinator_tick() through a tool. Background mode is
just `BackgroundExecutor.submit_dream(prompt, run=tick_callable)`.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .subagent import fork_subagent
from .tasks import TaskGraph
from .teams import TeamRegistry, Teammate

if TYPE_CHECKING:  # pragma: no cover
    from .engine import QueryEngine

_log = logging.getLogger(__name__)


@dataclass
class CoordResult:
    teammate: str
    inbox_count: int
    answer: str
    claimed_task_id: str | None = None


def _claim_one_task(tasks: TaskGraph, tm: Teammate) -> str | None:
    """Find a pending task whose deps are all done, prefer ones whose
    description/title overlaps with the teammate's skills. Mark active+parent."""
    pending = [t for t in tasks.list(status="pending") if all(
        (tasks.get(d) and tasks.get(d).status == "done") for d in t.deps
    )]
    if not pending:
        return None
    if tm.skills:
        skills_lc = [s.lower() for s in tm.skills]
        pending.sort(key=lambda t: -sum(
            int(s in (t.title + " " + t.description).lower()) for s in skills_lc
        ))
    pick = pending[0]
    tasks.update(pick.id, status="active", parent_id=tm.id)
    return pick.id


def tick(
    engine: "QueryEngine",
    *,
    team: str | None = None,
    auto_claim: bool = False,
    max_per_teammate: int = 1,
) -> list[CoordResult]:
    registry: TeamRegistry = engine.teams
    tasks = engine.task_graph
    results: list[CoordResult] = []

    teammates = registry.list_teammates(team)
    for tm in teammates:
        if tm.busy:
            continue

        # One teammate's failure must not abort the whole tick — guard the
        # entire per-teammate flow and turn a raise into an error result.
        try:
            # Optionally claim a task to add to mailbox.
            claimed_id: str | None = None
            if auto_claim and not tm.mailbox:
                claimed_id = _claim_one_task(tasks, tm)
                if claimed_id:
                    t = tasks.get(claimed_id)
                    if t:
                        registry.deliver(
                            tm.team, tm.id, sender="coordinator",
                            kind="task",
                            content=f"[claimed task {t.id}]\n{t.title}\n\n{t.description}",
                        )
                        tm = registry.get_teammate(tm.team, tm.id) or tm

            if not tm.mailbox:
                continue

            # Mark busy and snapshot mailbox.
            tm.busy = True
            registry.update_teammate(tm)
            try:
                mailbox = list(tm.mailbox[:max_per_teammate])
                prompt = _format_prompt(tm, mailbox)
                answer = fork_subagent(
                    engine, prompt,
                    title=f"{tm.team}/{tm.name}",
                    role=tm.role,
                )
                # Drain processed mail; keep any that came in during the run.
                tm = registry.get_teammate(tm.team, tm.id) or tm
                tm.mailbox = tm.mailbox[len(mailbox):]
                tm.transcript.append({
                    "ts": __import__("time").time(),
                    "role": "outbound",
                    "content": answer,
                })
                results.append(CoordResult(
                    teammate=f"{tm.team}/{tm.name}",
                    inbox_count=len(mailbox),
                    answer=answer,
                    claimed_task_id=claimed_id,
                ))
                # If we claimed a task, mark it done (or failed if 'ERROR').
                if claimed_id:
                    t = tasks.get(claimed_id)
                    if t:
                        status = "done" if not answer.startswith("ERROR") else "failed"
                        tasks.update(claimed_id, status=status, result=answer[:2000])
            finally:
                tm.busy = False
                registry.update_teammate(tm)
        except Exception as e:
            _log.warning(
                "coordinator: teammate %s/%s failed: %s",
                tm.team, tm.name, e, exc_info=True,
            )
            results.append(CoordResult(
                teammate=f"{tm.team}/{tm.name}",
                inbox_count=len(tm.mailbox),
                answer=f"ERROR: coordinator failed processing teammate: {type(e).__name__}: {e}",
                claimed_task_id=None,
            ))

    return results


def _format_prompt(tm: Teammate, mailbox: list[dict]) -> str:
    head = f"You are {tm.name}, a teammate on team {tm.team}."
    if tm.role:
        head += f"\nRole: {tm.role}"
    if tm.skills:
        head += f"\nSkills: {', '.join(tm.skills)}"
    body = "\n\n".join(
        f"From {m.get('from','?')} ({m.get('kind','msg')}):\n{m.get('content','')}"
        for m in mailbox
    )
    return f"{head}\n\nIncoming:\n{body}\n\nRespond concisely and act if needed."
