"""Teams + mailboxes + coordinator — multi-agent collaboration tools.

The underlying registry is `collama.teams.TeamRegistry`; these are the
model-facing wrappers grouped together because they all share state.
"""
from __future__ import annotations

from ..coordinator import tick as _coordinator_tick
from .base import ToolContext


def _teams(ctx: ToolContext):
    return getattr(ctx, "teams", None)


def t_team_create(args: dict, ctx: ToolContext) -> str:
    reg = _teams(ctx)
    if reg is None:
        return "ERROR: team registry not available"
    name = args["name"]
    reg.create_team(name)
    return f"OK: team '{name}' ready at {reg.root / name}"


def t_team_delete(args: dict, ctx: ToolContext) -> str:
    reg = _teams(ctx)
    if reg is None:
        return "ERROR: team registry not available"
    name = args["name"]
    if not ctx.confirm("delete team", f"team {name} and all its teammates"):
        return "ERROR: user denied"
    return "OK: deleted" if reg.delete_team(name) else f"ERROR: no team {name}"


def t_team_list(args: dict, ctx: ToolContext) -> str:
    reg = _teams(ctx)
    if reg is None:
        return "ERROR: team registry not available"
    teams = reg.list_teams()
    if not teams:
        return "(no teams)"
    out: list[str] = []
    for t in teams:
        members = reg.list_teammates(t)
        out.append(f"{t}  ({len(members)} member{'s' if len(members) != 1 else ''})")
        for m in members:
            out.append(f"  - {m.short()}")
    return "\n".join(out)


def t_teammate_create(args: dict, ctx: ToolContext) -> str:
    reg = _teams(ctx)
    if reg is None:
        return "ERROR: team registry not available"
    tm = reg.add_teammate(
        team=args["team"],
        name=args["name"],
        role=args.get("role", ""),
        skills=args.get("skills") or [],
    )
    return f"OK: created teammate {tm.short()}"


def t_teammate_delete(args: dict, ctx: ToolContext) -> str:
    reg = _teams(ctx)
    if reg is None:
        return "ERROR: team registry not available"
    return ("OK: deleted"
            if reg.delete_teammate(args["team"], args["name"])
            else f"ERROR: no teammate {args['name']} on {args['team']}")


def t_teammate_list(args: dict, ctx: ToolContext) -> str:
    reg = _teams(ctx)
    if reg is None:
        return "ERROR: team registry not available"
    members = reg.list_teammates(args.get("team"))
    if not members:
        return "(no teammates)"
    return "\n".join(m.short() for m in members)


def t_send_message(args: dict, ctx: ToolContext) -> str:
    """SendMessageTool — request/response across teammates via mailboxes."""
    reg = _teams(ctx)
    if reg is None:
        return "ERROR: team registry not available"
    team = args["team"]
    to = args["to"]
    sender = args.get("from", "lead")
    content = args["content"]
    kind = args.get("kind", "msg")
    tm = reg.deliver(team, to, sender, content, kind=kind)
    if tm is None:
        return f"ERROR: no teammate {to} on team {team}"
    return f"OK: delivered to {team}/{tm.name}  (inbox now {len(tm.mailbox)})"


def t_inbox(args: dict, ctx: ToolContext) -> str:
    reg = _teams(ctx)
    if reg is None:
        return "ERROR: team registry not available"
    tm = reg.get_teammate(args["team"], args["name"])
    if not tm:
        return f"ERROR: no teammate {args['name']} on {args['team']}"
    if not tm.mailbox:
        return f"{tm.team}/{tm.name}: inbox empty"
    out = [f"{tm.team}/{tm.name}: {len(tm.mailbox)} message(s)"]
    for i, m in enumerate(tm.mailbox, 1):
        head = (m.get("content") or "").splitlines()[0][:160]
        out.append(f"  {i}. [{m.get('kind','msg')}] from {m.get('from','?')}: {head}")
    return "\n".join(out)


def t_coordinator_tick(args: dict, ctx: ToolContext) -> str:
    engine = getattr(ctx, "engine", None)
    if engine is None:
        return "ERROR: engine not available"
    results = _coordinator_tick(
        engine,
        team=args.get("team"),
        auto_claim=bool(args.get("auto_claim", False)),
        max_per_teammate=int(args.get("max_per_teammate", 1)),
    )
    if not results:
        return "(no teammates with pending mail or claimable tasks)"
    out = [f"processed {len(results)} teammate(s):"]
    for r in results:
        first = r.answer.splitlines()[0][:140] if r.answer else ""
        claimed = f"  claimed={r.claimed_task_id}" if r.claimed_task_id else ""
        out.append(f"  - {r.teammate}  inbox={r.inbox_count}{claimed}")
        if first:
            out.append(f"      → {first}")
    return "\n".join(out)


def t_coordinator_run(args: dict, ctx: ToolContext) -> str:
    """Tick repeatedly until everyone is idle (or `max_rounds` reached)."""
    engine = getattr(ctx, "engine", None)
    if engine is None:
        return "ERROR: engine not available"
    max_rounds = int(args.get("max_rounds", 5))
    auto_claim = bool(args.get("auto_claim", True))
    team = args.get("team")
    rounds: list[str] = []
    for r in range(1, max_rounds + 1):
        results = _coordinator_tick(engine, team=team, auto_claim=auto_claim)
        if not results:
            break
        rounds.append(f"round {r}: processed {len(results)} teammate(s)")
    return "\n".join(rounds) if rounds else "(idle — nothing to do)"


TOOLS = {
    "team_create":      t_team_create,
    "team_delete":      t_team_delete,
    "team_list":        t_team_list,
    "teammate_create":  t_teammate_create,
    "teammate_delete":  t_teammate_delete,
    "teammate_list":    t_teammate_list,
    "send_message":     t_send_message,
    "inbox":            t_inbox,
    "coordinator_tick": t_coordinator_tick,
    "coordinator_run":  t_coordinator_run,
}


TOOL_SCHEMAS = [
    {"type": "function", "function": {
        "name": "team_create",
        "description": "Create a persistent team (a directory of long-lived teammate personas).",
        "parameters": {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]},
    }},
    {"type": "function", "function": {
        "name": "team_delete",
        "description": "Delete a team and all its teammates. Requires user approval.",
        "parameters": {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]},
    }},
    {"type": "function", "function": {
        "name": "team_list",
        "description": "List all teams and their teammates.",
        "parameters": {"type": "object", "properties": {}},
    }},
    {"type": "function", "function": {
        "name": "teammate_create",
        "description": "Add a teammate to a team. `role` is appended to the system prompt for that teammate; `skills` are tags used by the coordinator's auto-claim.",
        "parameters": {"type": "object", "properties": {
            "team": {"type": "string"},
            "name": {"type": "string"},
            "role": {"type": "string"},
            "skills": {"type": "array", "items": {"type": "string"}},
        }, "required": ["team", "name"]},
    }},
    {"type": "function", "function": {
        "name": "teammate_delete",
        "description": "Remove a teammate from a team.",
        "parameters": {"type": "object", "properties": {
            "team": {"type": "string"}, "name": {"type": "string"},
        }, "required": ["team", "name"]},
    }},
    {"type": "function", "function": {
        "name": "teammate_list",
        "description": "List teammates, optionally filtered to one team.",
        "parameters": {"type": "object", "properties": {"team": {"type": "string"}}},
    }},
    {"type": "function", "function": {
        "name": "send_message",
        "description": "Send a message to a teammate's inbox (request-response protocol). Recipient processes mail next coordinator_tick / coordinator_run.",
        "parameters": {"type": "object", "properties": {
            "team": {"type": "string"},
            "to": {"type": "string", "description": "Recipient teammate name or id."},
            "content": {"type": "string"},
            "from": {"type": "string", "description": "Sender label. Default 'lead'."},
            "kind": {"type": "string", "description": "msg | task | question | reply"},
        }, "required": ["team", "to", "content"]},
    }},
    {"type": "function", "function": {
        "name": "inbox",
        "description": "Read a teammate's pending mailbox.",
        "parameters": {"type": "object", "properties": {
            "team": {"type": "string"}, "name": {"type": "string"},
        }, "required": ["team", "name"]},
    }},
    {"type": "function", "function": {
        "name": "coordinator_tick",
        "description": "One coordinator tick: process every teammate's mailbox by spawning a sub-agent. With auto_claim=true, idle teammates also pick up matching pending tasks from the task graph.",
        "parameters": {"type": "object", "properties": {
            "team": {"type": "string", "description": "Limit to one team. Default: all teams."},
            "auto_claim": {"type": "boolean"},
            "max_per_teammate": {"type": "integer"},
        }},
    }},
    {"type": "function", "function": {
        "name": "coordinator_run",
        "description": "Tick the coordinator repeatedly until no teammate has work (or max_rounds is reached).",
        "parameters": {"type": "object", "properties": {
            "team": {"type": "string"},
            "max_rounds": {"type": "integer"},
            "auto_claim": {"type": "boolean"},
        }},
    }},
]
