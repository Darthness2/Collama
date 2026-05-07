"""Task IDs: short prefixed identifiers per kind of work item.

Mirrors Claude Code's prefix+8char scheme so logs / events can be cheaply
correlated. Used by QueryEngine to tag every tool invocation, every
sub-query, etc.
"""
from __future__ import annotations

import secrets
from enum import Enum


class TaskKind(str, Enum):
    BASH    = "b"   # local_bash
    TOOL    = "t"   # local tool (file ops, github)
    AGENT   = "a"   # local_agent (sub-agent / planner step)
    REMOTE  = "r"   # remote_agent — unused for now
    DREAM   = "d"   # background reflection — unused for now
    FLOW    = "f"   # workflow — unused for now


def new_id(kind: TaskKind | str = TaskKind.TOOL) -> str:
    """`b3f9c2a1` style: 1-char kind prefix + 8 hex chars."""
    if isinstance(kind, TaskKind):
        kind = kind.value
    return f"{kind}{secrets.token_hex(4)}"
