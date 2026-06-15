"""Persistent task graph + task IDs.

Tasks are JSON files at ~/.config/collama/tasks/<id>.json. The model calls
task_create / task_update / task_get / task_list / task_delete tools to
maintain a persistent record of what it's working on, including
dependencies and status.

Each task can carry a `worktree` (directory). enter_worktree pushes
the current workspace onto a stack and switches to that dir; exit_worktree
pops back. The worktree is bound by task id when created via task_create
with worktree=<path>.
"""
from __future__ import annotations

import json
import logging
import secrets
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path

from .config import config_dir

logger = logging.getLogger(__name__)


class TaskKind(str, Enum):
    BASH    = "b"
    TOOL    = "t"
    AGENT   = "a"   # local_agent / sub-agent
    REMOTE  = "r"
    DREAM   = "d"   # background reflection
    FLOW    = "f"


VALID_STATUS = {"pending", "active", "done", "blocked", "failed", "cancelled"}


def new_id(kind: TaskKind | str = TaskKind.TOOL) -> str:
    if isinstance(kind, TaskKind):
        kind = kind.value
    return f"{kind}{secrets.token_hex(4)}"


@dataclass
class Task:
    id: str
    title: str
    description: str = ""
    status: str = "pending"
    deps: list[str] = field(default_factory=list)
    parent_id: str | None = None
    kind: str = "t"
    result: str = ""
    worktree: str | None = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_raw(cls, raw: object) -> "Task | None":
        """Build a Task from untrusted on-disk JSON.

        Whitelists and coerces known fields instead of splatting ``**raw`` into
        the constructor (which would let a crafted/corrupt file inject unknown
        kwargs or wrong types). Unknown keys are ignored; required fields (id,
        title) must be present and string-coercible. Returns None on bad data.
        """
        if not isinstance(raw, dict):
            return None
        try:
            tid = raw["id"]
            title = raw["title"]
        except (KeyError, TypeError):
            return None
        if not isinstance(tid, str) or not isinstance(title, str):
            return None

        def _str(key: str, default: str = "") -> str:
            v = raw.get(key, default)
            return v if isinstance(v, str) else default

        def _opt_str(key: str):
            v = raw.get(key)
            return v if isinstance(v, str) else None

        def _float(key: str) -> float:
            v = raw.get(key)
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                return float(v)
            return time.time()

        deps_raw = raw.get("deps")
        deps = [d for d in deps_raw if isinstance(d, str)] if isinstance(deps_raw, list) else []

        return cls(
            id=tid,
            title=title,
            description=_str("description"),
            status=_str("status", "pending"),
            deps=deps,
            parent_id=_opt_str("parent_id"),
            kind=_str("kind", "t"),
            result=_str("result"),
            worktree=_opt_str("worktree"),
            created_at=_float("created_at"),
            updated_at=_float("updated_at"),
        )

    def short(self) -> str:
        title = self.title if len(self.title) < 60 else self.title[:57] + "…"
        wt = f"  [wt:{self.worktree}]" if self.worktree else ""
        return f"{self.id}  [{self.status:<8}]  {title}{wt}"


class TaskGraph:
    """File-backed CRUD for Task objects."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = root or (config_dir() / "tasks")
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, task_id: str) -> Path:
        return self.root / f"{task_id}.json"

    def create(
        self,
        title: str,
        *,
        description: str = "",
        deps: list[str] | None = None,
        parent_id: str | None = None,
        kind: TaskKind | str = TaskKind.TOOL,
        worktree: str | None = None,
    ) -> Task:
        task = Task(
            id=new_id(kind),
            title=title,
            description=description,
            deps=list(deps or []),
            parent_id=parent_id,
            kind=kind.value if isinstance(kind, TaskKind) else str(kind),
            worktree=worktree,
        )
        self._write(task)
        return task

    def get(self, task_id: str) -> Task | None:
        p = self._path(task_id)
        if not p.exists():
            # Allow id prefixes ("t12ab..." → "t12ab*").
            for q in self.root.glob(f"{task_id}*.json"):
                p = q
                break
            else:
                return None
        try:
            raw = json.loads(p.read_text())
        except (json.JSONDecodeError, OSError):
            logger.warning("failed to read task file %s", p, exc_info=True)
            return None
        task = Task.from_raw(raw)
        if task is None:
            logger.warning("ignoring malformed task file %s", p)
        return task

    def update(self, task_id: str, **changes) -> Task | None:
        task = self.get(task_id)
        if not task:
            return None
        if "status" in changes and changes["status"] not in VALID_STATUS:
            raise ValueError(f"invalid status: {changes['status']}")
        for k, v in changes.items():
            if hasattr(task, k):
                setattr(task, k, v)
        task.updated_at = time.time()
        self._write(task)
        return task

    def list(self, *, status: str | None = None, parent_id: str | None = None) -> list[Task]:
        out: list[Task] = []
        for p in sorted(self.root.glob("*.json")):
            try:
                raw = json.loads(p.read_text())
            except (json.JSONDecodeError, OSError):
                logger.warning("failed to read task file %s", p, exc_info=True)
                continue
            t = Task.from_raw(raw)
            if t is None:
                logger.warning("ignoring malformed task file %s", p)
                continue
            if status and t.status != status:
                continue
            if parent_id and t.parent_id != parent_id:
                continue
            out.append(t)
        out.sort(key=lambda t: t.updated_at, reverse=True)
        return out

    def delete(self, task_id: str) -> bool:
        task = self.get(task_id)
        if not task:
            return False
        try:
            self._path(task.id).unlink()
            return True
        except OSError:
            logger.warning("failed to delete task file for %s", task.id, exc_info=True)
            return False

    def _write(self, task: Task) -> None:
        p = self._path(task.id)
        tmp = p.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(task.to_dict(), indent=2))
        tmp.replace(p)
