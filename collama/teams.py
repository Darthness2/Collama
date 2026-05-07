"""s09 — Agent Teams.

A *team* is a directory of long-lived *teammate* personas. Each teammate
has an id, a role (system-prompt addendum), a mailbox (pending messages
sent by other agents), and a transcript (everything ever delivered or
emitted). State persists to ~/.config/collama/teams/<team>/<id>.json so
teammates outlive a session.

Mailbox shape: [{ts, from, kind, content}, ...]
Transcript shape: [{ts, role, from?, content}, ...]
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .config import config_dir
from .tasks import new_id, TaskKind


def teams_root() -> Path:
    d = config_dir() / "teams"
    d.mkdir(parents=True, exist_ok=True)
    return d


@dataclass
class Teammate:
    id: str
    team: str
    name: str
    role: str = ""                     # system-prompt addendum
    skills: list[str] = field(default_factory=list)  # advisory tags for s11 auto-claim
    mailbox: list[dict] = field(default_factory=list)
    transcript: list[dict] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    busy: bool = False                  # true while coordinator is processing this teammate

    def to_dict(self) -> dict:
        return asdict(self)

    def short(self) -> str:
        bar = f"  [busy]" if self.busy else ""
        skills = f"  {{{', '.join(self.skills)}}}" if self.skills else ""
        return f"{self.team}/{self.name}  ({self.id})  inbox={len(self.mailbox)}{bar}{skills}"


class TeamRegistry:
    """File-backed CRUD for teams and teammates."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = root or teams_root()
        self.root.mkdir(parents=True, exist_ok=True)

    # -- teams -----------------------------------------------------------

    def create_team(self, name: str) -> Path:
        d = self.root / name
        d.mkdir(parents=True, exist_ok=True)
        return d

    def delete_team(self, name: str) -> bool:
        d = self.root / name
        if not d.exists():
            return False
        for f in d.glob("*.json"):
            f.unlink()
        d.rmdir()
        return True

    def list_teams(self) -> list[str]:
        return sorted(p.name for p in self.root.iterdir() if p.is_dir())

    # -- teammates -------------------------------------------------------

    def add_teammate(
        self, team: str, name: str, role: str = "", skills: list[str] | None = None,
    ) -> Teammate:
        self.create_team(team)
        tm = Teammate(
            id=new_id(TaskKind.AGENT),
            team=team, name=name, role=role,
            skills=list(skills or []),
        )
        self._write(tm)
        return tm

    def get_teammate(self, team: str, name_or_id: str) -> Teammate | None:
        d = self.root / team
        if not d.exists():
            return None
        # try id first (filename), then name
        p = d / f"{name_or_id}.json"
        if not p.exists():
            for q in d.glob("*.json"):
                try:
                    data = json.loads(q.read_text())
                except (json.JSONDecodeError, OSError):
                    continue
                if data.get("name") == name_or_id or data.get("id") == name_or_id:
                    p = q
                    break
            else:
                return None
        try:
            return Teammate(**json.loads(p.read_text()))
        except (json.JSONDecodeError, OSError, TypeError):
            return None

    def list_teammates(self, team: str | None = None) -> list[Teammate]:
        out: list[Teammate] = []
        teams = [team] if team else self.list_teams()
        for t in teams:
            d = self.root / t
            if not d.exists():
                continue
            for p in sorted(d.glob("*.json")):
                try:
                    out.append(Teammate(**json.loads(p.read_text())))
                except (json.JSONDecodeError, OSError, TypeError):
                    continue
        return out

    def update_teammate(self, tm: Teammate) -> Teammate:
        tm.updated_at = time.time()
        self._write(tm)
        return tm

    def delete_teammate(self, team: str, name_or_id: str) -> bool:
        tm = self.get_teammate(team, name_or_id)
        if not tm:
            return False
        p = self.root / team / f"{tm.id}.json"
        try:
            p.unlink()
            return True
        except OSError:
            return False

    # -- protocol: deliver a message to a teammate's mailbox -------------

    def deliver(
        self, team: str, recipient: str, sender: str, content: str, *, kind: str = "msg",
    ) -> Teammate | None:
        tm = self.get_teammate(team, recipient)
        if not tm:
            return None
        entry = {"ts": time.time(), "from": sender, "kind": kind, "content": content}
        tm.mailbox.append(entry)
        tm.transcript.append({"ts": entry["ts"], "role": "inbound", "from": sender, "content": content})
        return self.update_teammate(tm)

    # -- internals -------------------------------------------------------

    def _write(self, tm: Teammate) -> None:
        d = self.root / tm.team
        d.mkdir(parents=True, exist_ok=True)
        p = d / f"{tm.id}.json"
        tmp = p.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(tm.to_dict(), indent=2))
        tmp.replace(p)
