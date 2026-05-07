"""Saved conversation sessions on disk.

Each session is JSON at ~/.config/collama/sessions/<id>.json with:
    {id, title, model, created_at, updated_at, messages: [...]}
"""
from __future__ import annotations

import json
import os
import re
import stat
import time
import uuid
from pathlib import Path
from typing import Any

from .config import config_dir


def sessions_dir() -> Path:
    d = config_dir() / "sessions"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _path(session_id: str) -> Path:
    return sessions_dir() / f"{session_id}.json"


def new_id() -> str:
    return uuid.uuid4().hex[:12]


def make(model: str, title: str | None = None) -> dict[str, Any]:
    now = int(time.time())
    return {
        "id": new_id(),
        "title": title or "untitled",
        "model": model,
        "created_at": now,
        "updated_at": now,
        "messages": [],
    }


def derive_title(messages: list[dict]) -> str:
    for m in messages:
        if m.get("role") == "user":
            text = (m.get("content") or "").strip()
            text = re.sub(r"\s+", " ", text)
            return text[:60] if text else "untitled"
    return "untitled"


def save(session: dict) -> Path:
    session["updated_at"] = int(time.time())
    if session.get("title") in (None, "", "untitled"):
        session["title"] = derive_title(session.get("messages", []))
    p = _path(session["id"])
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(session, indent=2))
    tmp.replace(p)
    try:
        os.chmod(p, stat.S_IRUSR | stat.S_IWUSR)
    except OSError:
        pass
    return p


def load(session_id: str) -> dict | None:
    p = _path(session_id)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def delete(session_id: str) -> bool:
    p = _path(session_id)
    if p.exists():
        p.unlink()
        return True
    return False


def list_all() -> list[dict]:
    out = []
    for p in sessions_dir().glob("*.json"):
        try:
            data = json.loads(p.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        out.append({
            "id": data.get("id", p.stem),
            "title": data.get("title", "untitled"),
            "model": data.get("model", "?"),
            "updated_at": data.get("updated_at", 0),
            "turns": sum(1 for m in data.get("messages", []) if m.get("role") == "user"),
        })
    out.sort(key=lambda s: s["updated_at"], reverse=True)
    return out


def fmt_time(ts: int) -> str:
    if not ts:
        return "?"
    delta = int(time.time()) - ts
    if delta < 60:
        return f"{delta}s ago"
    if delta < 3600:
        return f"{delta // 60}m ago"
    if delta < 86400:
        return f"{delta // 3600}h ago"
    return f"{delta // 86400}d ago"
