"""recordTranscript(): append-only JSONL of every message in a session.

Files live at ~/.config/collama/transcripts/<session_id>.jsonl. One JSON
object per line, recorded as soon as a message is finalized so a crash
mid-turn still leaves a usable history.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from ..config import config_dir


def transcripts_dir() -> Path:
    d = config_dir() / "transcripts"
    d.mkdir(parents=True, exist_ok=True)
    return d


def transcript_path(session_id: str) -> Path:
    return transcripts_dir() / f"{session_id}.jsonl"


def record(session_id: str, role: str, content: Any, **extra) -> None:
    if not session_id:
        return
    line: dict[str, Any] = {
        "ts": time.time(),
        "role": role,
        "content": content if isinstance(content, str) else str(content),
    }
    line.update(extra)
    p = transcript_path(session_id)
    try:
        with p.open("a", encoding="utf-8") as f:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
    except OSError:
        pass
