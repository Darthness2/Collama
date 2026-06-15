"""Persistent JSON config for Collama.

Stored at $XDG_CONFIG_HOME/collama/config.json (default ~/.config/collama/config.json).
File is chmod 600 since it can hold a GitHub PAT.
"""
from __future__ import annotations

import json
import logging
import os
import stat
import tempfile
import threading
from pathlib import Path
from typing import Any

_log = logging.getLogger(__name__)

# Serializes config writes within this process so two concurrent saves can't
# race on the temp-file / os.replace dance and lose or corrupt a write.
_save_lock = threading.Lock()


def config_dir() -> Path:
    base = os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
    return Path(base) / "collama"


def config_path() -> Path:
    return config_dir() / "config.json"


_DEFAULTS: dict[str, Any] = {
    "model": None,
    "host": "http://localhost:11434",
    "temperature": 0.2,
    "effort": "medium",
    "yolo": False,
    "github": {"token": None},
}


def _merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def load() -> dict:
    p = config_path()
    if not p.exists():
        return dict(_DEFAULTS)
    try:
        data = json.loads(p.read_text())
    except (json.JSONDecodeError, OSError) as e:
        _log.warning("config load failed, using defaults: %s", e, exc_info=True)
        return dict(_DEFAULTS)
    if not isinstance(data, dict):
        return dict(_DEFAULTS)
    return _merge(_DEFAULTS, data)


def save(cfg: dict) -> None:
    d = config_dir()
    d.mkdir(parents=True, exist_ok=True)
    p = config_path()
    payload = json.dumps(cfg, indent=2)
    # Create the temp file in the same dir with a UNIQUE name and 0o600 perms
    # BEFORE writing, so secrets (e.g. a GitHub PAT) are never world-readable
    # even for a brief window. fsync before the atomic os.replace so a crash
    # can't leave a torn file. A module-level lock serializes concurrent saves.
    with _save_lock:
        fd, tmp_name = tempfile.mkstemp(dir=str(d), prefix="config.", suffix=".json.tmp")
        os.fchmod(fd, stat.S_IRUSR | stat.S_IWUSR)  # 0o600 before any data lands
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(payload)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_name, p)
        except OSError as e:
            _log.warning("config save failed: %s", e, exc_info=True)
            try:
                os.unlink(tmp_name)
            except OSError:
                pass
            raise


def set_value(cfg: dict, dotted_key: str, value: Any) -> dict:
    """Set 'github.token' style keys."""
    parts = dotted_key.split(".")
    cur = cfg
    for k in parts[:-1]:
        if not isinstance(cur.get(k), dict):
            cur[k] = {}
        cur = cur[k]
    cur[parts[-1]] = value
    return cfg


def get_value(cfg: dict, dotted_key: str, default: Any = None) -> Any:
    cur: Any = cfg
    for k in dotted_key.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur
