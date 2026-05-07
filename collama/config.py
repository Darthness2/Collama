"""Persistent JSON config for Collama.

Stored at $XDG_CONFIG_HOME/collama/config.json (default ~/.config/collama/config.json).
File is chmod 600 since it can hold a GitHub PAT.
"""
from __future__ import annotations

import json
import os
import stat
from pathlib import Path
from typing import Any


def config_dir() -> Path:
    base = os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
    return Path(base) / "collama"


def config_path() -> Path:
    return config_dir() / "config.json"


_DEFAULTS: dict[str, Any] = {
    "model": None,
    "host": "http://localhost:11434",
    "temperature": 0.2,
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
    except (json.JSONDecodeError, OSError):
        return dict(_DEFAULTS)
    if not isinstance(data, dict):
        return dict(_DEFAULTS)
    return _merge(_DEFAULTS, data)


def save(cfg: dict) -> None:
    d = config_dir()
    d.mkdir(parents=True, exist_ok=True)
    p = config_path()
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(cfg, indent=2))
    tmp.replace(p)
    try:
        os.chmod(p, stat.S_IRUSR | stat.S_IWUSR)
    except OSError:
        pass


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
