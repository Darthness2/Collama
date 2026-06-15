"""config_get / config_set / sleep / skill / powershell — misc system tools."""
from __future__ import annotations

import logging
import subprocess

from .base import ToolContext, _analyze_failure, _truncate

logger = logging.getLogger(__name__)

# Keys the model is permitted to write via config_set. Deliberately EXCLUDES
# security-sensitive keys: `yolo` (would disable every approval prompt),
# `github.token` (a secret credential), and anything not listed here. The user
# can still set those by hand in the config file or via CLI flags.
_CONFIG_SET_ALLOWLIST = frozenset({
    "model",
    "host",
    "temperature",
    "effort",
})


def t_config_get(args: dict, ctx: ToolContext) -> str:
    """ConfigTool (get) — read a value from the persistent config."""
    engine = getattr(ctx, "engine", None)
    if engine is None or engine.config is None:
        return "ERROR: config not available"
    from ..config import get_value
    key = args["key"]
    v = get_value(engine.config, key, None)
    if v is None:
        return f"(unset: {key})"
    return f"{key} = {v}"


def t_config_set(args: dict, ctx: ToolContext) -> str:
    engine = getattr(ctx, "engine", None)
    if engine is None or engine.config is None:
        return "ERROR: config not available"
    from ..config import set_value, save
    key = args["key"]
    val = args["value"]
    if key not in _CONFIG_SET_ALLOWLIST:
        allowed = ", ".join(sorted(_CONFIG_SET_ALLOWLIST))
        return (
            f"ERROR: config_set refuses key '{key}'. Only these keys may be set "
            f"via this tool: {allowed}. Security-sensitive keys (yolo, "
            f"github.token) must be changed by the user directly."
        )
    if not ctx.confirm("config set", f"{key} = {val}"):
        return "ERROR: user denied"
    set_value(engine.config, key, val)
    save(engine.config)
    return f"OK: {key} = {val} (saved)"


def t_sleep(args: dict, ctx: ToolContext) -> str:
    """SleepTool — pause for N seconds. Capped at 60s."""
    import time as _time
    secs = float(args.get("seconds", 1))
    secs = max(0.0, min(60.0, secs))
    _time.sleep(secs)
    return f"OK: slept {secs}s"


def t_skill(args: dict, ctx: ToolContext) -> str:
    """SkillTool — append a named skill's instructions onto the engine for the
    rest of the session. Skills live at ~/.config/collama/skills/<name>.md
    (or .txt). The contents become a system-prompt addendum.
    """
    from ..config import config_dir
    op = args.get("op", "use")
    name = args.get("name", "")
    skills_dir = config_dir() / "skills"
    skills_dir.mkdir(parents=True, exist_ok=True)

    if op == "list":
        files = sorted(skills_dir.glob("*"))
        if not files:
            return "(no skills installed; drop *.md files in " + str(skills_dir) + ")"
        return "\n".join(f"- {f.stem}  ({f.stat().st_size} bytes)" for f in files)

    if not name:
        return "ERROR: skill name required"
    candidate = None
    for ext in (".md", ".txt", ""):
        c = skills_dir / f"{name}{ext}"
        if c.exists():
            candidate = c
            break
    if op == "get":
        if not candidate:
            return f"ERROR: no skill '{name}'"
        return _truncate(candidate.read_text(errors="replace"))

    if op == "use":
        if not candidate:
            return f"ERROR: no skill '{name}'"
        engine = getattr(ctx, "engine", None)
        if engine is None:
            return "ERROR: engine not available"
        body = candidate.read_text(errors="replace")
        if engine.messages and engine.messages[0].get("role") == "system":
            engine.messages[0]["content"] += f"\n\n=== SKILL: {name} ===\n{body}"
        return f"OK: skill '{name}' attached to system prompt ({len(body)} chars)"

    return f"ERROR: unknown op '{op}'"


def t_powershell(args: dict, ctx: ToolContext) -> str:
    """PowerShellTool — run a command via pwsh / powershell.exe."""
    import shutil as _shutil
    pwsh = _shutil.which("pwsh") or _shutil.which("powershell")
    if not pwsh:
        return "ERROR: PowerShell not installed (pwsh / powershell.exe not on PATH)"
    cmd = args["command"]
    timeout = int(args.get("timeout", 60))
    if not ctx.confirm("PowerShell command", cmd):
        return "ERROR: user denied command"
    try:
        proc = subprocess.run(
            [pwsh, "-NoProfile", "-NonInteractive", "-Command", cmd],
            cwd=str(ctx.root), capture_output=True,
            encoding="utf-8", errors="replace", timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return f"ERROR: timed out after {timeout}s"
    status = "PASS" if proc.returncode == 0 else "FAIL"
    parts = [f"{status} (exit code {proc.returncode})"]
    if proc.stdout:
        parts.append(f"--- stdout ---\n{proc.stdout}")
    if proc.stderr:
        parts.append(f"--- stderr ---\n{proc.stderr}")
    if proc.returncode != 0:
        hint = _analyze_failure(proc.stdout, proc.stderr)
        if hint:
            parts.append(hint)
    return _truncate("\n".join(parts))


TOOLS = {
    "config_get": t_config_get,
    "config_set": t_config_set,
    "sleep":      t_sleep,
    "skill":      t_skill,
    "powershell": t_powershell,
}


TOOL_SCHEMAS = [
    {"type": "function", "function": {
        "name": "config_get",
        "description": "Read a value from the persistent config (dotted key, e.g. 'github.token').",
        "parameters": {"type": "object", "properties": {"key": {"type": "string"}}, "required": ["key"]},
    }},
    {"type": "function", "function": {
        "name": "config_set",
        "description": "Set a config value. Asks for approval.",
        "parameters": {"type": "object", "properties": {
            "key": {"type": "string"}, "value": {},
        }, "required": ["key", "value"]},
    }},
    {"type": "function", "function": {
        "name": "sleep",
        "description": "Pause execution for `seconds` (capped at 60).",
        "parameters": {"type": "object", "properties": {"seconds": {"type": "number"}}},
    }},
    {"type": "function", "function": {
        "name": "skill",
        "description": "Manage and apply skills (markdown bundles in ~/.config/collama/skills/). op: list | get | use.",
        "parameters": {"type": "object", "properties": {
            "op": {"type": "string", "enum": ["list", "get", "use"]},
            "name": {"type": "string"},
        }},
    }},
    {"type": "function", "function": {
        "name": "powershell",
        "description": "Run a command via pwsh / powershell.exe. Asks for approval.",
        "parameters": {"type": "object", "properties": {
            "command": {"type": "string"}, "timeout": {"type": "integer"},
        }, "required": ["command"]},
    }},
]
