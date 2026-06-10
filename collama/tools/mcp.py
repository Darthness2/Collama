"""MCP control tools — inspect and manage configured MCP servers.

The actual MCP traffic flows through :mod:`collama.mcp`; these tools are
just the surface area the model uses to introspect the registry. The
servers themselves spawn lazily when :func:`collama.mcp.registry` starts
them up (typically the first time ``mcp`` is in the enabled tool groups).
"""
from __future__ import annotations

from .. import mcp as _mcp
from .base import ToolContext, _truncate


def t_mcp_servers(args: dict, ctx: ToolContext) -> str:
    """List every configured MCP server and its state.

    With ``ensure_started=true`` (default), spawn any not-yet-running
    servers first so the listing reflects what tools will actually be
    available to the next turn. Pass ``ensure_started=false`` for a
    read-only view that won't touch processes.
    """
    reg = _mcp.registry()
    if bool(args.get("ensure_started", True)):
        reg.ensure_started()
    servers = reg.servers()
    if not servers:
        return ("(no MCP servers configured)\n"
                "Create ~/.config/collama/mcp.json with:\n"
                '  {"servers": {"<name>": {"command": "<cmd>", "args": [...]}}}')
    lines: list[str] = []
    for name, srv in servers.items():
        tool_count = len(srv.tools) if srv.state == "ready" else 0
        head = f"{name}  state={srv.state}  tools={tool_count}"
        if srv.error:
            head += f"  error={srv.error}"
        lines.append(head)
        for t in srv.tools[:30]:
            desc = (t.get("description") or "").splitlines()[0][:100]
            lines.append(f"    - mcp__{name}__{t.get('name')}  —  {desc}")
        if srv.state == "error" and srv.stderr_tail:
            lines.append("    stderr (tail):")
            for tail_line in srv.stderr_tail.splitlines()[-5:]:
                lines.append(f"      {tail_line}")
    return _truncate("\n".join(lines))


def t_mcp_restart(args: dict, ctx: ToolContext) -> str:
    """Stop and re-spawn a single MCP server. Handy after editing its
    environment or upgrading the server binary."""
    name = args.get("name") or args.get("server")
    if not name:
        return "ERROR: missing argument 'name'"
    if not ctx.confirm("restart MCP server", str(name)):
        return "ERROR: user denied"
    return _mcp.registry().restart(str(name))


TOOLS = {
    "mcp_servers": t_mcp_servers,
    "mcp_restart": t_mcp_restart,
}


TOOL_SCHEMAS = [
    {"type": "function", "function": {
        "name": "mcp_servers",
        "description": (
            "List every MCP server configured in ~/.config/collama/mcp.json, "
            "its state (stopped/starting/ready/error), and the tools it "
            "exposes. Auto-starts any not-yet-running servers unless "
            "ensure_started=false."
        ),
        "parameters": {"type": "object", "properties": {
            "ensure_started": {"type": "boolean"},
        }},
    }},
    {"type": "function", "function": {
        "name": "mcp_restart",
        "description": "Stop and respawn one configured MCP server.",
        "parameters": {"type": "object", "properties": {
            "name": {"type": "string"},
        }, "required": ["name"]},
    }},
]
