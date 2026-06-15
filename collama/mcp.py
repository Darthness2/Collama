"""Model Context Protocol client.

Speaks the stdio NDJSON transport of MCP (modelcontextprotocol.io):
each request is a JSON-RPC envelope on its own line of the server's
stdin; responses arrive line-by-line on stdout. Supports the minimal
set we need to plug third-party MCP servers into Collama's tool loop:

    initialize
    notifications/initialized
    tools/list
    tools/call

Servers are configured in ``~/.config/collama/mcp.json``::

    {
      "servers": {
        "everything": {
          "command": "npx",
          "args": ["-y", "@modelcontextprotocol/server-everything"]
        },
        "github": {
          "command": "uvx",
          "args": ["mcp-server-github"],
          "env": {"GITHUB_TOKEN": "${GITHUB_TOKEN}"}
        }
      }
    }

Discovered tools are exposed to the agent as ``mcp__<server>__<tool>`` so
multiple servers can expose tools with the same name without collisions.
Servers spawn lazily on the first call to :func:`registry` (typically via
``all_tool_schemas`` when the ``mcp`` tool group is enabled) and are
torn down via ``atexit``.
"""
from __future__ import annotations

import atexit
import json
import logging
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from queue import Empty, Queue
from typing import Any

from .config import config_dir

_log = logging.getLogger(__name__)


# Protocol version Collama negotiates. Servers built against newer or
# older drafts negotiate this down; "2024-11-05" is the broadly-deployed
# baseline and what every public SDK targets at minimum.
PROTOCOL_VERSION = "2024-11-05"

CLIENT_NAME = "collama"


class MCPError(RuntimeError):
    """Raised when MCP startup, dispatch, or shutdown can't recover."""


# ----------------------------------------------------------------------
# Per-server state
# ----------------------------------------------------------------------


@dataclass
class _ServerConfig:
    """Parsed entry from mcp.json — describes how to spawn one server."""
    name: str
    command: str
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    cwd: str | None = None


class _Server:
    """One running MCP server: subprocess + JSON-RPC plumbing + tool cache."""

    def __init__(self, cfg: _ServerConfig) -> None:
        self.cfg = cfg
        self.proc: subprocess.Popen | None = None
        self.tools: list[dict] = []      # raw MCP tool defs from tools/list
        self.state: str = "stopped"      # stopped | starting | ready | error
        self.error: str | None = None
        self._next_id: int = 0
        self._pending: dict[int, Queue] = {}
        self._stderr_tail: list[str] = []
        self._reader: threading.Thread | None = None
        self._stderr_reader: threading.Thread | None = None
        self._send_lock = threading.Lock()
        self._id_lock = threading.Lock()

    # -- lifecycle --------------------------------------------------------

    def start(self, timeout: float = 30.0) -> None:
        """Spawn the process, handshake, and pull the tool list. Idempotent."""
        if self.state == "ready":
            return
        self.state = "starting"
        cmd = [self.cfg.command, *self.cfg.args]
        exe = shutil.which(self.cfg.command)
        if exe is None:
            self.state = "error"
            self.error = f"command not found on PATH: {self.cfg.command}"
            raise MCPError(self.error)

        # Inherit env, overlay/expand the server's own env block.
        env = dict(os.environ)
        for k, v in self.cfg.env.items():
            env[k] = os.path.expandvars(v) if isinstance(v, str) else str(v)

        try:
            self.proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=self.cfg.cwd,
                env=env,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,  # line-buffered — important for NDJSON over a pipe
                # Own session/process group so we can signal the whole tree:
                # `npx` spawns `node` grandchildren that would otherwise leak
                # when we only kill the immediate child.
                start_new_session=True,
            )
        except OSError as e:
            self.state = "error"
            self.error = f"spawn failed: {e}"
            raise MCPError(self.error) from e

        # Background reader threads. The stderr reader is critical: an
        # unread stderr buffer eventually backs up and blocks the server.
        self._reader = threading.Thread(
            target=self._read_loop, name=f"mcp-{self.cfg.name}-stdout", daemon=True,
        )
        self._reader.start()
        self._stderr_reader = threading.Thread(
            target=self._stderr_loop, name=f"mcp-{self.cfg.name}-stderr", daemon=True,
        )
        self._stderr_reader.start()

        deadline = time.monotonic() + timeout
        # 1. initialize handshake
        try:
            self._request("initialize", {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": CLIENT_NAME, "version": _client_version()},
            }, timeout=max(1.0, deadline - time.monotonic()))
            # 2. notify ready
            self._notify("notifications/initialized")
            # 3. discover tools
            result = self._request("tools/list", {},
                                   timeout=max(1.0, deadline - time.monotonic()))
            tools = result.get("tools") if isinstance(result, dict) else None
            self.tools = list(tools or [])
        except MCPError:
            self.state = "error"
            raise
        except Exception as e:
            self.state = "error"
            self.error = f"handshake failed: {e}"
            raise MCPError(self.error) from e

        self.state = "ready"

    def _signal_group(self, sig: int) -> None:
        """Signal the whole process group (npx → node → ...) if we have one,
        falling back to the single process otherwise."""
        proc = self.proc
        if proc is None:
            return
        try:
            os.killpg(os.getpgid(proc.pid), sig)
        except (ProcessLookupError, PermissionError):
            pass
        except (OSError, AttributeError):
            # No process groups (e.g. Windows) or pgid lookup failed —
            # signal just the immediate child.
            try:
                if sig == signal.SIGKILL:
                    proc.kill()
                else:
                    proc.terminate()
            except OSError:
                pass

    def stop(self, timeout: float = 3.0) -> None:
        """Best-effort: close stdin, wait briefly, terminate, then kill — and
        always reap the child so we don't leak zombies. Kills the whole
        process group so grandchildren (npx→node) go down too."""
        proc = self.proc
        if proc is None:
            self.state = "stopped"
            return
        try:
            if proc.stdin and not proc.stdin.closed:
                try:
                    proc.stdin.close()
                except OSError:
                    pass
            try:
                proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                self._signal_group(signal.SIGTERM)
                try:
                    proc.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    self._signal_group(signal.SIGKILL)
                    # Reap after the kill so the child doesn't linger as a
                    # zombie holding fds / pgid.
                    try:
                        proc.wait(timeout=timeout)
                    except subprocess.TimeoutExpired:
                        _log.warning(
                            "MCP server '%s' did not exit after SIGKILL",
                            self.cfg.name,
                        )
        finally:
            self.proc = None
            self.state = "stopped"

    # -- public API -------------------------------------------------------

    def call_tool(self, tool_name: str, arguments: dict, timeout: float = 60.0) -> str:
        """Invoke a tool via tools/call. Returns the concatenated text
        content blocks the server returned. Errors and isError=True
        responses become an "ERROR: ..." string so the engine surfaces
        them like any other tool failure."""
        if self.state != "ready":
            return f"ERROR: MCP server '{self.cfg.name}' not ready (state={self.state})"
        try:
            result = self._request(
                "tools/call",
                {"name": tool_name, "arguments": arguments or {}},
                timeout=timeout,
            )
        except MCPError as e:
            return f"ERROR: MCP call failed: {e}"

        if not isinstance(result, dict):
            return f"ERROR: malformed MCP tools/call result: {result!r}"

        # Render content blocks into plain text. Anything we don't
        # understand is dumped as JSON so it's at least inspectable.
        blocks = result.get("content") or []
        parts: list[str] = []
        for b in blocks:
            if not isinstance(b, dict):
                parts.append(json.dumps(b))
                continue
            t = b.get("type")
            if t == "text":
                parts.append(str(b.get("text", "")))
            elif t == "resource":
                # Resource block: try to surface the inline text if present.
                res = b.get("resource") or {}
                parts.append(str(res.get("text") or json.dumps(b)))
            else:
                parts.append(json.dumps(b))
        text = "\n".join(parts).strip() or "(no content)"
        if result.get("isError"):
            return f"ERROR: {text}"
        return text

    # -- JSON-RPC plumbing ------------------------------------------------

    def _next_request_id(self) -> int:
        with self._id_lock:
            self._next_id += 1
            return self._next_id

    def _send_raw(self, payload: dict) -> None:
        proc = self.proc
        if proc is None or proc.stdin is None or proc.stdin.closed:
            raise MCPError("server stdin closed")
        line = json.dumps(payload, separators=(",", ":")) + "\n"
        with self._send_lock:
            try:
                proc.stdin.write(line)
                proc.stdin.flush()
            except (BrokenPipeError, OSError) as e:
                raise MCPError(f"write to server failed: {e}") from e

    def _request(self, method: str, params: dict | None, *, timeout: float) -> Any:
        req_id = self._next_request_id()
        slot: Queue = Queue(maxsize=1)
        self._pending[req_id] = slot
        try:
            self._send_raw({
                "jsonrpc": "2.0",
                "id": req_id,
                "method": method,
                "params": params or {},
            })
            try:
                resp = slot.get(timeout=timeout)
            except Empty:
                raise MCPError(
                    f"{method} timed out after {timeout:.1f}s "
                    f"(server '{self.cfg.name}')"
                )
        finally:
            self._pending.pop(req_id, None)

        if not isinstance(resp, dict):
            raise MCPError(f"{method}: non-dict response {resp!r}")
        if "error" in resp:
            err = resp["error"] or {}
            raise MCPError(f"{method}: {err.get('message') or err}")
        return resp.get("result")

    def _notify(self, method: str, params: dict | None = None) -> None:
        self._send_raw({
            "jsonrpc": "2.0",
            "method": method,
            "params": params or {},
        })

    def _read_loop(self) -> None:
        proc = self.proc
        if proc is None or proc.stdout is None:
            return
        for line in proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                # Some servers log to stdout. Skip junk lines but keep a
                # tail in the stderr buffer so debugging stays possible.
                self._stderr_tail.append(f"(stdout non-JSON) {line[:200]}")
                del self._stderr_tail[:-50]
                continue
            if not isinstance(msg, dict):
                continue
            rid = msg.get("id")
            if rid is None:
                # Notification or log from server — currently we don't
                # forward these. Future: hook for resources/list changes.
                continue
            slot = self._pending.get(rid)
            if slot is not None:
                try:
                    slot.put_nowait(msg)
                except Exception:
                    # Slot already filled (duplicate/stale response id) — the
                    # waiter has its answer; drop the extra rather than block.
                    _log.warning(
                        "MCP server '%s': dropped duplicate response id=%r",
                        self.cfg.name, rid, exc_info=True,
                    )

    def _stderr_loop(self) -> None:
        proc = self.proc
        if proc is None or proc.stderr is None:
            return
        for line in proc.stderr:
            line = line.rstrip("\n")
            self._stderr_tail.append(line)
            del self._stderr_tail[:-200]

    @property
    def stderr_tail(self) -> str:
        return "\n".join(self._stderr_tail[-40:])


# ----------------------------------------------------------------------
# Registry — owns all configured servers
# ----------------------------------------------------------------------


class MCPRegistry:
    """Owns the set of configured MCP servers across the session.

    A single instance is shared via :func:`registry`. Servers are read
    from ``mcp.json`` at construction but **not** started until the first
    schema lookup or tool dispatch.
    """

    def __init__(self) -> None:
        self._servers: dict[str, _Server] = {}
        self._config_path: Path | None = None
        self._lock = threading.Lock()
        self._started = False

    # -- config -----------------------------------------------------------

    def load_from_file(self, path: Path) -> None:
        """Parse a JSON file mapping server name → {command, args, env, cwd}.

        Silent no-op if the file is missing or empty; raises only on
        outright malformed JSON. Reloading replaces the prior set
        (no merging across reloads).
        """
        self._config_path = path
        if not path.exists():
            return
        try:
            raw = json.loads(path.read_text(encoding="utf-8") or "{}")
        except json.JSONDecodeError as e:
            raise MCPError(f"mcp.json: {e}") from e

        entries = raw.get("servers") if isinstance(raw, dict) else None
        if not isinstance(entries, dict):
            return

        new_servers: dict[str, _Server] = {}
        for name, spec in entries.items():
            if not isinstance(spec, dict):
                continue
            command = spec.get("command")
            if not isinstance(command, str) or not command:
                continue
            cfg = _ServerConfig(
                name=str(name),
                command=command,
                args=[str(a) for a in (spec.get("args") or [])],
                env={str(k): str(v) for k, v in (spec.get("env") or {}).items()},
                cwd=spec.get("cwd"),
            )
            new_servers[cfg.name] = _Server(cfg)

        # Stop any previously running server that isn't in the new config.
        for name, srv in list(self._servers.items()):
            if name not in new_servers:
                srv.stop()
                del self._servers[name]

        for name, srv in new_servers.items():
            if name not in self._servers:
                self._servers[name] = srv

    # -- lifecycle --------------------------------------------------------

    def ensure_started(self) -> list[str]:
        """Idempotently start every configured server. Returns the names
        that ended in 'error' state so callers can surface partial failure."""
        with self._lock:
            failed: list[str] = []
            for name, srv in self._servers.items():
                if srv.state == "ready":
                    continue
                try:
                    srv.start()
                except MCPError:
                    failed.append(name)
            self._started = True
            return failed

    def restart(self, name: str) -> str:
        srv = self._servers.get(name)
        if srv is None:
            return f"ERROR: no MCP server '{name}'"
        srv.stop()
        srv.state = "stopped"
        srv.error = None
        srv.tools = []
        try:
            srv.start()
        except MCPError as e:
            return f"ERROR: restart '{name}' failed: {e}"
        return f"OK: '{name}' restarted, {len(srv.tools)} tool(s) discovered"

    def shutdown(self) -> None:
        with self._lock:
            for srv in self._servers.values():
                try:
                    srv.stop()
                except Exception:
                    _log.warning(
                        "error stopping MCP server '%s' during shutdown",
                        srv.cfg.name, exc_info=True,
                    )

    # -- public read-only access -----------------------------------------

    def servers(self) -> dict[str, _Server]:
        return dict(self._servers)

    def all_tool_schemas(self) -> list[dict]:
        """Convert every ready server's MCP tool defs into Collama schemas,
        namespaced as ``mcp__<server>__<tool>``."""
        out: list[dict] = []
        for name, srv in self._servers.items():
            if srv.state != "ready":
                continue
            for t in srv.tools:
                if not isinstance(t, dict):
                    continue
                tname = t.get("name")
                if not isinstance(tname, str) or not tname:
                    continue
                out.append({
                    "type": "function",
                    "function": {
                        "name": f"mcp__{name}__{tname}",
                        "description": (t.get("description") or "")[:1000],
                        "parameters": t.get("inputSchema")
                                       or {"type": "object", "properties": {}},
                    },
                })
        return out

    def dispatch(self, prefixed_name: str, args: dict) -> str:
        """Route ``mcp__<server>__<tool>`` to the right server."""
        if not prefixed_name.startswith("mcp__"):
            return f"ERROR: not an MCP tool: {prefixed_name}"
        parts = prefixed_name.split("__", 2)
        if len(parts) != 3:
            return (f"ERROR: malformed MCP tool name '{prefixed_name}' — "
                    f"expected mcp__<server>__<tool>")
        _, server, tool_name = parts
        srv = self._servers.get(server)
        if srv is None:
            return f"ERROR: no MCP server '{server}' configured"
        if srv.state != "ready":
            try:
                srv.start()
            except MCPError as e:
                return f"ERROR: MCP server '{server}' failed to start: {e}"
        return srv.call_tool(tool_name, args)


# ----------------------------------------------------------------------
# Module-level singleton
# ----------------------------------------------------------------------


_registry: MCPRegistry | None = None
_registry_lock = threading.Lock()


def registry() -> MCPRegistry:
    """Return the process-wide :class:`MCPRegistry`, constructing and
    auto-loading from ``~/.config/collama/mcp.json`` on first call."""
    global _registry
    with _registry_lock:
        if _registry is None:
            r = MCPRegistry()
            cfg_path = config_dir() / "mcp.json"
            try:
                r.load_from_file(cfg_path)
            except MCPError:
                # Bad mcp.json shouldn't crash the agent — surfaced via
                # mcp_servers control tool instead.
                _log.warning("failed to load mcp.json from %s", cfg_path, exc_info=True)
            atexit.register(r.shutdown)
            _registry = r
        return _registry


def _client_version() -> str:
    # Avoid a hard import of collama (would create a small cycle during
    # collama.__init__ if MCP is imported very early).
    try:
        from . import __version__
        return __version__
    except Exception:
        return "0+unknown"
