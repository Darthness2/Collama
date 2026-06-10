"""MCP client tests.

Spawn a real subprocess (a tiny inline Python "MCP server") instead of
monkey-patching subprocess.Popen — that way the JSON-RPC stdio plumbing,
the threaded reader, the lifecycle, and the request/response correlation
all get exercised end-to-end. The fake server understands just enough of
the MCP wire protocol to handshake, list tools, and call them.
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

from collama.mcp import MCPError, MCPRegistry, _Server, _ServerConfig, registry


# A self-contained "MCP server" that lives in a Python -c string. Tiny
# enough to read inline; correct enough to handshake + list + call.
FAKE_SERVER = r'''
import sys, json
def reply(req, result):
    sys.stdout.write(json.dumps({"jsonrpc":"2.0","id":req["id"],"result":result})+"\n")
    sys.stdout.flush()
def reply_err(req, code, msg):
    sys.stdout.write(json.dumps({"jsonrpc":"2.0","id":req["id"],"error":{"code":code,"message":msg}})+"\n")
    sys.stdout.flush()
for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    try:
        req = json.loads(line)
    except Exception:
        continue
    m = req.get("method")
    if "id" not in req:
        continue  # JSON-RPC notification — no reply expected
    if m == "initialize":
        reply(req, {"protocolVersion": "2024-11-05", "capabilities": {},
                    "serverInfo": {"name": "fake", "version": "0"}})
    elif m == "tools/list":
        reply(req, {"tools": [
            {"name": "echo", "description": "Echo back the msg arg.",
             "inputSchema": {"type": "object",
                             "properties": {"msg": {"type": "string"}},
                             "required": ["msg"]}},
            {"name": "fail", "description": "Always returns isError=true.",
             "inputSchema": {"type": "object", "properties": {}}},
        ]})
    elif m == "tools/call":
        name = (req.get("params") or {}).get("name")
        args = (req.get("params") or {}).get("arguments") or {}
        if name == "echo":
            reply(req, {"content": [{"type": "text", "text": f"echo: {args.get('msg','')}"}],
                        "isError": False})
        elif name == "fail":
            reply(req, {"content": [{"type": "text", "text": "intentional failure"}],
                        "isError": True})
        else:
            reply_err(req, -32601, f"no such tool: {name}")
    else:
        reply_err(req, -32601, f"unknown method: {m}")
'''


def _spawn_fake(name: str = "fake") -> _Server:
    cfg = _ServerConfig(name=name, command=sys.executable, args=["-c", FAKE_SERVER])
    srv = _Server(cfg)
    srv.start(timeout=15.0)
    return srv


class HandshakeTests(unittest.TestCase):
    def test_start_completes_and_discovers_tools(self):
        srv = _spawn_fake()
        try:
            self.assertEqual(srv.state, "ready")
            self.assertEqual([t["name"] for t in srv.tools], ["echo", "fail"])
        finally:
            srv.stop()

    def test_missing_command_errors_loudly(self):
        cfg = _ServerConfig(name="nope",
                            command="/absolutely/nonexistent/binary_xyzzy")
        srv = _Server(cfg)
        with self.assertRaises(MCPError):
            srv.start(timeout=5.0)
        self.assertEqual(srv.state, "error")


class CallToolTests(unittest.TestCase):
    def setUp(self):
        self.srv = _spawn_fake()

    def tearDown(self):
        self.srv.stop()

    def test_text_content_is_returned_verbatim(self):
        out = self.srv.call_tool("echo", {"msg": "hello"})
        self.assertEqual(out, "echo: hello")

    def test_iserror_response_returns_error_prefixed_string(self):
        out = self.srv.call_tool("fail", {})
        self.assertTrue(out.startswith("ERROR:"), out)
        self.assertIn("intentional failure", out)

    def test_unknown_tool_surfaces_as_error_string(self):
        out = self.srv.call_tool("doesnotexist", {})
        self.assertTrue(out.startswith("ERROR:"), out)

    def test_call_before_ready_returns_error(self):
        # Build a fresh server but don't start it.
        cfg = _ServerConfig(name="cold", command=sys.executable,
                            args=["-c", FAKE_SERVER])
        cold = _Server(cfg)
        out = cold.call_tool("echo", {"msg": "x"})
        self.assertTrue(out.startswith("ERROR:"))
        self.assertIn("not ready", out)


class ConfigLoadingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_missing_file_is_silent_no_op(self):
        reg = MCPRegistry()
        reg.load_from_file(self.dir / "absent.json")
        self.assertEqual(reg.servers(), {})

    def test_bad_json_raises_mcp_error(self):
        p = self.dir / "mcp.json"
        p.write_text("{not json")
        reg = MCPRegistry()
        with self.assertRaises(MCPError):
            reg.load_from_file(p)

    def test_valid_config_registers_servers_without_starting(self):
        p = self.dir / "mcp.json"
        p.write_text(json.dumps({
            "servers": {
                "alpha": {"command": "true"},
                "beta":  {"command": "true", "args": ["x"]},
            }
        }))
        reg = MCPRegistry()
        reg.load_from_file(p)
        self.assertEqual(sorted(reg.servers().keys()), ["alpha", "beta"])
        # ensure_started would actually fork these — we don't want that
        # in this test. Verify they are still 'stopped'.
        for srv in reg.servers().values():
            self.assertEqual(srv.state, "stopped")

    def test_invalid_server_entries_are_skipped(self):
        """A missing command or non-dict spec must be dropped silently —
        otherwise one typo in mcp.json would brick every server."""
        p = self.dir / "mcp.json"
        p.write_text(json.dumps({
            "servers": {
                "good":      {"command": "true"},
                "no_cmd":    {"args": ["x"]},
                "junk":      "not a dict",
            }
        }))
        reg = MCPRegistry()
        reg.load_from_file(p)
        self.assertEqual(list(reg.servers().keys()), ["good"])


class RegistryDispatchTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        p = Path(self.tmp.name) / "mcp.json"
        p.write_text(json.dumps({
            "servers": {
                "fake": {"command": sys.executable, "args": ["-c", FAKE_SERVER]},
            }
        }))
        self.reg = MCPRegistry()
        self.reg.load_from_file(p)
        self.reg.ensure_started()

    def tearDown(self):
        self.reg.shutdown()
        self.tmp.cleanup()

    def test_all_tool_schemas_namespaces_names(self):
        names = [s["function"]["name"] for s in self.reg.all_tool_schemas()]
        self.assertIn("mcp__fake__echo", names)
        self.assertIn("mcp__fake__fail", names)

    def test_schemas_carry_input_schema_through_as_parameters(self):
        s = next(s for s in self.reg.all_tool_schemas()
                 if s["function"]["name"] == "mcp__fake__echo")
        params = s["function"]["parameters"]
        self.assertEqual(params.get("type"), "object")
        self.assertIn("msg", params.get("properties", {}))

    def test_dispatch_routes_to_correct_server(self):
        out = self.reg.dispatch("mcp__fake__echo", {"msg": "ok"})
        self.assertEqual(out, "echo: ok")

    def test_dispatch_unknown_server_errors_cleanly(self):
        out = self.reg.dispatch("mcp__unknown_server__foo", {})
        self.assertTrue(out.startswith("ERROR:"))
        self.assertIn("unknown_server", out)

    def test_dispatch_malformed_name_errors_cleanly(self):
        out = self.reg.dispatch("mcp__incomplete", {})
        self.assertTrue(out.startswith("ERROR:"))

    def test_restart_brings_server_back_to_ready(self):
        msg = self.reg.restart("fake")
        self.assertTrue(msg.startswith("OK:"), msg)
        self.assertEqual(self.reg.servers()["fake"].state, "ready")

    def test_restart_unknown_returns_error(self):
        self.assertTrue(self.reg.restart("nope").startswith("ERROR:"))


class IntegrationWithToolsRegistryTests(unittest.TestCase):
    """Verify dispatch() in collama.tools routes mcp__ names through MCP."""

    def test_unconfigured_mcp_name_errors_via_tools_dispatch(self):
        """If no MCP servers are configured, an mcp__ call must NOT fall
        through to the unknown-tool suggester (which would suggest a
        wildly wrong canonical name)."""
        from collama.tools import ToolContext, dispatch
        out = dispatch("mcp__noserver__nope", {},
                       ToolContext(root=Path("/tmp")))
        self.assertTrue(out.startswith("ERROR:"), out)
        # Must NOT include the "did you mean" hint of the suggester —
        # that would mean we routed the wrong way.
        self.assertNotIn("Did you mean", out)


if __name__ == "__main__":
    unittest.main()
