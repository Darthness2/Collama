"""can_use_tool() — the gate that decides whether a tool call is allowed.

Security-relevant: a regression here means the agent can mutate files
without prompting, or that the plan-mode read-only invariant breaks.
"""
from __future__ import annotations

import unittest
from pathlib import Path

from collama.permissions import can_use_tool
from collama.state import AppState


def _state(**overrides):
    s = AppState(workspace=Path("/tmp"), home=Path("/tmp"))
    for k, v in overrides.items():
        setattr(s, k, v)
    return s


class CacheTests(unittest.TestCase):
    def test_cached_always_short_circuits_with_no_resolver_call(self):
        s = _state(permissions={"write_file": "always"})
        calls = []
        ok, reason = can_use_tool("write_file", {}, s,
                                  resolver=lambda *a: calls.append(a) or "no")
        self.assertTrue(ok)
        self.assertIn("cache", reason)
        self.assertEqual(calls, [], "resolver must not be called when cached")

    def test_cached_never_blocks_with_no_resolver_call(self):
        s = _state(permissions={"run_bash": "never"})
        calls = []
        ok, reason = can_use_tool("run_bash", {}, s,
                                  resolver=lambda *a: calls.append(a) or "yes")
        self.assertFalse(ok)
        self.assertIn("cache", reason)
        self.assertEqual(calls, [])


class ReadOnlyTests(unittest.TestCase):
    def test_read_only_tools_pass_without_resolver(self):
        s = _state()
        called = []
        for name in ("read_file", "list_dir", "grep", "glob",
                     "task_get", "task_list", "config_get"):
            ok, reason = can_use_tool(name, {}, s,
                                      resolver=lambda *a: called.append(a) or "no")
            self.assertTrue(ok, f"{name} should be allowed read-only")
            self.assertEqual(reason, "read-only tool")
        self.assertEqual(called, [], "resolver must not run for read-only tools")


class YoloTests(unittest.TestCase):
    def test_yolo_allows_mutating_tool_without_resolver(self):
        s = _state(yolo=True)
        called = []
        ok, _ = can_use_tool("write_file", {"path": "x"}, s,
                             resolver=lambda *a: called.append(a) or "no")
        self.assertTrue(ok)
        self.assertEqual(called, [])


class PlanModeTests(unittest.TestCase):
    def test_plan_mode_blocks_write_file(self):
        s = _state(plan_mode=True)
        ok, reason = can_use_tool("write_file", {}, s,
                                  resolver=lambda *a: "yes")
        self.assertFalse(ok)
        self.assertIn("plan mode", reason)

    def test_plan_mode_blocks_run_bash(self):
        s = _state(plan_mode=True)
        ok, _ = can_use_tool("run_bash", {"command": "ls"}, s,
                             resolver=lambda *a: "yes")
        self.assertFalse(ok)

    def test_plan_mode_still_allows_read_file(self):
        s = _state(plan_mode=True)
        ok, _ = can_use_tool("read_file", {}, s,
                             resolver=lambda *a: "no")
        self.assertTrue(ok)

    def test_plan_mode_does_not_short_circuit_exit_plan_mode(self):
        """`exit_plan_mode` is on the plan-mode bypass list — it can REACH
        the resolver in plan mode (other mutating tools can't). It still
        needs resolver approval, it just isn't auto-blocked by the gate."""
        s = _state(plan_mode=True)
        ok_yes, reason_yes = can_use_tool("exit_plan_mode", {}, s,
                                          resolver=lambda *a: "yes")
        self.assertTrue(ok_yes, "should reach resolver and be approved")
        self.assertNotIn("plan mode", reason_yes)

        ok_no, _ = can_use_tool("exit_plan_mode", {}, _state(plan_mode=True),
                                resolver=lambda *a: "no")
        self.assertFalse(ok_no, "denied resolver still denies")

    def test_plan_mode_yolo_does_not_override(self):
        """Plan-mode protection must beat yolo — otherwise yolo silently
        breaks the read-only contract."""
        s = _state(plan_mode=True, yolo=True)
        ok, reason = can_use_tool("write_file", {}, s,
                                  resolver=lambda *a: "yes")
        self.assertFalse(ok)
        self.assertIn("plan mode", reason)


class ResolverTests(unittest.TestCase):
    def test_yes_allows_one_call_without_caching(self):
        s = _state()
        ok, reason = can_use_tool("write_file", {}, s,
                                  resolver=lambda *a: "yes")
        self.assertTrue(ok)
        self.assertNotIn("write_file", s.permissions,
                         "yes must not promote to 'always'")
        self.assertEqual(reason, "user approved")

    def test_always_caches(self):
        s = _state()
        ok, _ = can_use_tool("write_file", {}, s,
                             resolver=lambda *a: "always")
        self.assertTrue(ok)
        self.assertEqual(s.permissions.get("write_file"), "always")

    def test_never_caches(self):
        s = _state()
        ok, _ = can_use_tool("run_bash", {}, s,
                             resolver=lambda *a: "never")
        self.assertFalse(ok)
        self.assertEqual(s.permissions.get("run_bash"), "never")

    def test_no_and_default_both_deny_without_caching(self):
        for answer in ("no", ""):
            s = _state()
            ok, _ = can_use_tool("write_file", {}, s,
                                 resolver=lambda *a, _ans=answer: _ans)
            self.assertFalse(ok, f"resolver returning {answer!r} should deny")
            self.assertNotIn("write_file", s.permissions)


class DefaultResolverTests(unittest.TestCase):
    def test_default_resolver_denies_mutating_tools_in_headless_mode(self):
        s = _state()
        ok, _ = can_use_tool("write_file", {}, s)
        self.assertFalse(ok, "SDK / headless callers must default to deny")


if __name__ == "__main__":
    unittest.main()
