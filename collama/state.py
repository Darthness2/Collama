"""Centralized runtime state — AppState store with subscriber notifications.

Mirrors the React `useAppState` pattern in plain Python: a single mutable
record, observers register a callback and get notified on every update.
Owners of state live here so QueryEngine, tools, and the REPL all read the
same source of truth instead of passing fields around.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable


Listener = Callable[["AppState", str, Any], None]


@dataclass
class AppState:
    # core paths / identity
    workspace: Path
    home: Path

    # service auth
    github_token: str | None = None

    # behavior switches
    yolo: bool = False                  # auto-approve all tool calls
    fast_mode: bool = False             # lower temp / shorter answers
    insecure_ssl: bool = False          # turn off TLS verify (school MITM)
    tools_enabled: bool = True          # native ollama tool calls vs text-protocol

    # Effort dial — how much effort/thoroughness the model applies. One of
    # "low" | "medium" | "high". Injected into the system prompt; higher means
    # more exploration before acting, more verification after, and more rigor.
    effort: str = "medium"

    # per-tool permission cache: name -> "always"|"once"|"never"|None
    permissions: dict[str, str] = field(default_factory=dict)

    # files we've read/edited this session (relative paths, in order)
    file_history: list[str] = field(default_factory=list)

    # worktree stack — entered worktree dirs we'll pop back to.
    worktree_stack: list[str] = field(default_factory=list)

    # Plan mode: when on, the agent must produce a plan and
    # NOT call mutating tools — read-only inspection only.
    plan_mode: bool = False

    # Lightweight per-session todo list (TodoWrite).
    todos: list[dict] = field(default_factory=list)

    # In-memory briefs (BriefTool): name -> markdown.
    briefs: dict[str, str] = field(default_factory=dict)

    # Enabled tool groups — controls which tool schemas are sent to the model.
    # None means "use tools.DEFAULT_GROUPS".
    tool_groups: set[str] | None = None

    # Edit history for /undo and /diff: list of
    #   {"ts", "path", "before", "after", "op"}
    # capped at MAX_EDIT_HISTORY entries (oldest dropped).
    edit_history: list[dict] = field(default_factory=list)

    # Per-turn map of {absolute_path: failed_edit_count}. Reset by the engine
    # each turn. After 2 failures on the same file the error escalates and
    # tells the model to switch to write_file.
    edit_fails: dict = field(default_factory=dict)

    # Per-turn set of absolute paths the model has already read this turn —
    # a path-only fallback for the read cache. Catches the case where the
    # model varies line-range args to bypass the dict cache.
    files_read: set[str] = field(default_factory=set)

    # per-model facts learned at runtime (e.g. tools_supported)
    models: dict[str, dict] = field(default_factory=dict)

    _listeners: list[Listener] = field(default_factory=list, repr=False)

    # ------------------------------------------------------------------
    def subscribe(self, fn: Listener) -> Callable[[], None]:
        self._listeners.append(fn)
        return lambda: self._listeners.remove(fn) if fn in self._listeners else None

    def update(self, **changes: Any) -> None:
        for key, value in changes.items():
            if not hasattr(self, key):
                raise AttributeError(f"AppState has no field '{key}'")
            old = getattr(self, key)
            if old == value:
                continue
            setattr(self, key, value)
            for fn in list(self._listeners):
                try:
                    fn(self, key, value)
                except Exception:
                    pass

    def push_file(self, path: str) -> None:
        if path in self.file_history:
            self.file_history.remove(path)
        self.file_history.append(path)
        del self.file_history[:-50]  # cap

    # convenience: turn the state into the dict ToolContext-shaped objects expect
    def to_tool_ctx_kwargs(self) -> dict[str, Any]:
        return {
            "root": self.workspace,
            "yolo": self.yolo,
            "github_token": self.github_token,
            "insecure_ssl": self.insecure_ssl,
        }
