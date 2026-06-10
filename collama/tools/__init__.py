"""Public API for the tools package.

Re-exports everything the rest of the codebase imported off the old
flat `collama.tools` module, so `from collama.tools import ...` keeps
working unchanged.
"""
from .base import ToolContext, _truncate
from .registry import (
    DEFAULT_GROUPS,
    TOOL_ALIASES,
    TOOL_GROUPS,
    TOOL_SCHEMAS,
    TOOLS,
    _all_tools,
    _compact_schema,
    all_tool_schemas,
    dispatch,
)

__all__ = [
    "ToolContext",
    "_truncate",
    "TOOLS",
    "TOOL_SCHEMAS",
    "TOOL_GROUPS",
    "DEFAULT_GROUPS",
    "TOOL_ALIASES",
    "_all_tools",
    "_compact_schema",
    "all_tool_schemas",
    "dispatch",
]
