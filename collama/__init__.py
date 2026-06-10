"""Collama — a terminal coding agent powered by Ollama."""
from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version as _version

try:
    __version__ = _version("collama")
except PackageNotFoundError:
    # Source-tree import without `pip install`. Useful for hacking on the
    # repo directly; published builds always have package metadata.
    __version__ = "0+unknown"
