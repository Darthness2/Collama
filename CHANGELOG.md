# Changelog

All notable changes to Collama are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **MCP (Model Context Protocol) support.** Configure servers in
  `~/.config/collama/mcp.json`; their tools are auto-discovered and exposed
  to the model as `mcp__<server>__<tool>`. Lazy-started on first use,
  cleanly torn down on exit. Enable with `/groups enable mcp`.
- `mcp_servers` control tool to inspect configured / running MCP servers.
- `mcp_restart` control tool to bounce a single server without restarting Collama.
- `tests/` directory with 59 unit tests covering permissions, config,
  Ollama host normalization, context compaction, diffs, and tool dispatch.
- GitHub Actions CI matrix (Python 3.9–3.13 on Linux + macOS).
- GitHub Actions release workflow that builds and publishes to PyPI on
  tags matching `v*` using PyPI Trusted Publishing.
- `LICENSE` (MIT) and this `CHANGELOG.md`.

### Fixed
- Streamed output no longer writes over the screen. The bottom status bar's
  scroll-region escape (DECSTBM) homes the cursor to the top-left; on a
  terminal resize the re-reservation ran *before* the cursor save, stranding
  the cursor at home so the next tokens overwrote the banner and prior
  output. The resize escape is now emitted inside the save/restore bracket.
  Streaming and in-place tool-line writes also share the paint lock now, so a
  background status-bar/spinner frame can't slip between a write and its flush
  and fight over the cursor.

### Changed
- `collama/tools.py` (2521 lines) split into a `collama/tools/` package
  with one module per tool group. Public API is unchanged — every existing
  `from collama.tools import ...` still works.
- `__version__` now reads from installed package metadata via
  `importlib.metadata`, eliminating the old two-source-of-truth between
  `__init__.py` and `pyproject.toml`.
- `pyproject.toml` now includes PyPI classifiers, project URLs, keywords,
  and an optional `dev` dependency group (`pip install -e ".[dev]"`).

### Removed
- The four placeholder MCP / LSP stub tools (`mcp`, `mcp_list_resources`,
  `mcp_read_resource`, `lsp`). MCP is now real; LSP integration is deferred
  rather than shipped as a stub.
- `requirements.txt` (it was stale and listed an unrelated dependency —
  the real deps live in `pyproject.toml`).
- `data/`, `assets/`, `debug-*.log` and stray `.DS_Store` files —
  housekeeping from the workspace root.

## [0.1.0] — Initial release

First public release. Terminal coding agent powered by a local Ollama
model, with native tool calling, plan mode, sessions, GitHub integration,
worktree stack, and ~60 tools across read/write/edit/search/run/web/tasks
and multi-agent coordination.

[Unreleased]: https://github.com/YOUR_USERNAME/Collama/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/YOUR_USERNAME/Collama/releases/tag/v0.1.0
