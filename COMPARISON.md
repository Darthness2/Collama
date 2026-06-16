# Collama vs. OpenCode: Architectural Comparison

This document compares Collama with [OpenCode](https://github.com/anthropics/opencode), a multi-provider terminal coding agent platform by SST/Anomaly.

## Executive Summary

| Dimension | Collama | OpenCode |
|-----------|---------|----------|
| **Architecture** | Local-only, single-machine | Client/server with LSP support |
| **Providers** | Ollama only | Vercel AI SDK (30+ models: Claude, GPT, Gemini, local via MCP) |
| **Language** | Python | TypeScript (v1.0+; was Go) |
| **IDE Support** | None (terminal only) | VS Code extension, JetBrains plugin, LSP |
| **Maturity** | Early-stage, single author | Backed by SST/Anomaly, ~175k GitHub stars |
| **Privacy Model** | Everything stays local by default | Depends on provider selection |
| **Deploy** | `pip install` → `ollama serve` → `collama` | Web app + CLI + IDE extensions |

---

## Architecture & Deployment

### Collama: Local-First Single Machine
- Runs entirely on your machine; no network requests to proprietary backends
- Ollama process handles model inference locally
- Python CLI with interactive REPL or one-shot mode
- Sessions and tasks persisted to `~/.config/collama/`
- Single author (early-stage project); community contributions welcome

### OpenCode: Multi-Tier Networked
- Client/server architecture: CLI/web UI talks to backend
- LSP server for deeper IDE integration
- Vercel AI SDK abstracts 30+ models (Claude, OpenAI, Google, local via MCP)
- Enterprise deployment: self-host or use SST/Anomaly SaaS
- Well-resourced team; active development and community

---

## Model & Provider Support

### Collama: Ollama Ecosystem Only
**Supported models:**
- `qwen2.5-coder:14b` (recommended, ~9 GB)
- `qwen2.5-coder:7b` (low VRAM, ~5 GB)
- `qwen3-coder:30b-a3b` (MoE, 24+ GB VRAM)
- `llama3.1:8b`
- DeepSeek-Coder (via text-protocol fallback, less reliable)

**Constraints:**
- Must run Ollama locally (`ollama serve`)
- VRAM-bound: 14B model needs 12+ GB
- Single model per session (though configurable)

### OpenCode: Multi-Provider via Vercel AI SDK
**Supported providers & models:**
- **Claude** (Anthropic): latest Opus, Sonnet, Haiku
- **OpenAI**: GPT-4, GPT-3.5
- **Google**: Gemini Pro
- **Local via MCP**: OpenCode can wrap local models through custom MCP servers
- **Models.dev**: SST's unified interface for third-party models

**Advantages:**
- Mix and match providers in same workflow
- Fallback chains (e.g., try Claude → fall back to GPT)
- Cloud inference removes local VRAM pressure
- API usage tracked per provider

---

## Features & Tool Capabilities

### Collama
- **File operations**: read, write, edit with fuzzy whitespace matching
- **Code search**: glob, grep (uses ripgrep if available)
- **Execution**: bash with loop detection and parse-based error hints
- **Jupyter support**: notebook_edit for cell-level edits
- **Syntax checking**: parser-only checks for 12+ languages (no execution)
- **Session persistence**: auto-save conversations, resume with `/resume`
- **Task system**: persistent cross-session task graph
- **MCP servers**: Model Context Protocol support (requires manual setup in `mcp.json`)
- **Loop detection**: same `(tool, result)` 3× triggers steer, 6× aborts turn

### OpenCode
- All of Collama's core tools plus:
- **Browser automation**: playwright-based page navigation & evaluation
- **Native IDE integrations**: VS Code extension, JetBrains plugins
- **LSP server**: language-aware diagnostics and completions
- **GitHub API**: PR creation, issue management, CI status checks
- **CI/CD integration**: GitHub Actions hooks, auto-review on PRs
- **Plugin ecosystem**: extensible via published plugins
- **Streaming**: token-by-token rendering with live markdown
- **Plan mode**: read-only exploration before write approval

Both projects implement similar streaming, markdown rendering, and model loop-detection heuristics.

---

## Security & Privacy Model

### Collama: Local-First Privacy by Default
- ✅ All model inference stays on your machine
- ✅ Sessions and tasks stored locally (chmod-600)
- ✅ No telemetry or cloud calls
- ✅ Optional GitHub token for local git operations (stored chmod-600)
- ⚠️ SSRF guards on web_fetch and web_search (validates IPs + DNS resolution)
- ⚠️ Path containment enforced with realpath + symlink checks

### OpenCode: Provider-Dependent Privacy
- Provider choice determines data handling (Claude, OpenAI, self-hosted, etc.)
- Can self-host entire stack (backend + LSP)
- SST/Anomaly operates the default SaaS (privacy policy covers it)
- Also implements SSRF guards and path containment

**Privacy matrix:**
| Scenario | Collama | OpenCode |
|----------|---------|----------|
| Using local Ollama | ✅ Local only | ❌ Needs backend |
| Using cloud APIs (Claude, GPT) | ❌ N/A | depends on provider |
| Self-hosted OpenCode + local MCP | ✅ Local | ✅ Local |

---

## Extensibility & Integration

### Collama: MCP-Based
- **Tool groups**: enable/disable sets of tools (`/groups enable github`)
- **MCP servers**: plug in via `~/.config/collama/mcp.json`
  ```json
  { "servers": { "custom": { "command": "uvx", "args": [...] } } }
  ```
- **Skills**: append markdown briefs to system prompt
- **No IDE plugins** or LSP — terminal only
- **No SDK** for embedding in other tools

### OpenCode: SDK + Plugins + LSP
- **Vercel AI SDK**: build custom agents programmatically
- **Plugin system**: npm packages that extend the agent
- **LSP server**: deep IDE integration (completions, diagnostics, hover)
- **Web framework**: deployable UI on your own domain
- **Extensible providers**: MCP support + custom provider adapters

---

## Maturity & Community

### Collama
- **GitHub**: ~500–1k stars (early-stage)
- **Team**: Single maintainer (open to contributions)
- **Release cadence**: Active but occasional
- **Issue tracker**: Community-driven
- **Documentation**: Comprehensive README with examples
- **Stability**: Core features work; API not stabilized

### OpenCode
- **GitHub**: ~175k stars (flagship project)
- **Team**: SST/Anomaly (well-resourced)
- **Release cadence**: Regular updates and bug fixes
- **Ecosystem**: Plugins, extensions, SaaS offering
- **Documentation**: Extensive docs, tutorials, and blog posts
- **Stability**: Production-ready; backward-compatible releases

---

## When to Use Each

### Choose **Collama** if you:
- Value **privacy above all** — want inference & data fully local
- Have a **GPU with 12+ GB VRAM** and want to run a solid 14B coder
- Prefer **minimal setup** — just `pip install` + `ollama serve`
- Work in the **terminal exclusively** (no IDE integration needed)
- Want a **lightweight, dependency-light** agent
- Are comfortable with an **early-stage, single-author project**

### Choose **OpenCode** if you:
- Want **model flexibility** — switch between Claude, GPT, Gemini
- Need **IDE integration** — VS Code, JetBrains, or LSP
- Require **GitHub CI/CD integration** — PR reviews, issue automation
- Prefer a **well-maintained, battle-tested platform** (175k+ stars)
- Want **plugin extensibility** — published third-party tools
- Willing to trade **local privacy** for **cloud inference convenience**
- Need a **deployable, scalable solution** for teams

---

## Performance Comparison

### Token Latency
- **Collama (Ollama 14B)**: ~10–50 tokens/sec on 12GB VRAM (GPU-dependent)
- **OpenCode (Claude)**: ~50+ tokens/sec (cloud-backed)
- **OpenCode (local MCP)**: Same as Collama (depends on model)

### Inference Cost
| Scenario | Collama | OpenCode |
|----------|---------|----------|
| Ollama 14B local | VRAM only | N/A |
| Claude API (cloud) | N/A | ~$0.003 per 1K input tokens |
| GPT-4 (cloud) | N/A | ~$0.03 per 1K input tokens |

### Startup Time
- **Collama**: ~2–5s (model already in VRAM if kept alive)
- **OpenCode (cloud)**: ~1s (API latency)
- **OpenCode (local MCP)**: Same as Collama

---

## Verdict

**Collama** is an excellent **privacy-first, local-only coding agent** for developers who:
- Control their own infrastructure
- Prioritize data sovereignty
- Have GPU resources
- Work exclusively in terminals

**OpenCode** is the **production-grade, multi-provider platform** for:
- Teams that need IDE integration and CI/CD automation
- Developers who want flexibility to choose models and providers
- Enterprises requiring support, stability, and extensibility

Both projects implement **streaming, markdown rendering, tool-use orchestration**, and **loop detection** with comparable quality. The choice ultimately depends on whether you prioritize **privacy/control** (Collama) or **features/ecosystem** (OpenCode).

---

*Comparison based on public documentation and GitHub repositories as of June 2026. Both projects are actively maintained.*
