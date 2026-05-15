# Collama

A terminal coding agent like Claude Code / Codex CLI, but powered by [Ollama](https://ollama.com) running locally. Everything stays on your machine.

Collama gives a local LLM tools to read, edit, and create files; run shell commands; search the codebase; query GitHub; and persist tasks across sessions. It streams answers token-by-token, renders markdown live, detects when the model is stuck in a loop, and lets you `/undo` if it edits the wrong thing.

## Requirements

- Python 3.9+
- [Ollama](https://ollama.com) running (`ollama serve`)
- A model with native tool calling — recommended:
  ```
  ollama pull qwen2.5-coder:14b
  ```
  Other working choices: `qwen2.5-coder:7b` (low VRAM), `qwen3-coder:30b-a3b` (≥24 GB VRAM), `llama3.1:8b`. DeepSeek-Coder works via the text-protocol fallback but is less reliable.

## Install

```bash
git clone https://github.com/Darthness2/Collama.git ~/Collama
cd ~/Collama
pip install -e .
```

If `pip install` errors on macOS with *externally-managed-environment*:
```bash
pip install -e . --user        # easiest
# or
brew install pipx && pipx install -e .
# or use a venv
```

## Quickstart

```bash
collama                                  # interactive REPL
collama -p "add --verbose to main.py"    # one-shot
collama --model qwen2.5-coder:14b -C ~/my-project
```

The first launch lists installed Ollama models and asks which to use; the choice is saved to `~/.config/collama/config.json`. Re-run with `--reset-config` to start fresh.

When you type `/` in the prompt a popup of available commands appears. If it doesn't, run `/diag` — if the `input:` line says `readline` instead of `prompt_toolkit`, install it (`pip install prompt_toolkit`).

## Configuration

`~/.config/collama/config.json`:

```json
{
  "model": "qwen2.5-coder:14b",
  "host": "http://localhost:11434",
  "temperature": 0.2,
  "yolo": false,
  "tool_groups": ["core", "search", "tasks", "background",
                  "planning", "notebook", "worktree",
                  "interaction", "system"],
  "ollama": {
    "num_ctx": 8192,           // KV cache cap — protects VRAM
    "keep_alive": "30m",       // how long to keep the model in VRAM
    "read_timeout": 600,       // per-chunk gap, not whole response
    "connect_timeout": 15,
    "stream": true             // turn off on networks that break chunked transfers
  },
  "github": { "token": null }
}
```

Almost everything is also toggleable in the REPL: `/model`, `/host`, `/stream on|off`, `/insecure on|off`, `/groups enable github`, `/set temperature 0.5`, etc.

## Slash commands

### Conversation
| | |
|---|---|
| `/new [title]` | start a fresh conversation |
| `/resume [id]` | list saved sessions or resume one |
| `/sessions` | list all saved sessions |
| `/save [title]` | force-save the current session |
| `/delete <id>` | remove a saved session |
| `/clear` | wipe history but keep the saved session |
| `/retry` | re-run your last message |
| `/exit`, `/quit` | leave |

### Editing & inspection
| | |
|---|---|
| `/diff [N]` | show the last N file edits this session (defaults to all) |
| `/undo` | revert the most recent edit and pop it from history |
| `/cd [path]` | show or change the workspace dir |
| `/wt` | show the worktree stack |
| `/diag` | print model, workspace, tools mode, num_ctx, etc. |

### Tools & permissions
| | |
|---|---|
| `/tools` | list every tool the model can call |
| `/groups [enable\|disable G]` | manage which tool groups are sent (see below) |
| `/tools-on` / `/tools-off` | force native vs text-protocol mode for this model |
| `/yolo` | toggle auto-approve for all tool calls |
| `/stream on\|off` | toggle token streaming (turn off on flaky networks) |
| `/insecure on\|off` | disable TLS verification (school/corp MITM proxies) |

### Planning & work
| | |
|---|---|
| `/plan on\|off` | plan mode — read-only, no mutating tools |
| `/todo`, `/todo add <t>`, `/todo done <n>`, `/todo clear` | session todo list |
| `/brief [name]` | show stored markdown briefs |
| `/tasks` | list persistent tasks (survive restarts) |
| `/jobs` | list background jobs (`bash_async`, `agent_call_async`) |
| `/teams`, `/tick [team] [claim]` | multi-agent teams (advanced) |

### GitHub & system
| | |
|---|---|
| `/login github <token>`, `/logout github`, `/whoami` | auth |
| `/model [name]`, `/models` | switch model / list installed |
| `/host [url]` | switch Ollama host |
| `/config`, `/set <key> <value>` | view / edit saved config |
| `/help` | show this list inside the REPL |

## Tool groups

Sending all ~60 tools every request is a heavy prompt-eval cost on a local model. Tools are grouped; only the on-by-default ones are sent:

**On by default:** `core` (read/write/edit/list/grep/bash/set_workspace), `search` (glob/web_fetch/web_search), `tasks` (task graph + todo + brief), `background`, `planning`, `notebook`, `worktree`, `interaction`, `system`.

**Off by default** (enable with `/groups enable <name>`): `github`, `teams`, `subagent`, `stubs` (mcp/lsp/tungsten placeholders).

```
/groups                        # show what's on
/groups enable github          # turn a group on (persisted)
/groups disable background
```

## `@path` mentions

Reference files directly in your prompt and Collama inlines their contents before sending — saves the model a round of exploration tool calls:

```
why does /ask freeze? look at @agent/react.py:80-130
explain the diff between @cli/dashboard.py:431-460 and the spec
```

Supports `@path`, `@path:N`, and `@path:N-M`.

## Sessions

Every conversation is auto-saved to `~/.config/collama/sessions/<id>.json` after each turn. `/resume` replays the prior conversation so you have context. Transcripts also stream to `~/.config/collama/transcripts/<id>.jsonl` (append-only).

## Debugging features

When the model edits a file, Collama prints a short colorized diff (`+` green, `−` red, capped at ~12 lines). `run_bash` results are prefixed **PASS** or **FAIL (exit code N)**, and failed commands get a one-line hint pointing at the most likely error site, parsed from the traceback:

```
↳ ValueError: list index out of range  ·  likely at game.py:42
```

`edit_file` is whitespace-tolerant — if the exact `old_string` doesn't match, it tries again with line-ending normalization and per-line trailing-whitespace fuzzy matching, and on a real miss it shows the closest region of the file.

Per-turn **loop detection**: if the same `(tool, result)` comes back 3 times the model gets a hard steer; at 6 the turn is aborted with a `/retry` suggestion. Re-reading the same file is short-circuited from a `read_file` cache that tells the model to scroll back.

## GitHub integration

Generate a fine-grained PAT at <https://github.com/settings/tokens>, then:

```
/login github ghp_xxxxxxxxxxxxxxxxxxxxx
/groups enable github
/whoami
```

Token is stored chmod-600 in the config file. `GITHUB_TOKEN` / `GH_TOKEN` env vars also work. Tools available once enabled: `gh_whoami`, `gh_list_repos`, `gh_get_repo`, `gh_get_file`, `gh_list_issues`, `gh_create_issue`, `gh_list_pulls`, `gh_get_pull`, `gh_search_code`, plus a raw `github_api` escape hatch. Mutating calls always ask for approval.

## Hardware tips

The single biggest perf knob is **whether your model fits in VRAM**. At q4_K_M:

| Model | Approx weight | Fits |
|---|---|---|
| qwen2.5-coder:7b | ~5 GB | 8 GB+ |
| qwen2.5-coder:14b | ~9 GB | 12 GB+ |
| qwen3-coder:30b-a3b (MoE) | ~18 GB | 24 GB+ |
| qwen2.5-coder:32b | ~19 GB | 24 GB+ |

If the model doesn't fit, Ollama spills layers to system RAM and prompt-eval crawls. Keep `num_ctx` modest (8192 is a good default) so the KV cache doesn't push layers off the GPU. If everything still feels slow, drop a model size — a fully-resident 14B is sharper *and* faster than a partially-offloaded 30B.

Linux/macOS bonus: install [`ripgrep`](https://github.com/BurntSushi/ripgrep). Collama's `grep` tool shells out to it when available — way faster on large repos.

## File locations

```
~/.config/collama/
├── config.json          # persistent config (chmod 600)
├── history              # REPL input history
├── sessions/            # auto-saved conversations
├── transcripts/         # append-only JSONL of every turn
├── tasks/               # persistent task graph
├── teams/               # team mailboxes
└── skills/              # *.md skills appendable to the system prompt
```

## License

MIT
