# Collama

A terminal coding agent like Claude Code / Codex CLI, but powered by [Ollama](https://ollama.com) running locally.

Collama gives a local LLM the ability to read, edit, and create files, run shell commands, and search your codebase — all from an interactive terminal session.

## Requirements

- Python 3.9+
- [Ollama](https://ollama.com) installed and running (`ollama serve`)
- A model that supports tool calling, e.g. `qwen2.5-coder`, `llama3.1`, `llama3.2`, `mistral-nemo`:
  ```
  ollama pull qwen2.5-coder
  ```

## Install

```bash
pip install -e .
```

Or run without installing:

```bash
python -m collama
```

## Usage

Interactive REPL in the current directory:

```bash
collama
```

One-shot prompt:

```bash
collama -p "add a CLI flag --verbose to main.py and wire it through"
```

Pick a model / host:

```bash
collama --model llama3.1 --host http://localhost:11434
```

## First run

The first time you start Collama it asks which model to use, lists what's
already installed via Ollama, and saves your choice to
`~/.config/collama/config.json` (chmod 600). On subsequent runs it just uses
the saved model unless you override it.

You can wipe the saved config and start fresh with:

```bash
collama --reset-config
```

As soon as you type `/` in the prompt, a popup of available slash commands
appears with short descriptions; arrow keys + Enter pick one. (Requires
`prompt_toolkit`, which is installed as a dependency. Without it Collama
falls back to TAB-completion via `readline`.)

## Slash commands inside the REPL

```
/help                   show help
/tools                  list tools the model can call
/model [name]           show or switch model (saved)
/models                 list locally installed Ollama models
/host [url]             show or change the Ollama host (saved)
/config                 show current config (token redacted)
/set <key> <value>      set a config value (e.g. /set temperature 0.5)
/login github <token>   save a GitHub Personal Access Token
/logout github          remove the saved GitHub token
/whoami                 show authenticated GitHub user
/clear                  reset conversation history (does not delete the saved session)
/new [title]            start a new conversation
/resume [id|number]     list saved conversations or resume one
/sessions               list saved conversations
/save [title]           force-save / set the title of the current conversation
/delete <id|number>     delete a saved conversation
/yolo                   toggle auto-approve for tool calls
/exit                   quit
```

## GitHub integration

Generate a fine-grained Personal Access Token at
<https://github.com/settings/tokens> (give it the scopes you want — `repo`
for private repos, `read:user`, etc.), then inside the REPL:

```
/login github ghp_xxxxxxxxxxxxxxxxxxxxx
/whoami
```

Or set `GITHUB_TOKEN` / `GH_TOKEN` in your environment.

Once logged in, the model can call: `gh_whoami`, `gh_list_repos`,
`gh_get_repo`, `gh_get_file`, `gh_list_issues`, `gh_create_issue`,
`gh_list_pulls`, `gh_get_pull`, `gh_search_code`, and `github_api`
(raw escape hatch). Mutating calls always ask for approval unless `--yolo`.

## Local tools the model can call

| Tool | Purpose |
| --- | --- |
| `read_file` | Read a text file (with optional line range) |
| `write_file` | Create or overwrite a file |
| `edit_file` | Exact string replacement in a file |
| `list_dir` | List files in a directory |
| `grep` | Regex search across the workspace |
| `run_bash` | Execute a shell command (asks for approval) |

File-mutating and shell tools require interactive approval. Pass `--yolo` (or
`/yolo`) to auto-approve everything — use with care.

Whenever the model writes or edits a file, Collama prints a colorized unified
diff (green `+`, red `-`) so you can see exactly what changed.

## Filesystem access

Collama can read and write any file your user account can. Relative paths
resolve against the workspace root (`-C`/cwd); absolute paths and `~`-paths
work too — e.g. `~/Downloads/report.md`, `/etc/hosts`.

## Conversations

Every session is auto-saved to `~/.config/collama/sessions/<id>.json` after
each turn. Manage them with:

```
/sessions              list everything
/new [title]           start fresh
/resume                list and resume by id or number
/save [title]          force-save with a title
/delete <id|number>    remove a saved session
```

## License

MIT
