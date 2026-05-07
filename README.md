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

### Slash commands inside the REPL

- `/help` — list commands
- `/model <name>` — swap model mid-session
- `/clear` — reset conversation history
- `/tools` — list available tools
- `/exit` — quit

## Tools the model can call

| Tool | Purpose |
| --- | --- |
| `read_file` | Read a text file (with optional line range) |
| `write_file` | Create or overwrite a file |
| `edit_file` | Exact string replacement in a file |
| `list_dir` | List files in a directory |
| `grep` | Regex search across the workspace |
| `run_bash` | Execute a shell command (asks for approval) |

By default, file-mutating and shell tools require interactive approval. Pass `--yolo` to auto-approve everything (use with care).

## License

MIT
