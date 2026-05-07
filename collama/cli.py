"""Command-line entry point."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from . import __version__, config, sessions, ui
from .agent import Agent
from .ollama_client import OllamaClient, OllamaError
from .prompt import Prompt


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="collama",
        description="A Claude Code / Codex-style coding agent powered by Ollama.",
    )
    p.add_argument("-p", "--prompt", help="One-shot prompt; print response and exit.")
    p.add_argument("-m", "--model", help="Ollama model (overrides saved config).")
    p.add_argument("--host", help="Ollama host (overrides saved config).")
    p.add_argument("-C", "--cwd", default=".", help="Workspace root (default: cwd).")
    p.add_argument("--yolo", action="store_true", help="Auto-approve all tool calls.")
    p.add_argument("-t", "--temperature", type=float, help="Sampling temperature.")
    p.add_argument("--reset-config", action="store_true", help="Wipe saved config and start fresh.")
    p.add_argument("-V", "--version", action="version", version=f"collama {__version__}")
    return p.parse_args(argv)


HELP_TEXT = """\
Slash commands:
  /help                   show this help
  /tools                  list tools the model can call (and current mode)
  /tools-on               force native tool calls for this model (saves)
  /tools-off              force text-protocol tool fallback for this model (saves)
  /cd [path]              show or change the workspace directory
  /diag                   print model / workspace / home / tools / github status
  /model [name]           show or switch model
  /models                 list locally installed Ollama models
  /host [url]             show or change the Ollama host
  /config                 show current config (token redacted)
  /set <key> <value>      set a config value (e.g. temperature 0.5)
  /login github <token>   save a GitHub Personal Access Token
  /logout github          remove the saved GitHub token
  /whoami                 show authenticated GitHub user
  /clear                  reset conversation history (does not delete saved session)
  /new [title]            start a new conversation
  /resume [id|number]     list saved conversations or resume one
  /sessions               list saved conversations
  /save [title]           force-save the current conversation (sets title)
  /delete <id|number>     delete a saved conversation
  /yolo                   toggle auto-approve for tool calls
  /exit, /quit            leave
"""


def _pick_model_interactive(client: OllamaClient) -> str | None:
    """First-run model picker. Returns chosen model name or None on cancel."""
    try:
        models = client.list_models()
    except OllamaError as e:
        ui.error(str(e))
        ui.warn("Is `ollama serve` running?")
        return None

    print()
    ui.info("Welcome to Collama. Pick the Ollama model to use.")
    if models:
        print()
        for i, m in enumerate(models, 1):
            print(f"  {i:>2}. {m}")
        print()
        prompt = "Choose a number, or type a model name to pull/use: "
    else:
        ui.warn("No models installed locally.")
        ui.warn("Suggested: qwen2.5-coder, llama3.1, llama3.2, mistral-nemo")
        prompt = "Type a model name: "

    try:
        ans = input(prompt).strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return None
    if not ans:
        return None
    if ans.isdigit() and models:
        idx = int(ans) - 1
        if 0 <= idx < len(models):
            return models[idx]
    return ans


def _redact(cfg: dict) -> dict:
    out = {**cfg, "github": dict(cfg.get("github", {}))}
    tok = out["github"].get("token")
    if tok:
        out["github"]["token"] = tok[:4] + "…" + tok[-4:] if len(tok) > 8 else "••••"
    return out


def _apply_to_agent(agent: Agent, cfg: dict) -> None:
    agent.ctx.github_token = config.get_value(cfg, "github.token")
    agent.ctx.yolo = bool(cfg.get("yolo", agent.ctx.yolo))


def _autosave(session: dict, agent: Agent) -> None:
    session["model"] = agent.model
    session["messages"] = [m for m in agent.messages if m.get("role") != "system"]
    sessions.save(session)


def _print_sessions(active_id: str | None = None) -> list[dict]:
    listed = sessions.list_all()
    if not listed:
        ui.info("(no saved conversations)")
        return []
    print()
    print(ui.color(f"  {'#':<3} {'id':<14} {'updated':<10} {'turns':<6} {'model':<20} title", ui.GRAY))
    for i, s in enumerate(listed, 1):
        marker = ui.color(" *", ui.GREEN) if s["id"] == active_id else "  "
        print(f"{marker}{i:<3} {s['id']:<14} {sessions.fmt_time(s['updated_at']):<10} "
              f"{s['turns']:<6} {s['model'][:19]:<20} {s['title'][:60]}")
    return listed


def _resolve_session_arg(arg: str, listed: list[dict]) -> dict | None:
    if not arg:
        return None
    if arg.isdigit():
        idx = int(arg) - 1
        if 0 <= idx < len(listed):
            return listed[idx]
        return None
    for s in listed:
        if s["id"] == arg or s["id"].startswith(arg):
            return s
    return None


def repl(agent: Agent, cfg: dict) -> int:
    ui.banner(agent.model, str(agent.ctx.root), tools_enabled=agent.tools_enabled)

    # Active session (auto-created, auto-saved after each turn)
    session = sessions.make(agent.model)
    agent.on_turn_complete = lambda a: _autosave(session, a)
    ui.info(f"new session: {session['id']}")

    prompt = Prompt()
    while True:
        ui.prepare_for_input()
        try:
            line = prompt.ask(ui.color("\n❯ ", ui.TEAL_BRIGHT)).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not line:
            continue

        if line.startswith("/"):
            parts = line.split(maxsplit=2)
            cmd = parts[0][1:].lower()
            arg1 = parts[1] if len(parts) > 1 else ""
            arg2 = parts[2] if len(parts) > 2 else ""

            if cmd in ("exit", "quit"):
                return 0
            if cmd == "help":
                print(HELP_TEXT)
                continue
            if cmd == "tools":
                from .tools import _all_tools
                mode = "native" if agent.tools_enabled else "text-protocol fallback"
                ui.info(f"mode: {mode}")
                for n in _all_tools():
                    print(f"  - {n}")
                continue
            if cmd == "tools-on":
                agent.tools_enabled = True
                agent._refresh_system_prompt()
                config.set_value(cfg, f"models.{agent.model}.tools_supported", True)
                config.save(cfg)
                ui.info(f"native tools force-enabled for '{agent.model}' (saved).")
                continue
            if cmd == "tools-off":
                agent.tools_enabled = False
                agent._refresh_system_prompt()
                config.set_value(cfg, f"models.{agent.model}.tools_supported", False)
                config.save(cfg)
                ui.info(f"using text-protocol fallback for '{agent.model}' (saved).")
                continue
            if cmd == "cd":
                if not arg1:
                    ui.info(f"workspace: {agent.ctx.root}")
                    continue
                target = Path(os.path.expanduser(os.path.expandvars(arg1))).resolve()
                if not target.is_dir():
                    ui.error(f"not a directory: {target}")
                    continue
                agent.ctx.root = target
                agent._refresh_system_prompt()
                ui.info(f"workspace → {target}")
                continue
            if cmd == "diag":
                ui.info(f"model:    {agent.model}")
                ui.info(f"workspace: {agent.ctx.root}")
                ui.info(f"home:     {Path.home()}")
                ui.info(f"tools:    {'native' if agent.tools_enabled else 'text-protocol fallback'}")
                ui.info(f"github:   {'logged in' if agent.ctx.github_token else 'no token'}")
                continue
            if cmd == "model":
                if not arg1:
                    ui.info(f"current model: {agent.model}")
                else:
                    agent.model = arg1
                    cfg["model"] = arg1
                    config.save(cfg)
                    # Re-evaluate tool support for the new model.
                    supported = config.get_value(cfg, f"models.{arg1}.tools_supported", True)
                    agent.tools_enabled = bool(supported)
                    agent._refresh_system_prompt()
                    agent.on_tools_disabled = lambda _a: (
                        config.set_value(cfg, f"models.{arg1}.tools_supported", False)
                        or config.save(cfg)
                    )
                    note = "" if supported else " (no tool support — tool-less)"
                    ui.info(f"switched to {arg1}{note} (saved)")
                continue
            if cmd == "models":
                try:
                    for m in agent.client.list_models():
                        marker = " *" if m == agent.model else ""
                        print(f"  - {m}{marker}")
                except OllamaError as e:
                    ui.error(str(e))
                continue
            if cmd == "host":
                if not arg1:
                    ui.info(f"current host: {agent.client.host}")
                else:
                    agent.client.host = arg1.rstrip("/")
                    cfg["host"] = agent.client.host
                    config.save(cfg)
                    ui.info(f"host set to {agent.client.host} (saved)")
                continue
            if cmd == "config":
                import json as _json
                print(_json.dumps(_redact(cfg), indent=2))
                ui.info(f"file: {config.config_path()}")
                continue
            if cmd == "set":
                if not arg1 or not arg2:
                    ui.warn("usage: /set <key> <value>")
                    continue
                v: object = arg2
                if arg2.lower() in ("true", "false"):
                    v = arg2.lower() == "true"
                else:
                    try:
                        v = float(arg2) if "." in arg2 else int(arg2)
                    except ValueError:
                        pass
                config.set_value(cfg, arg1, v)
                config.save(cfg)
                _apply_to_agent(agent, cfg)
                ui.info(f"set {arg1} = {v}")
                continue
            if cmd == "login":
                if arg1.lower() != "github" or not arg2:
                    ui.warn("usage: /login github <token>")
                    continue
                config.set_value(cfg, "github.token", arg2)
                config.save(cfg)
                _apply_to_agent(agent, cfg)
                ui.info("GitHub token saved.")
                continue
            if cmd == "logout":
                if arg1.lower() != "github":
                    ui.warn("usage: /logout github")
                    continue
                config.set_value(cfg, "github.token", None)
                config.save(cfg)
                _apply_to_agent(agent, cfg)
                ui.info("GitHub token removed.")
                continue
            if cmd == "whoami":
                from .github import t_gh_whoami
                print(t_gh_whoami({}, agent.ctx))
                continue
            if cmd == "clear":
                agent.reset()
                ui.info("history cleared")
                continue
            if cmd == "new":
                # Save current (autosave already runs on turns; do a final save with title).
                title = (arg1 + (" " + arg2 if arg2 else "")).strip() or None
                if agent.messages and len(agent.messages) > 1:
                    if title:
                        session["title"] = title
                    _autosave(session, agent)
                    ui.info(f"saved {session['id']}")
                # Spin up a fresh session.
                session.clear()
                session.update(sessions.make(agent.model, title=title))
                agent.reset()
                agent.on_turn_complete = lambda a: _autosave(session, a)
                ui.info(f"new session: {session['id']}")
                continue
            if cmd == "resume":
                listed = _print_sessions(active_id=session.get("id"))
                if not arg1:
                    if listed:
                        ui.info("usage: /resume <id|number>")
                    continue
                target = _resolve_session_arg(arg1, listed)
                if not target:
                    ui.warn(f"no session matching '{arg1}'")
                    continue
                data = sessions.load(target["id"])
                if not data:
                    ui.error(f"could not load {target['id']}")
                    continue
                # Save the current session before swapping.
                if len(agent.messages) > 1:
                    _autosave(session, agent)
                session.clear()
                session.update(data)
                agent.model = data.get("model") or agent.model
                agent.load_messages(data.get("messages", []))
                agent.on_turn_complete = lambda a: _autosave(session, a)
                ui.info(f"resumed {session['id']} — {session.get('title', '')}")
                continue
            if cmd == "sessions":
                _print_sessions(active_id=session.get("id"))
                continue
            if cmd == "save":
                if arg1 or arg2:
                    session["title"] = (arg1 + (" " + arg2 if arg2 else "")).strip()
                _autosave(session, agent)
                ui.info(f"saved {session['id']} — {session.get('title', '')}")
                continue
            if cmd == "delete":
                listed = sessions.list_all()
                target = _resolve_session_arg(arg1, listed)
                if not target:
                    ui.warn(f"no session matching '{arg1}'")
                    continue
                if target["id"] == session.get("id"):
                    ui.warn("can't delete the active session — use /new first")
                    continue
                if sessions.delete(target["id"]):
                    ui.info(f"deleted {target['id']}")
                else:
                    ui.warn("delete failed")
                continue
            if cmd == "yolo":
                agent.ctx.yolo = not agent.ctx.yolo
                cfg["yolo"] = agent.ctx.yolo
                config.save(cfg)
                ui.info(f"yolo: {'ON' if agent.ctx.yolo else 'OFF'}")
                continue
            ui.warn(f"unknown command: /{cmd}")
            continue

        try:
            agent.turn(line)
        except KeyboardInterrupt:
            ui.warn("interrupted")
            continue


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv if argv is not None else sys.argv[1:])

    if args.reset_config:
        try:
            config.config_path().unlink()
            ui.info(f"removed {config.config_path()}")
        except FileNotFoundError:
            ui.info("no config to reset")

    cfg = config.load()

    host = args.host or os.environ.get("OLLAMA_HOST") or cfg.get("host", "http://localhost:11434")
    cfg["host"] = host
    client = OllamaClient(host=host)

    model = args.model or os.environ.get("COLLAMA_MODEL") or cfg.get("model")
    if not model:
        chosen = _pick_model_interactive(client)
        if not chosen:
            ui.error("No model selected. Exiting.")
            return 1
        model = chosen
        cfg["model"] = model
        config.save(cfg)
        ui.info(f"saved model '{model}' to {config.config_path()}")

    root = Path(args.cwd).resolve()
    if not root.is_dir():
        ui.error(f"workspace not a directory: {root}")
        return 2

    try:
        models = client.list_models()
    except OllamaError as e:
        ui.error(str(e))
        ui.warn("Is `ollama serve` running?")
        return 1
    if model not in models and models:
        ui.warn(f"model '{model}' not installed locally. Available: {', '.join(models[:8])}")
        ui.warn(f"Pull it with:  ollama pull {model}")

    temperature = args.temperature if args.temperature is not None else float(cfg.get("temperature", 0.2))
    yolo = args.yolo or bool(cfg.get("yolo", False))

    # Per-model tool-support memory: if we've already learned a model can't
    # do tools (e.g. deepseek-coder), start with tools off rather than probe.
    tools_supported = config.get_value(cfg, f"models.{model}.tools_supported", True)
    if tools_supported is False:
        ui.warn(f"note: '{model}' is known not to support tool calls — running tool-less.")

    def _remember_no_tools(_a):
        config.set_value(cfg, f"models.{model}.tools_supported", False)
        config.save(cfg)

    agent = Agent(
        client=client,
        model=model,
        root=root,
        yolo=yolo,
        temperature=temperature,
        tools_enabled=bool(tools_supported),
        on_tools_disabled=_remember_no_tools,
    )
    agent.ctx.github_token = config.get_value(cfg, "github.token")

    if args.prompt:
        agent.turn(args.prompt)
        return 0
    return repl(agent, cfg)


if __name__ == "__main__":
    sys.exit(main())
