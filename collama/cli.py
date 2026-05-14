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
  /tasks                  list persistent tasks (s07)
  /jobs                   list background jobs (s08)
  /wt                     show worktree stack (s12)
  /teams                  list teams and teammates (s09)
  /tick [team] [claim]    coordinator tick — process mailboxes; pass 'claim' to auto-claim tasks (s11)
  /plan on|off            toggle plan mode (read-only, no mutating tools)
  /todo [add|done|clear]  view or modify the session todo list
  /brief [name]           list briefs, or print one
  /insecure on|off        toggle SSL verification for HTTPS calls (school/corp MITM proxies)
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
    agent.ctx.insecure_ssl = bool(config.get_value(cfg, "insecure_ssl", False))


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


def _replay_conversation(messages: list[dict], max_turns: int = 12) -> None:
    """Print a saved conversation's history so a resumed session has context.

    Shows the last `max_turns` user turns (and the assistant/tool activity
    between them); older history is summarized as a one-line marker.
    """
    convo = [m for m in messages if m.get("role") != "system"]
    if not convo:
        ui.info("(empty conversation)")
        return

    # Find where to start: keep the tail with at most `max_turns` user msgs.
    user_idxs = [i for i, m in enumerate(convo) if m.get("role") == "user"]
    start = 0
    if len(user_idxs) > max_turns:
        start = user_idxs[-max_turns]
        ui.info(f"… {user_idxs.index(user_idxs[-max_turns])} earlier turn(s) hidden")

    ui.hr()
    for m in convo[start:]:
        role = m.get("role")
        content = (m.get("content") or "").strip()
        if role == "user":
            # Skip our own injected tool-result / background frames.
            if content.startswith(("Tool result for ", "[background] ", "[older context")):
                continue
            print(ui.color("❯ ", ui.TEAL_BRIGHT) + ui.color(content, ui.SURFACE))
        elif role == "assistant":
            if content:
                ui.assistant(content)
            for tc in m.get("tool_calls") or []:
                fn = tc.get("function", {}) or {}
                print(ui.color("  ▸ ", ui.TEAL_BRIGHT) + ui.color(fn.get("name", "?"), ui.TEAL_BRIGHT))
        elif role == "tool":
            first = content.splitlines()[0][:120] if content else ""
            ui.tool_result(first, ok=not first.startswith("ERROR"))
    ui.hr()


def repl(agent: Agent, cfg: dict) -> int:
    ui.banner(agent.model, str(agent.ctx.root), tools_enabled=agent.tools_enabled)

    # Active session (auto-created, auto-saved after each turn)
    session = sessions.make(agent.model)
    agent.on_turn_complete = lambda a: _autosave(session, a)
    agent.engine.session_id = session["id"]
    ui.info(f"new session: {session['id']}")

    prompt = Prompt()
    if prompt.status_note:
        ui.warn(prompt.status_note)
    while True:
        ui.prepare_for_input()
        # The blank separator line must be printed SEPARATELY — never embed a
        # newline in the prompt string. prompt_toolkit computes cursor/line
        # position relative to the prompt; a '\n' inside it corrupts that
        # math and causes glitches when editing/pasting (and misplaces the
        # completion popup), especially on macOS terminals.
        print()
        try:
            line = prompt.ask(ui.color("❯ ", ui.TEAL_BRIGHT)).strip()
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
            if cmd == "tasks":
                tasks = agent.engine.task_graph.list()
                if not tasks:
                    ui.info("(no tasks)")
                else:
                    for t in tasks[:50]:
                        print(f"  {t.short()}")
                continue
            if cmd == "jobs":
                jobs = agent.engine.background.list()
                if not jobs:
                    ui.info("(no background jobs)")
                else:
                    for j in jobs:
                        first = j.result.splitlines()[0][:80] if j.result else ""
                        print(f"  {j.id}  [{j.status:<7}] {j.kind:<6} {j.label[:40]}  {first}")
                continue
            if cmd == "wt":
                stack = list(agent.state.worktree_stack or [])
                ui.info(f"workspace: {agent.state.workspace}")
                ui.info(f"worktree stack ({len(stack)}): " + (", ".join(stack) if stack else "(empty)"))
                continue
            if cmd == "teams":
                reg = agent.engine.teams
                teams = reg.list_teams()
                if not teams:
                    ui.info("(no teams)")
                else:
                    for t in teams:
                        members = reg.list_teammates(t)
                        ui.info(f"{t}  ({len(members)})")
                        for m in members:
                            print(f"    - {m.short()}")
                continue
            if cmd == "tick":
                from .coordinator import tick as _tick
                results = _tick(
                    agent.engine,
                    team=arg1 or None,
                    auto_claim=(arg2.lower() == "claim") if arg2 else False,
                )
                if not results:
                    ui.info("(idle — no teammate had work)")
                for r in results:
                    ui.info(f"  → {r.teammate}  inbox={r.inbox_count}  claimed={r.claimed_task_id}")
                    first = r.answer.splitlines()[0][:120] if r.answer else ""
                    if first:
                        print(ui.color(f"      {first}", ui.MUTED))
                continue
            if cmd == "plan":
                want = arg1.lower() if arg1 else ("off" if agent.state.plan_mode else "on")
                if want not in ("on", "off"):
                    ui.warn("usage: /plan on|off")
                    continue
                agent.state.update(plan_mode=(want == "on"))
                agent.engine.refresh_system_prompt()
                ui.info(f"plan mode: {'ON (read-only)' if agent.state.plan_mode else 'off'}")
                continue
            if cmd == "todo":
                todos = list(agent.state.todos or [])
                if not arg1:
                    if not todos:
                        ui.info("(no todos)")
                    for i, t in enumerate(todos, 1):
                        mark = {"done": "✓", "pending": " ", "active": "▸", "blocked": "✗"}.get(t.get("status", "pending"), "?")
                        print(f"  [{mark}] {i}. {t.get('text', '')}")
                    continue
                # /todo add <text>  or  /todo done N  or  /todo clear
                if arg1 == "add":
                    text = arg2.strip()
                    if not text:
                        ui.warn("usage: /todo add <text>")
                        continue
                    todos.append({"text": text, "status": "pending"})
                    agent.state.update(todos=todos)
                    ui.info(f"added: {text}")
                    continue
                if arg1 == "done" and arg2.isdigit():
                    i = int(arg2) - 1
                    if 0 <= i < len(todos):
                        todos[i]["status"] = "done"
                        agent.state.update(todos=todos)
                        ui.info(f"done: {todos[i]['text']}")
                    continue
                if arg1 == "clear":
                    agent.state.update(todos=[])
                    ui.info("cleared todos")
                    continue
                ui.warn("usage: /todo  |  /todo add <text>  |  /todo done <n>  |  /todo clear")
                continue
            if cmd == "brief":
                briefs = agent.state.briefs
                if not arg1:
                    if not briefs:
                        ui.info("(no briefs)")
                    for k, v in briefs.items():
                        print(f"  - {k}  ({len(v)} chars)")
                    continue
                if arg1 in briefs:
                    print(briefs[arg1])
                else:
                    ui.warn(f"no brief named '{arg1}'")
                continue
            if cmd == "diag":
                ui.info(f"model:    {agent.model}")
                ui.info(f"workspace: {agent.ctx.root}")
                ui.info(f"home:     {Path.home()}")
                ui.info(f"tools:    {'native' if agent.tools_enabled else 'text-protocol fallback'}")
                ui.info(f"github:   {'logged in' if agent.ctx.github_token else 'no token'}")
                ui.info(f"ssl:      {'INSECURE (verification off)' if agent.ctx.insecure_ssl else 'verify enabled'}")
                ui.info(f"input:    {prompt.backend}"
                        + ("" if prompt.backend == "prompt_toolkit"
                           else "  (install prompt_toolkit for the / command popup)"))
                continue
            if cmd == "insecure":
                want = arg1.lower() if arg1 else ("off" if agent.ctx.insecure_ssl else "on")
                if want not in ("on", "off"):
                    ui.warn("usage: /insecure on|off")
                    continue
                agent.ctx.insecure_ssl = (want == "on")
                config.set_value(cfg, "insecure_ssl", agent.ctx.insecure_ssl)
                config.save(cfg)
                if agent.ctx.insecure_ssl:
                    ui.warn("SSL verification DISABLED for outbound HTTPS (e.g. GitHub).")
                    ui.warn("Use only on networks that intercept TLS (school/corp proxies).")
                else:
                    ui.info("SSL verification re-enabled.")
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
                agent.engine.session_id = session["id"]
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
                saved_messages = data.get("messages", [])
                agent.load_messages(saved_messages)
                agent.on_turn_complete = lambda a: _autosave(session, a)
                agent.engine.session_id = session["id"]
                ui.info(f"resumed {session['id']} — {session.get('title', '')}")
                _replay_conversation(saved_messages)
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
    # keep_alive keeps the model resident in VRAM between turns (avoids a
    # cold disk->VRAM reload every turn); read_timeout is the max gap between
    # streamed tokens, so long generations never hit a wall.
    client = OllamaClient(
        host=host,
        connect_timeout=float(config.get_value(cfg, "ollama.connect_timeout", 15.0)),
        read_timeout=float(config.get_value(cfg, "ollama.read_timeout", 600.0)),
        keep_alive=config.get_value(cfg, "ollama.keep_alive", "30m"),
    )

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
    agent.ctx.insecure_ssl = bool(config.get_value(cfg, "insecure_ssl", False))

    if args.prompt:
        agent.turn(args.prompt)
        return 0
    return repl(agent, cfg)


if __name__ == "__main__":
    sys.exit(main())
