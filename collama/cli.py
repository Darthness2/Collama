"""Command-line entry point."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from . import __version__, config, sessions, ui
from . import diff as _diff
from .agent import Agent
from .coordinator import tick as _coordinator_tick
from .ollama_client import OllamaClient, OllamaError, _is_apple_silicon
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
  /groups [en/disable G]  show or change which tool groups are sent to the model
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
  /stream on|off          toggle token streaming (turn off on networks that break chunked streams)
  /insecure on|off        toggle SSL verification for HTTPS calls (school/corp MITM proxies)
  /diag                   print model / workspace / home / tools / github status
  /model [name]           show or switch model (auto-applies saved /preset)
  /preset [save|clear]    show/save/clear per-model presets (num_ctx, temp, groups…)
  /models                 list locally installed Ollama models
  /host [url]             show or change the Ollama host
  /config                 show current config (token redacted)
  /set <key> <value>      set a config value (e.g. temperature 0.5)
  /login github <token>   save a GitHub Personal Access Token
  /logout github          remove the saved GitHub token
  /whoami                 show authenticated GitHub user
  /clear                  reset conversation history (does not delete saved session)
  /diff [N]               show the last N (default all) file edits this session
  /undo                   revert the most recent file edit
  /retry                  re-run your last message (handy after a bad turn)
  /new [title]            start a new conversation
  /resume [id|number]     list saved conversations or resume one
  /sessions               list saved conversations
  /save [title]           force-save the current conversation (sets title)
  /rename <new title>     rename the current conversation
  /delete <id|number>     delete a saved conversation
  /yolo [on|off]          toggle / set auto-approve for tool calls
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


def _apply_setting_live(agent: Agent, key: str, value) -> bool:
    """Apply a config change to the running agent so /set takes effect now,
    not next session. Returns True if it landed on something live."""
    if key == "temperature":
        try:
            agent.engine.temperature = float(value)
            return True
        except (TypeError, ValueError):
            return False
    if key == "ollama.num_ctx":
        try:
            agent.client.num_ctx = int(value)
            return True
        except (TypeError, ValueError):
            return False
    if key == "ollama.num_predict":
        try:
            agent.client.num_predict = int(value)
            return True
        except (TypeError, ValueError):
            return False
    if key == "ollama.stream":
        agent.engine.stream = bool(value); return True
    if key == "ollama.keep_alive":
        agent.client.keep_alive = value; return True
    if key == "ollama.compact_schemas":
        agent.engine.compact_schemas = bool(value); return True
    if key == "ollama.read_timeout":
        try:
            agent.client.read_timeout = float(value)
            return True
        except (TypeError, ValueError):
            return False
    if key == "ollama.nonstream_read_timeout":
        try:
            agent.client.nonstream_read_timeout = float(value)
            return True
        except (TypeError, ValueError):
            return False
    if key == "yolo":
        agent.state.update(yolo=bool(value)); return True
    if key == "insecure_ssl":
        agent.state.update(insecure_ssl=bool(value)); return True
    if key == "tool_groups" and isinstance(value, list):
        agent.state.update(tool_groups=set(value))
        agent.engine.refresh_system_prompt()
        return True
    return False


def _apply_to_agent(agent: Agent, cfg: dict) -> None:
    agent.ctx.github_token = config.get_value(cfg, "github.token")
    agent.ctx.yolo = bool(cfg.get("yolo", agent.ctx.yolo))
    agent.ctx.insecure_ssl = bool(config.get_value(cfg, "insecure_ssl", False))


def _apply_model_presets(cfg: dict, agent: Agent, model: str) -> list[str]:
    """Apply saved presets for `model` to the live agent. Returns the list of
    keys that were applied (for logging). Quietly ignores keys that aren't
    real settings."""
    presets = config.get_value(cfg, f"models.{model}.presets", {}) or {}
    applied: list[str] = []
    if "num_ctx" in presets:
        agent.client.num_ctx = int(presets["num_ctx"])
        applied.append(f"num_ctx={agent.client.num_ctx}")
    if "num_predict" in presets:
        agent.client.num_predict = int(presets["num_predict"])
        applied.append(f"num_predict={agent.client.num_predict}")
    if "temperature" in presets:
        agent.engine.temperature = float(presets["temperature"])
        applied.append(f"temp={agent.engine.temperature}")
    if "stream" in presets:
        agent.engine.stream = bool(presets["stream"])
        applied.append(f"stream={agent.engine.stream}")
    if "compact_schemas" in presets:
        agent.engine.compact_schemas = bool(presets["compact_schemas"])
        applied.append(f"compact_schemas={agent.engine.compact_schemas}")
    if "tool_groups" in presets and isinstance(presets["tool_groups"], list):
        agent.state.tool_groups = set(presets["tool_groups"])
        agent.engine.refresh_system_prompt()
        applied.append(f"tool_groups={sorted(agent.state.tool_groups)}")
    return applied


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
            if content.startswith((
                "Tool result for ", "[background] ", "[older context",
                "STOP. You have called",          # loop steer
                "STOP. Tool outputs are mine",    # fake-outputs steer
                "You called ", "You have called", # other steer variants
            )):
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

    # VRAM check: warn early if the model is partially CPU-offloaded so the
    # user knows what to expect (slow responses, occasional truncations) and
    # gets a one-line concrete recommendation. Triggers at startup if the
    # model is already resident (kept warm by Ollama's keep_alive), and again
    # after the first turn loads it.
    _warned_offload = {"shown": False}

    def _check_vram_after_turn(a):
        _autosave(session, a)
        if _warned_offload["shown"]:
            return
        st = a.client.model_vram_status(a.model)
        if st and not st["fully_gpu"]:
            size_gb = st["size"] / (1024**3)
            cpu_gb = st["cpu_bytes"] / (1024**3)
            pct = st["cpu_percent"]
            if _is_apple_silicon():
                # Unified memory: nothing is 'spilled to CPU' the way it is
                # on a discrete GPU. The Metal/MLX runner does still hit
                # context-pressure crashes when total resident bytes are
                # close to system RAM though.
                ui.warn(
                    f"{a.model} resident bytes: {size_gb:.1f} GB total "
                    f"({size_gb - cpu_gb:.1f} GB in the Metal buffer). On "
                    f"Apple Silicon all memory is unified, so this isn't 'spillover' "
                    f"per se — but heavy models still cause MLX/Metal context "
                    f"crashes under pressure. If you see freezes, close other "
                    f"GPU-using apps (browser, Discord) or try a smaller model."
                )
            else:
                ui.warn(
                    f"{a.model} is {cpu_gb:.1f} GB on CPU "
                    f"({pct:.0f}% offloaded — total {size_gb:.1f} GB). "
                    f"This is why responses are slow / get truncated. Try a smaller "
                    f"model with /model (qwen2.5-coder:14b fits most 16 GB GPUs)."
                )
            _warned_offload["shown"] = True

    agent.on_turn_complete = _check_vram_after_turn
    agent.engine.session_id = session["id"]
    ui.info(f"new session: {session['id']}")

    # Run the same check up-front in case the model is already resident
    # (Ollama's keep_alive can leave it loaded between Collama runs).
    _check_vram_after_turn(agent)

    # Workspace warning: a workspace set to the home dir means grep / list_dir
    # walk the entire user profile (cache dirs, downloads, etc.) — the model
    # loses focus and bails on /ask-like questions. Nudge the user to /cd into
    # a specific project before doing real work.
    if str(agent.state.workspace).rstrip("\\/") == str(Path.home()).rstrip("\\/"):
        ui.warn(
            f"workspace is your home dir ({agent.state.workspace}) — the model "
            f"will struggle to focus. /cd into a project, launch collama with "
            f"-C <project>, or @path your file in the prompt for sharper results."
        )

    prompt = Prompt()
    if prompt.status_note:
        ui.warn(prompt.status_note)
    if not agent.engine.stream:
        ui.warn("streaming is OFF — you'll only see the answer after generation "
                "completes. Use /stream on to see tokens as they're generated.")
    last_user_input = ""
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
            if cmd == "groups":
                from .tools import TOOL_GROUPS, DEFAULT_GROUPS
                active = agent.state.tool_groups if agent.state.tool_groups is not None else DEFAULT_GROUPS
                if not arg1:
                    ui.info("tool groups (✓ = sent to the model):")
                    for g, names in TOOL_GROUPS.items():
                        mark = ui.color("✓", ui.OK) if g in active else ui.color("·", ui.SOFT)
                        print(f"  {mark} {g:<12} ({len(names)} tools)")
                    ui.info("use /groups enable <name> or /groups disable <name>")
                    continue
                if arg1 in ("enable", "disable") and arg2:
                    if arg2 not in TOOL_GROUPS:
                        ui.warn(f"unknown group '{arg2}' — see /groups")
                        continue
                    groups = set(active)
                    if arg1 == "enable":
                        groups.add(arg2)
                    else:
                        groups.discard(arg2)
                    agent.state.tool_groups = groups
                    agent.engine.refresh_system_prompt()
                    config.set_value(cfg, "tool_groups", sorted(groups))
                    config.save(cfg)
                    ui.info(f"{arg1}d '{arg2}' — {len(groups)} group(s) active (saved)")
                else:
                    ui.warn("usage: /groups  |  /groups enable <name>  |  /groups disable <name>")
                continue
            if cmd == "tools-on":
                agent.tools_enabled = True
                agent.engine.refresh_system_prompt()
                config.set_value(cfg, f"models.{agent.model}.tools_supported", True)
                config.save(cfg)
                ui.info(f"native tools force-enabled for '{agent.model}' (saved).")
                continue
            if cmd == "tools-off":
                agent.tools_enabled = False
                agent.engine.refresh_system_prompt()
                config.set_value(cfg, f"models.{agent.model}.tools_supported", False)
                config.save(cfg)
                ui.info(f"using text-protocol fallback for '{agent.model}' (saved).")
                continue
            if cmd == "cd":
                if not arg1:
                    ui.info(f"workspace: {agent.state.workspace}")
                    continue
                # Resolve relative to the CURRENT workspace, not the shell's
                # cwd — the shell cwd is wherever the user launched collama
                # from, which usually isn't what they mean by 'cd'.
                raw = os.path.expanduser(os.path.expandvars(arg1))
                target = Path(raw)
                if not target.is_absolute():
                    target = agent.state.workspace / target
                target = target.resolve()
                if not target.is_dir():
                    # Helpful: if a same-named dir exists under home, suggest it.
                    hint = ""
                    alt = (agent.state.home / arg1).resolve()
                    if alt.is_dir() and alt != target:
                        hint = f"  Did you mean: {alt} ?"
                    ui.error(f"not a directory: {target}{hint}")
                    continue
                agent.state.update(workspace=target)
                agent.engine.refresh_system_prompt()
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
                results = _coordinator_tick(
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
                from .tools import DEFAULT_GROUPS
                groups = agent.state.tool_groups if agent.state.tool_groups is not None else DEFAULT_GROUPS
                ui.info(f"model:    {agent.model}")
                ws_line = f"workspace: {agent.ctx.root}"
                try:
                    import subprocess as _sp
                    r = _sp.run(
                        ["git", "-C", str(agent.ctx.root), "rev-parse", "--abbrev-ref", "HEAD"],
                        capture_output=True, text=True, timeout=2,
                    )
                    if r.returncode == 0 and r.stdout.strip():
                        ws_line += f"  (git: {r.stdout.strip()})"
                except Exception:
                    pass
                ui.info(ws_line)
                ui.info(f"home:     {Path.home()}")
                ui.info(f"tools:    {'native' if agent.tools_enabled else 'text-protocol fallback'}")
                ui.info(f"groups:   {', '.join(sorted(groups))}")
                ui.info(f"stream:   {'on' if agent.engine.stream else 'off'}")
                ui.info(f"num_ctx:  {agent.client.num_ctx}")
                np_val = agent.client.num_predict
                ui.info(f"num_predict: {'unlimited' if np_val in (None, -1) else np_val}")
                ui.info(f"timeout:  stream {agent.client.read_timeout:.0f}s per-chunk · "
                        f"non-stream {agent.client.nonstream_read_timeout:.0f}s whole-response")
                status = agent.client.model_vram_status(agent.model)
                mac = _is_apple_silicon()
                label = "memory" if mac else "vram"
                if status is None:
                    ui.info(f"{label}:    model not currently loaded (first turn will load it)")
                elif status["fully_gpu"]:
                    place = "in Metal buffer (unified)" if mac else "fully on GPU ✓"
                    ui.info(f"{label}:    {status['size_vram'] / (1024**3):.1f} GB · {place}")
                else:
                    size_gb = status["size"] / (1024**3)
                    vram_gb = status["size_vram"] / (1024**3)
                    cpu_gb = status["cpu_bytes"] / (1024**3)
                    if mac:
                        ui.info(f"{label}:    {size_gb:.1f} GB resident · "
                                f"{vram_gb:.1f} GB in Metal · {cpu_gb:.1f} GB outside "
                                f"(unified — Metal can still reach it; may pressure under load)")
                    else:
                        ui.warn(f"{label}:    {vram_gb:.1f}/{size_gb:.1f} GB on GPU · "
                                f"{cpu_gb:.1f} GB on CPU ({status['cpu_percent']:.0f}% offloaded — "
                                f"expect slowness and truncations)")
                ui.info(f"github:   {'logged in' if agent.ctx.github_token else 'no token'}")
                ui.info(f"ssl:      {'INSECURE (verification off)' if agent.ctx.insecure_ssl else 'verify enabled'}")
                ui.info(f"input:    {prompt.backend}"
                        + ("" if prompt.backend == "prompt_toolkit"
                           else "  (install prompt_toolkit for the / command popup)"))
                # Per-model profile from the lightweight probe.
                profile = config.get_value(cfg, f"models.{agent.model}.profile", {}) or {}
                if profile:
                    n = profile.get("native_tool_calls", 0)
                    s = profile.get("salvaged_tool_calls", 0)
                    total = n + s
                    if total:
                        pct = n / total * 100
                        ui.info(f"profile:  {n} native + {s} salvaged tool calls "
                                f"({pct:.0f}% native)")
                continue
            if cmd == "stream":
                want = arg1.lower() if arg1 else ("off" if agent.engine.stream else "on")
                if want not in ("on", "off"):
                    ui.warn("usage: /stream on|off")
                    continue
                agent.engine.stream = (want == "on")
                config.set_value(cfg, "ollama.stream", agent.engine.stream)
                config.save(cfg)
                ui.info(f"streaming: {'on' if agent.engine.stream else 'off'} (saved)"
                        + ("" if agent.engine.stream else " — use this on networks that break chunked streams"))
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
                    agent.engine.refresh_system_prompt()
                    agent.on_tools_disabled = lambda _a: (
                        config.set_value(cfg, f"models.{arg1}.tools_supported", False)
                        or config.save(cfg)
                    )
                    # Apply saved per-model presets (num_ctx, temperature,
                    # tool_groups, stream, compact_schemas) so each model
                    # remembers its own sweet spot.
                    applied = _apply_model_presets(cfg, agent, arg1)
                    note = "" if supported else " (no tool support — tool-less)"
                    preset_note = f" · presets: {', '.join(applied)}" if applied else ""
                    ui.info(f"switched to {arg1}{note} (saved){preset_note}")
                continue
            if cmd == "preset":
                # /preset           — show current values
                # /preset save      — snapshot current settings under this model
                # /preset clear     — drop saved presets for this model
                sub = arg1.lower() if arg1 else ""
                if sub == "save":
                    presets = {
                        "num_ctx": agent.client.num_ctx,
                        "num_predict": agent.client.num_predict,
                        "temperature": agent.engine.temperature,
                        "stream": agent.engine.stream,
                        "compact_schemas": agent.engine.compact_schemas,
                    }
                    if agent.state.tool_groups is not None:
                        presets["tool_groups"] = sorted(agent.state.tool_groups)
                    config.set_value(cfg, f"models.{agent.model}.presets", presets)
                    config.save(cfg)
                    ui.info(f"saved presets for {agent.model}: {presets}")
                elif sub == "clear":
                    config.set_value(cfg, f"models.{agent.model}.presets", {})
                    config.save(cfg)
                    ui.info(f"cleared presets for {agent.model}")
                else:
                    presets = config.get_value(cfg, f"models.{agent.model}.presets", {}) or {}
                    if not presets:
                        ui.info(f"no presets saved for {agent.model}. /preset save to snapshot current settings.")
                    else:
                        ui.info(f"presets for {agent.model}:")
                        for k, v in presets.items():
                            print(f"    {k} = {v}")
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
                # Coerce value
                v: object = arg2
                if arg2.lower() in ("true", "false"):
                    v = arg2.lower() == "true"
                else:
                    try:
                        v = float(arg2) if "." in arg2 else int(arg2)
                    except ValueError:
                        pass
                # Normalize common keys. 'temperature' is top-level in our
                # config, not under 'ollama.' — but everyone reaches for
                # 'ollama.temperature' first. Accept both, store at the
                # canonical location, and APPLY to the live agent so the
                # change takes effect immediately (the old code only wrote
                # to config; the agent kept using its constructed value).
                ALIASES = {
                    # ollama.X aliases that map to a top-level config key
                    "ollama.temperature": "temperature",
                    "ollama.yolo": "yolo",
                    "ollama.tool_groups": "tool_groups",
                    # Bare names that should canonicalize under ollama.*
                    "num_ctx":               "ollama.num_ctx",
                    "num_predict":           "ollama.num_predict",
                    "max_tokens":            "ollama.num_predict",
                    "stream":                "ollama.stream",
                    "keep_alive":            "ollama.keep_alive",
                    "compact_schemas":       "ollama.compact_schemas",
                    "read_timeout":          "ollama.read_timeout",
                    "nonstream_read_timeout":"ollama.nonstream_read_timeout",
                    "connect_timeout":       "ollama.connect_timeout",
                    "host":                  "host",  # already top-level
                    # plain aliases
                    "temp":                  "temperature",
                }
                key = ALIASES.get(arg1, arg1)
                config.set_value(cfg, key, v)
                config.save(cfg)
                applied_live = _apply_setting_live(agent, key, v)
                _apply_to_agent(agent, cfg)
                if applied_live:
                    live_note = "  → applied live · /preset save to lock it for this model"
                else:
                    live_note = "  → config only (unknown key; takes effect next session if recognized then)"
                ui.info(f"set {key} = {v}{live_note}")
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
            if cmd == "diff":
                # Show what the model has edited since the session started.
                hist = list(agent.state.edit_history or [])
                if not hist:
                    ui.info("(no edits this session)")
                    continue
                limit = int(arg1) if arg1.isdigit() else len(hist)
                for entry in hist[-limit:]:
                    path = entry.get("path", "?")
                    op = entry.get("op", "edit")
                    before = entry.get("before", "")
                    after = entry.get("after", "")
                    adds, dels = _diff.stats(before, after)
                    print(ui.color(f"  {op}  {path}  ", ui.TEAL_BRIGHT) +
                          ui.color(f"(+{adds} -{dels})", ui.MUTED))
                    rendered = _diff.render(before, after, path, max_lines=20)
                    if rendered:
                        print(rendered)
                continue
            if cmd == "undo":
                hist = list(agent.state.edit_history or [])
                if not hist:
                    ui.info("(no edits to undo)")
                    continue
                entry = hist.pop()
                p = Path(entry["path"])
                try:
                    p.write_text(entry.get("before", ""))
                except OSError as e:
                    ui.error(f"undo failed: {e}")
                    continue
                agent.state.update(edit_history=hist)
                ui.info(f"reverted {entry.get('op', 'edit')} on {p}")
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
                # Reset workspace so the model doesn't write files into the
                # previous chat's project. Default = home dir; override with
                # `collama.new_chat_workspace` ("home" | "subdir" | <abs path>).
                # "subdir" creates ~/collama/<session_id>/ on demand.
                pref = str(config.get_value(cfg, "collama.new_chat_workspace", "home") or "home")
                if pref == "subdir":
                    target = Path.home() / "collama" / session["id"]
                    target.mkdir(parents=True, exist_ok=True)
                elif pref == "home":
                    target = Path.home()
                else:
                    target = Path(os.path.expanduser(pref))
                    if not target.exists():
                        target = Path.home()
                agent.state.update(workspace=target)
                ui.info(f"new session: {session['id']}  ·  workspace → {target}")
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
            if cmd == "rename":
                new_title = (arg1 + (" " + arg2 if arg2 else "")).strip()
                if not new_title:
                    ui.info(f"current title: {session.get('title', '(untitled)')}")
                    ui.warn("usage: /rename <new title>")
                    continue
                old = session.get("title", "(untitled)")
                session["title"] = new_title
                _autosave(session, agent)
                ui.info(f"renamed: '{old}' → '{new_title}'")
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
                # /yolo on|off  or  /yolo (toggle)
                sub = arg1.lower() if arg1 else ""
                if sub in ("on", "true", "1", "yes"):
                    want = True
                elif sub in ("off", "false", "0", "no"):
                    want = False
                elif sub == "":
                    want = not agent.state.yolo
                else:
                    ui.warn("usage: /yolo on|off  (or /yolo to toggle)")
                    continue
                # Write to AppState so future ToolContext objects pick it up
                # (the agent.ctx property builds a fresh ToolContext each call).
                agent.state.update(yolo=want)
                cfg["yolo"] = want
                config.save(cfg)
                ui.info(f"yolo: {'ON — no further approval prompts' if want else 'OFF'} (saved)")
                continue
            if cmd == "retry":
                if not last_user_input:
                    ui.warn("nothing to retry yet")
                    continue
                ui.info(f"retrying: {last_user_input[:80]}")
                line = last_user_input
                # fall through to the turn dispatch below
            else:
                ui.warn(f"unknown command: /{cmd}")
                continue

        last_user_input = line
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
        read_timeout=float(config.get_value(cfg, "ollama.read_timeout", 1800.0)),
        nonstream_read_timeout=float(config.get_value(cfg, "ollama.nonstream_read_timeout", 1800.0)),
        keep_alive=config.get_value(cfg, "ollama.keep_alive", "30m"),
        num_ctx=config.get_value(cfg, "ollama.num_ctx", 8192),
        num_predict=config.get_value(cfg, "ollama.num_predict", -1),
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

    # If launched from inside the Collama install dir itself (and -C wasn't
    # explicitly given), default the workspace to the home dir — otherwise the
    # model creates new projects inside the Collama repo.
    if args.cwd == ".":
        collama_root = Path(__file__).resolve().parent.parent
        inside_install = root == collama_root or collama_root in root.parents
        if inside_install:
            root = Path.home()
            ui.warn(f"launched inside the Collama install dir — workspace set to your "
                    f"home dir ({root}) so new projects don't land in the Collama repo. "
                    f"Use -C <path> or /cd to choose a different workspace.")

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
        stream=bool(config.get_value(cfg, "ollama.stream", True)),
    )
    agent.ctx.github_token = config.get_value(cfg, "github.token")
    agent.ctx.insecure_ssl = bool(config.get_value(cfg, "insecure_ssl", False))

    # Enabled tool groups (None = tools.DEFAULT_GROUPS).
    saved_groups = config.get_value(cfg, "tool_groups", None)
    if isinstance(saved_groups, list) and saved_groups:
        agent.state.tool_groups = set(saved_groups)

    # Unload the model on exit so closing the window / quitting doesn't leave
    # the model resident in VRAM. We unload whatever model is current at
    # shutdown — works for both normal exit (/exit, EOF) and abrupt closes
    # (Ctrl+C, SIGTERM, console window close on Windows -> SIGBREAK).
    import atexit
    import signal

    _unloaded = {"done": False}

    def _shutdown_unload(*_args):
        if _unloaded["done"]:
            return
        _unloaded["done"] = True
        try:
            client.unload(agent.model)
        except Exception:
            pass

    atexit.register(_shutdown_unload)
    for sig_name in ("SIGTERM", "SIGBREAK", "SIGHUP"):
        sig = getattr(signal, sig_name, None)
        if sig is None:
            continue
        try:
            signal.signal(sig, lambda *_: (_shutdown_unload(), sys.exit(0)))
        except (ValueError, OSError):
            pass

    # Apply per-model presets (num_ctx, temperature, tool_groups, etc.) for
    # whichever model is loaded at startup, so each model remembers its
    # tuning across sessions.
    applied = _apply_model_presets(cfg, agent, agent.model)
    if applied:
        ui.info(f"applied {agent.model} presets: {', '.join(applied)}")

    if args.prompt:
        agent.turn(args.prompt)
        return 0
    return repl(agent, cfg)


if __name__ == "__main__":
    sys.exit(main())
