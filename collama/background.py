"""Background tasks.

A small daemon-thread executor that runs slow ops (shell commands, sub-agent
queries) without blocking the agent's main loop. Each submission returns a
task_id immediately; on completion, a notification is enqueued. The engine
drains pending notifications before each Ollama call and injects them into
the conversation as user messages so the model can react.

Two kinds of background work:
  - bash_async(cmd)   — run a shell command in a thread.
  - dream(prompt)     — fork a sub-agent in a thread; result comes back later.
"""
from __future__ import annotations

import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from .tasks import TaskGraph, TaskKind, new_id


@dataclass
class BackgroundJob:
    id: str
    kind: str          # "bash" | "dream"
    label: str         # the cmd / prompt
    status: str = "running"  # running | done | failed
    result: str = ""
    started_at: float = field(default_factory=time.time)
    finished_at: float = 0.0


class BackgroundExecutor:
    def __init__(self, tasks: TaskGraph | None = None) -> None:
        self.tasks = tasks
        self._jobs: dict[str, BackgroundJob] = {}
        self._notifications: list[dict] = []
        self._lock = threading.Lock()
        self._threads: list[threading.Thread] = []

    # -- public ----------------------------------------------------------

    def submit_bash(self, command: str, cwd: Path, *, timeout: int = 600) -> str:
        job_id = new_id(TaskKind.BASH)
        job = BackgroundJob(id=job_id, kind="bash", label=command)
        with self._lock:
            self._jobs[job_id] = job
        if self.tasks is not None:
            self.tasks.create(
                title=f"bash: {command[:60]}",
                kind=TaskKind.BASH, status="active",
                description=command, worktree=str(cwd),
            )
        t = threading.Thread(
            target=self._run_bash, args=(job_id, command, cwd, timeout), daemon=True,
        )
        t.start()
        self._threads.append(t)
        return job_id

    def submit_dream(self, prompt: str, run: Callable[[str], str]) -> str:
        """Run an arbitrary callable in the background. `run(prompt)` should
        return the final text. Used by the agent_call tool's async variant."""
        job_id = new_id(TaskKind.DREAM)
        job = BackgroundJob(id=job_id, kind="dream", label=prompt[:120])
        with self._lock:
            self._jobs[job_id] = job
        t = threading.Thread(
            target=self._run_dream, args=(job_id, prompt, run), daemon=True,
        )
        t.start()
        self._threads.append(t)
        return job_id

    def status(self, job_id: str) -> Optional[BackgroundJob]:
        with self._lock:
            return self._jobs.get(job_id)

    def list(self) -> list[BackgroundJob]:
        with self._lock:
            return sorted(self._jobs.values(), key=lambda j: j.started_at)

    def wait(self, job_id: str, timeout: float = 60.0) -> Optional[BackgroundJob]:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            j = self.status(job_id)
            if j and j.status != "running":
                return j
            time.sleep(0.1)
        return self.status(job_id)

    def drain_notifications(self) -> list[dict]:
        with self._lock:
            out = list(self._notifications)
            self._notifications.clear()
        return out

    # -- runners ---------------------------------------------------------

    def _finish(self, job_id: str, status: str, result: str) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return
            job.status = status
            job.result = result
            job.finished_at = time.time()
            self._notifications.append({
                "id": job_id, "kind": job.kind, "label": job.label,
                "status": status, "result": result[:2000],
            })
        if self.tasks is not None:
            t = self.tasks.list()  # crude lookup; bash jobs are recent
            for task in t:
                if task.kind == "b" and task.description == job.label and task.status == "active":
                    self.tasks.update(task.id, status="done" if status == "done" else "failed",
                                      result=result[:2000])
                    break

    def _run_bash(self, job_id: str, command: str, cwd: Path, timeout: int) -> None:
        try:
            proc = subprocess.run(
                command, shell=True, cwd=str(cwd),
                capture_output=True,
                encoding="utf-8", errors="replace", timeout=timeout,
            )
            ok = proc.returncode == 0
            parts = [f"exit code: {proc.returncode}"]
            if proc.stdout:
                parts.append(f"--- stdout ---\n{proc.stdout}")
            if proc.stderr:
                parts.append(f"--- stderr ---\n{proc.stderr}")
            self._finish(job_id, "done" if ok else "failed", "\n".join(parts))
        except subprocess.TimeoutExpired:
            self._finish(job_id, "failed", f"timed out after {timeout}s")
        except Exception as e:
            self._finish(job_id, "failed", f"{type(e).__name__}: {e}")

    def _run_dream(self, job_id: str, prompt: str, run: Callable[[str], str]) -> None:
        try:
            text = run(prompt) or ""
            self._finish(job_id, "done", text)
        except Exception as e:
            self._finish(job_id, "failed", f"{type(e).__name__}: {e}")
