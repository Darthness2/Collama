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

import logging
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from .tasks import TaskGraph, TaskKind, new_id

logger = logging.getLogger(__name__)

# Cap on concurrently-running background jobs. Without a bound, an LLM that
# fires off bash_async / agent_call_async in a loop spawns unbounded threads
# (and subprocesses), exhausting the host. Submissions beyond this block until
# a slot frees up.
MAX_CONCURRENT_JOBS = 8


@dataclass
class BackgroundJob:
    id: str
    kind: str          # "bash" | "dream"
    label: str         # the cmd / prompt
    status: str = "running"  # running | done | failed
    result: str = ""
    started_at: float = field(default_factory=time.time)
    finished_at: float = 0.0
    # Concrete id of the persistent TaskGraph row this job created (if any), so
    # _finish can update the right task by id instead of fuzzy-matching on the
    # command/description string.
    task_id: str | None = None


class BackgroundExecutor:
    def __init__(self, tasks: TaskGraph | None = None,
                 max_concurrent: int = MAX_CONCURRENT_JOBS) -> None:
        self.tasks = tasks
        self._jobs: dict[str, BackgroundJob] = {}
        self._notifications: list[dict] = []
        self._lock = threading.Lock()
        self._threads: list[threading.Thread] = []
        # Bound concurrent in-flight jobs. Acquired by the worker thread, not
        # the submitter, so submit_* returns immediately; the worker releases
        # it in a finally so a crash can't leak a permit.
        self._slots = threading.BoundedSemaphore(max(1, int(max_concurrent)))

    # -- public ----------------------------------------------------------

    def submit_bash(self, command: str, cwd: Path, *, timeout: int = 600) -> str:
        job_id = new_id(TaskKind.BASH)
        job = BackgroundJob(id=job_id, kind="bash", label=command)
        if self.tasks is not None:
            task = self.tasks.create(
                title=f"bash: {command[:60]}",
                kind=TaskKind.BASH, status="active",
                description=command, worktree=str(cwd),
            )
            job.task_id = task.id
        with self._lock:
            self._jobs[job_id] = job
        t = threading.Thread(
            target=self._run_bash, args=(job_id, command, cwd, timeout), daemon=True,
        )
        t.start()
        self._reap_threads()
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
        self._reap_threads()
        self._threads.append(t)
        return job_id

    def _reap_threads(self) -> None:
        """Drop references to finished threads so the list can't grow without
        bound over a long session."""
        self._threads = [t for t in self._threads if t.is_alive()]

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
            task_id = job.task_id
            self._notifications.append({
                "id": job_id, "kind": job.kind, "label": job.label,
                "status": status, "result": result[:2000],
            })
        # Update the persistent task by its concrete id — never by matching the
        # command/description string (two jobs can share a command, and a
        # string match can hit the wrong row).
        if self.tasks is not None and task_id is not None:
            self.tasks.update(
                task_id,
                status="done" if status == "done" else "failed",
                result=result[:2000],
            )

    def _run_bash(self, job_id: str, command: str, cwd: Path, timeout: int) -> None:
        self._slots.acquire()
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
            logger.warning("background bash job %s failed", job_id, exc_info=True)
            self._finish(job_id, "failed", f"{type(e).__name__}: {e}")
        finally:
            self._slots.release()

    def _run_dream(self, job_id: str, prompt: str, run: Callable[[str], str]) -> None:
        self._slots.acquire()
        try:
            text = run(prompt) or ""
            self._finish(job_id, "done", text)
        except Exception as e:
            logger.warning("background dream job %s failed", job_id, exc_info=True)
            self._finish(job_id, "failed", f"{type(e).__name__}: {e}")
        finally:
            self._slots.release()
