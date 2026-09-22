"""Single-lane inference queue with session-scoped, Conda-isolated workers."""

from __future__ import annotations

import json
import logging
import os
import signal
import shutil
import sqlite3
import subprocess
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


LOGGER = logging.getLogger("cbt_live_interaction.inference")

RUNTIME_BY_METHOD = {
    "prompting": "base",
    "proact": "base",
    "topas": "base",
    "archer": "archer",
    "aria": "aria",
    "sweet_rl": "sweet_rl",
}

RUNTIME_DEFAULTS = {
    "base": {"env": "base", "gpu": "0"},
    "archer": {"env": "archer_env", "gpu": "0"},
    "aria": {"env": "aria_env", "gpu": "3"},
    "sweet_rl": {"env": "sweet_rl", "gpu": "3"},
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _connect(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path, timeout=30, isolation_level=None)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA busy_timeout = 30000")
    return connection


@contextmanager
def _database(path: Path):
    connection = _connect(path)
    try:
        yield connection
    finally:
        connection.close()


class ModelWorker:
    """One session's JSON-lines subprocess running inside its model Conda env."""

    def __init__(self, runtime: str, method: str, panel_id: str, server_dir: Path) -> None:
        self.runtime = runtime
        self.method = method
        self.panel_id = panel_id
        self.server_dir = server_dir
        upper = runtime.upper()
        defaults = RUNTIME_DEFAULTS[runtime]
        self.conda_env = os.getenv(f"STUDY_CONDA_ENV_{upper}", defaults["env"])
        self.gpu = os.getenv(f"STUDY_GPU_{upper}", defaults["gpu"])
        self.process: subprocess.Popen[str] | None = None
        self.log_handle: Any = None

    def _conda_executable(self) -> str:
        configured = os.getenv("STUDY_CONDA_EXE") or os.getenv("CONDA_EXE")
        if configured:
            return configured
        executable = shutil.which("conda")
        if not executable:
            raise RuntimeError("Conda was not found. Set STUDY_CONDA_EXE to the full conda executable path.")
        return executable

    def start(self) -> None:
        if self.process is not None and self.process.poll() is None:
            return
        self.stop()
        log_dir = self.server_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        self.log_handle = (log_dir / f"model-{self.runtime}.log").open("a", encoding="utf-8", buffering=1)
        command = [
            self._conda_executable(),
            "run",
            "--no-capture-output",
            "-n",
            self.conda_env,
            "python",
            "-u",
            str(self.server_dir / "method_worker.py"),
            "--runtime",
            self.runtime,
            "--method",
            self.method,
        ]
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = self.gpu
        environment["PYTHONUNBUFFERED"] = "1"
        LOGGER.info(
            "Starting panel=%s method=%s runtime=%s conda_env=%s gpu=%s",
            self.panel_id,
            self.method,
            self.runtime,
            self.conda_env,
            self.gpu,
        )
        self.process = subprocess.Popen(
            command,
            cwd=self.server_dir,
            env=environment,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self.log_handle,
            text=True,
            encoding="utf-8",
            bufsize=1,
            start_new_session=True,
        )

    def generate(self, method: str, history: list[dict[str, str]]) -> str:
        if method != self.method:
            raise RuntimeError(
                f"Session worker for {self.method!r} cannot run method {method!r}."
            )
        self.start()
        assert self.process is not None and self.process.stdin is not None and self.process.stdout is not None
        request_id = str(uuid.uuid4())
        request = {"id": request_id, "method": method, "history": history}
        try:
            self.process.stdin.write(json.dumps(request, ensure_ascii=False) + "\n")
            self.process.stdin.flush()
            line = self.process.stdout.readline()
        except (BrokenPipeError, OSError) as exc:
            self.stop()
            raise RuntimeError(f"The {self.runtime} model worker stopped unexpectedly.") from exc
        if not line:
            return_code = self.process.poll()
            self.stop()
            raise RuntimeError(f"The {self.runtime} model worker exited (code {return_code}).")
        try:
            response = json.loads(line)
        except json.JSONDecodeError as exc:
            self.stop()
            raise RuntimeError(f"The {self.runtime} model worker returned invalid protocol data.") from exc
        if response.get("id") != request_id:
            self.stop()
            raise RuntimeError(f"The {self.runtime} model worker returned a mismatched response.")
        if not response.get("ok"):
            raise RuntimeError(str(response.get("error") or "Model generation failed."))
        utterance = str(response.get("utterance") or "").strip()
        if not utterance:
            raise RuntimeError("The model returned an empty therapist response.")
        return utterance

    def stop(self) -> None:
        process = self.process
        self.process = None
        if process is not None and process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except (AttributeError, OSError):
                process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except (AttributeError, OSError):
                    process.kill()
                process.wait(timeout=5)
        if self.log_handle is not None:
            self.log_handle.close()
            self.log_handle = None

    @property
    def running(self) -> bool:
        return self.process is not None and self.process.poll() is None


class InferenceManager:
    """Processes all browser requests through one FIFO generation lane."""

    def __init__(self, database_path: Path, server_dir: Path, mode: str = "real") -> None:
        self.database_path = database_path
        self.server_dir = server_dir
        self.mode = mode
        # Workers are owned by panels, not by model type. This preserves TOPAS'
        # option state during one dialogue and lets us release the entire model
        # process as soon as that session ends.
        self.workers: dict[str, ModelWorker] = {}
        self._workers_lock = threading.RLock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        with _database(self.database_path) as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                "UPDATE inference_jobs SET status = 'queued', started_at = NULL "
                "WHERE status = 'running'"
            )
            connection.commit()
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="study-inference-queue", daemon=True)
        self._thread.start()
        self._wake.set()

    def notify(self) -> None:
        self._wake.set()

    def stop(self) -> None:
        self._stop.set()
        self._wake.set()
        # Stop subprocess groups first so a thread blocked on a model response
        # is released and no `conda run` child is left holding GPU memory.
        with self._workers_lock:
            workers = list(self.workers.values())
            self.workers.clear()
        for worker in workers:
            worker.stop()
        if self._thread is not None:
            self._thread.join(timeout=20)
            self._thread = None

    def _claim_next_job(self) -> sqlite3.Row | None:
        with _database(self.database_path) as connection:
            connection.execute("BEGIN IMMEDIATE")
            job = connection.execute(
                """
                SELECT j.*, p.method_key, p.study_id
                FROM inference_jobs AS j
                JOIN study_panels AS p ON p.id = j.panel_id
                JOIN studies AS s ON s.id = p.study_id
                WHERE j.status = 'queued' AND s.status = 'active'
                ORDER BY j.sequence
                LIMIT 1
                """
            ).fetchone()
            if job is None:
                connection.commit()
                return None
            connection.execute(
                "UPDATE inference_jobs SET status = 'running', started_at = ? WHERE id = ?",
                (utc_now(), job["id"]),
            )
            connection.commit()
            return job

    def _history(self, panel_id: str) -> list[dict[str, str]]:
        with _database(self.database_path) as connection:
            rows = connection.execute(
                "SELECT role, content FROM panel_messages WHERE panel_id = ? ORDER BY id",
                (panel_id,),
            ).fetchall()
        return [{"role": row["role"], "content": row["content"]} for row in rows]

    def _static_response(self, method: str, history: list[dict[str, str]]) -> str:
        if not history:
            return "Hello, it is good to meet you. What brought you in today?"
        if method == "topas":
            return (
                "Thank you for sharing that. What thoughts or feelings tend to come up most strongly "
                "when this happens?"
            )
        return (
            "I appreciate you telling me that. Could you say a little more about what feels most "
            "difficult for you right now?"
        )

    def _generate(
        self,
        panel_id: str,
        method: str,
        history: list[dict[str, str]],
    ) -> str:
        if self.mode != "real":
            return self._static_response(method, history)
        runtime = RUNTIME_BY_METHOD[method]
        with self._workers_lock:
            worker = self.workers.get(panel_id)
            if worker is None:
                worker = ModelWorker(runtime, method, panel_id, self.server_dir)
                self.workers[panel_id] = worker
        return worker.generate(method, history)

    def release_panel(self, panel_id: str) -> None:
        """Unload the model and policy state owned by a finished session."""
        with self._workers_lock:
            worker = self.workers.pop(panel_id, None)
        if worker is not None:
            LOGGER.info(
                "Releasing panel=%s method=%s runtime=%s",
                panel_id,
                worker.method,
                worker.runtime,
            )
            worker.stop()

    def _complete_job(self, job: sqlite3.Row, utterance: str) -> None:
        now = utc_now()
        with _database(self.database_path) as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                "INSERT INTO panel_messages(panel_id, role, content, created_at) VALUES (?, 'therapist', ?, ?)",
                (job["panel_id"], utterance, now),
            )
            connection.execute(
                "UPDATE inference_jobs SET status = 'completed', completed_at = ?, error = NULL WHERE id = ?",
                (now, job["id"]),
            )
            connection.execute("UPDATE studies SET updated_at = ? WHERE id = ?", (now, job["study_id"]))
            connection.commit()

    def _fail_job(self, job: sqlite3.Row, exc: Exception) -> None:
        LOGGER.exception("Inference job %s failed for its hidden method.", job["id"], exc_info=exc)
        now = utc_now()
        with _database(self.database_path) as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                "UPDATE inference_jobs SET status = 'failed', completed_at = ?, error = ? WHERE id = ?",
                (now, str(exc)[:2000], job["id"]),
            )
            connection.execute("UPDATE studies SET updated_at = ? WHERE id = ?", (now, job["study_id"]))
            connection.commit()

    def _run(self) -> None:
        while not self._stop.is_set():
            job = self._claim_next_job()
            if job is None:
                self._wake.clear()
                self._wake.wait(timeout=1)
                continue
            try:
                history = self._history(job["panel_id"])
                utterance = self._generate(job["panel_id"], job["method_key"], history)
                self._complete_job(job, utterance)
            except Exception as exc:  # A failed model must not stop later queued jobs.
                self._fail_job(job, exc)

    def health(self) -> dict[str, Any]:
        with _database(self.database_path) as connection:
            queued = connection.execute(
                "SELECT COUNT(*) FROM inference_jobs WHERE status = 'queued'"
            ).fetchone()[0]
            running = connection.execute(
                "SELECT COUNT(*) FROM inference_jobs WHERE status = 'running'"
            ).fetchone()[0]
        with self._workers_lock:
            loaded_workers = sum(worker.running for worker in self.workers.values())
        return {
            "mode": self.mode,
            "queue_depth": int(queued),
            "busy": bool(running),
            "loaded_workers": loaded_workers,
        }
