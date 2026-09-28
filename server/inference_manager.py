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
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

if __package__:
    from .session_rules import farewell_phrase
else:
    from session_rules import farewell_phrase


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
    "base": {"env": "base", "min_free_mb": 18000},
    "archer": {"env": "archer_env", "min_free_mb": 20000},
    "aria": {"env": "aria_env", "min_free_mb": 20000},
    "sweet_rl": {"env": "sweet_rl", "min_free_mb": 26000},
}

GPU_FULL_ERROR_CODE = "server_gpu_full"
GPU_FULL_PUBLIC_MESSAGE = "The server is currently full. Please retry again in 5 minutes."
GPU_RETRY_AFTER_SECONDS = 5 * 60


class GPUCapacityError(RuntimeError):
    """No configured GPU currently has enough free memory for a worker."""

    def __init__(self) -> None:
        super().__init__(GPU_FULL_PUBLIC_MESSAGE)


class GPUAllocator:
    """Select a physical GPU in configured order using live free-memory data."""

    def __init__(self) -> None:
        configured = os.getenv("STUDY_GPU_ORDER", "0,1,2,3")
        try:
            self.gpu_order = tuple(
                int(value.strip()) for value in configured.split(",") if value.strip()
            )
        except ValueError as exc:
            raise RuntimeError("STUDY_GPU_ORDER must be a comma-separated list of GPU indices.") from exc
        if not self.gpu_order or len(set(self.gpu_order)) != len(self.gpu_order):
            raise RuntimeError("STUDY_GPU_ORDER must contain unique GPU indices.")
        self.minimum_free_mb = {
            runtime: int(
                os.getenv(
                    f"STUDY_GPU_MIN_FREE_MB_{runtime.upper()}",
                    str(defaults["min_free_mb"]),
                )
            )
            for runtime, defaults in RUNTIME_DEFAULTS.items()
        }
        if any(value < 1 for value in self.minimum_free_mb.values()):
            raise RuntimeError("All STUDY_GPU_MIN_FREE_MB_* values must be positive.")

    def free_memory_mb(self) -> dict[int, int]:
        executable = shutil.which("nvidia-smi")
        if not executable:
            raise RuntimeError("nvidia-smi was not found; GPU capacity cannot be inspected.")
        try:
            result = subprocess.run(
                [
                    executable,
                    "--query-gpu=index,memory.free",
                    "--format=csv,noheader,nounits",
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=15,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise RuntimeError(f"Could not inspect GPU memory with nvidia-smi: {exc}") from exc
        snapshot: dict[int, int] = {}
        for line in result.stdout.splitlines():
            fields = [field.strip() for field in line.split(",")]
            if len(fields) != 2:
                continue
            try:
                snapshot[int(fields[0])] = int(fields[1])
            except ValueError:
                continue
        if not snapshot:
            raise RuntimeError("nvidia-smi did not return GPU free-memory data.")
        return snapshot

    def candidates(self, runtime: str, excluded: set[int] | None = None) -> list[int]:
        excluded = excluded or set()
        free_memory = self.free_memory_mb()
        required = self.minimum_free_mb[runtime]
        return [
            gpu
            for gpu in self.gpu_order
            if gpu not in excluded and free_memory.get(gpu, 0) >= required
        ]


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

    def __init__(
        self,
        runtime: str,
        method: str,
        panel_id: str,
        server_dir: Path,
        gpu: int,
    ) -> None:
        self.runtime = runtime
        self.method = method
        self.panel_id = panel_id
        self.server_dir = server_dir
        upper = runtime.upper()
        defaults = RUNTIME_DEFAULTS[runtime]
        self.conda_env = os.getenv(f"STUDY_CONDA_ENV_{upper}", defaults["env"])
        self.gpu = int(gpu)
        self.process: subprocess.Popen[str] | None = None
        self.log_handle: Any = None
        self.log_path: Path | None = None
        self.log_start_offset = 0

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
        self.log_path = log_dir / f"model-{self.runtime}.log"
        self.log_start_offset = self.log_path.stat().st_size if self.log_path.exists() else 0
        self.log_handle = self.log_path.open("a", encoding="utf-8", buffering=1)
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
        environment["CUDA_VISIBLE_DEVICES"] = str(self.gpu)
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
        process = self.process
        request_id = str(uuid.uuid4())
        request = {"id": request_id, "method": method, "history": history}
        try:
            process.stdin.write(json.dumps(request, ensure_ascii=False) + "\n")
            process.stdin.flush()
            line = process.stdout.readline()
        except (BrokenPipeError, OSError) as exc:
            self.stop()
            raise RuntimeError(f"The {self.runtime} model worker stopped unexpectedly.") from exc
        if not line:
            return_code = process.poll()
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

    def diagnostic_tail(self, limit: int = 65536) -> str:
        if self.log_handle is not None:
            self.log_handle.flush()
        if self.log_path is None or not self.log_path.is_file():
            return ""
        with self.log_path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(self.log_start_offset, size - limit), os.SEEK_SET)
            return handle.read().decode("utf-8", errors="replace")


class InferenceManager:
    """Processes all browser requests through one FIFO generation lane."""

    def __init__(
        self,
        database_path: Path,
        server_dir: Path,
        mode: str = "real",
        gpu_allocator: GPUAllocator | None = None,
        idle_timeout_seconds: float | None = None,
    ) -> None:
        self.database_path = database_path
        self.server_dir = server_dir
        self.mode = mode
        self.gpu_allocator = gpu_allocator or GPUAllocator()
        self.idle_timeout_seconds = float(
            idle_timeout_seconds
            if idle_timeout_seconds is not None
            else os.getenv("STUDY_MODEL_IDLE_TIMEOUT_SECONDS", "60")
        )
        if self.idle_timeout_seconds <= 0:
            raise RuntimeError("STUDY_MODEL_IDLE_TIMEOUT_SECONDS must be positive.")
        # Workers are owned by panels, not by model type. This preserves TOPAS'
        # option state during one dialogue and lets us release the entire model
        # process as soon as that session ends.
        self.workers: dict[str, ModelWorker] = {}
        self._paused_panels: set[str] = set()
        self._last_activity: dict[str, float] = {}
        self._busy_panels: set[str] = set()
        self._workers_lock = threading.RLock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._reaper_thread: threading.Thread | None = None

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
        self._reaper_thread = threading.Thread(
            target=self._reap_inactive_workers,
            name="study-model-idle-reaper",
            daemon=True,
        )
        self._reaper_thread.start()
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
            self._paused_panels.clear()
            self._last_activity.clear()
            self._busy_panels.clear()
        for worker in workers:
            worker.stop()
        if self._thread is not None:
            self._thread.join(timeout=20)
            self._thread = None
        if self._reaper_thread is not None:
            self._reaper_thread.join(timeout=5)
            self._reaper_thread = None

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
            if panel_id in self._paused_panels:
                raise RuntimeError("This session was left before generation completed.")
            worker = self.workers.get(panel_id)
        if worker is not None:
            try:
                return worker.generate(method, history)
            except Exception as exc:
                diagnostic = worker.diagnostic_tail()
                if not self._is_gpu_memory_error(exc, diagnostic):
                    raise
                with self._workers_lock:
                    if self.workers.get(panel_id) is worker:
                        self.workers.pop(panel_id, None)
                        self._last_activity.pop(panel_id, None)
                worker.stop()
                LOGGER.warning(
                    "GPU %s ran out of memory for existing panel=%s method=%s; "
                    "trying the next GPU.",
                    worker.gpu,
                    panel_id,
                    method,
                )
                excluded: set[int] = {worker.gpu}
        else:
            excluded = set()

        while True:
            candidates = self.gpu_allocator.candidates(runtime, excluded)
            if not candidates:
                raise GPUCapacityError()
            gpu = candidates[0]
            worker = ModelWorker(runtime, method, panel_id, self.server_dir, gpu)
            with self._workers_lock:
                if panel_id in self._paused_panels:
                    raise RuntimeError("This session was left before generation completed.")
                existing = self.workers.get(panel_id)
                if existing is not None:
                    worker = existing
                else:
                    self.workers[panel_id] = worker
                    self._last_activity[panel_id] = time.monotonic()
            try:
                return worker.generate(method, history)
            except Exception as exc:
                with self._workers_lock:
                    if self.workers.get(panel_id) is worker:
                        self.workers.pop(panel_id, None)
                        self._last_activity.pop(panel_id, None)
                diagnostic = worker.diagnostic_tail()
                worker.stop()
                if not self._is_gpu_memory_error(exc, diagnostic):
                    raise
                LOGGER.warning(
                    "GPU %s ran out of memory for panel=%s method=%s; trying the next GPU.",
                    gpu,
                    panel_id,
                    method,
                )
                excluded.add(gpu)

    @staticmethod
    def _is_gpu_memory_error(exc: Exception, diagnostic: str = "") -> bool:
        text = f"{exc}\n{diagnostic}".casefold()
        return any(
            marker in text
            for marker in (
                "out of memory",
                "cuda error: memory allocation",
                "cublas_status_alloc_failed",
                "failed to allocate memory",
            )
        )

    def expire_inactive_workers(self, now: float | None = None) -> list[str]:
        """Unload non-generating workers idle for the configured interval."""
        now = time.monotonic() if now is None else now
        expired: list[tuple[str, ModelWorker]] = []
        with self._workers_lock:
            for panel_id, worker in list(self.workers.items()):
                last_activity = self._last_activity.get(panel_id, now)
                if (
                    panel_id not in self._busy_panels
                    and now - last_activity >= self.idle_timeout_seconds
                ):
                    self.workers.pop(panel_id, None)
                    self._last_activity.pop(panel_id, None)
                    expired.append((panel_id, worker))
        for panel_id, worker in expired:
            LOGGER.info(
                "Unloading idle panel=%s method=%s runtime=%s gpu=%s after %.0f seconds.",
                panel_id,
                worker.method,
                worker.runtime,
                worker.gpu,
                self.idle_timeout_seconds,
            )
            worker.stop()
        return [panel_id for panel_id, _worker in expired]

    def _reap_inactive_workers(self) -> None:
        interval = min(5.0, max(1.0, self.idle_timeout_seconds / 4))
        while not self._stop.wait(timeout=interval):
            self.expire_inactive_workers()

    def release_panel(self, panel_id: str) -> None:
        """Unload the model and policy state owned by a finished session."""
        with self._workers_lock:
            self._paused_panels.discard(panel_id)
            worker = self.workers.pop(panel_id, None)
            self._last_activity.pop(panel_id, None)
        if worker is not None:
            LOGGER.info(
                "Releasing panel=%s method=%s runtime=%s",
                panel_id,
                worker.method,
                worker.runtime,
            )
            worker.stop()

    def pause_panel(self, panel_id: str) -> None:
        """Unload an unfinished session and block a racing queued generation."""
        with self._workers_lock:
            self._paused_panels.add(panel_id)
            worker = self.workers.pop(panel_id, None)
            self._last_activity.pop(panel_id, None)
        if worker is not None:
            LOGGER.info(
                "Pausing panel=%s method=%s runtime=%s",
                panel_id,
                worker.method,
                worker.runtime,
            )
            worker.stop()

    def activate_panel(self, panel_id: str) -> None:
        """Allow a new or resumed session to load its worker on demand."""
        with self._workers_lock:
            self._paused_panels.discard(panel_id)
            if panel_id in self.workers:
                self._last_activity[panel_id] = time.monotonic()

    def _complete_job(self, job: sqlite3.Row, utterance: str) -> bool:
        now = utc_now()
        automatic_end = farewell_phrase(utterance) is not None
        with _database(self.database_path) as connection:
            connection.execute("BEGIN IMMEDIATE")
            current = connection.execute(
                "SELECT status FROM inference_jobs WHERE id = ?", (job["id"],)
            ).fetchone()
            # Leaving a session marks queued/running work as failed before the
            # worker is stopped. If generation wins that race, discard its
            # stale result instead of appending a reply after the expert left.
            if current is None or current["status"] != "running":
                connection.commit()
                return False
            connection.execute(
                "INSERT INTO panel_messages(panel_id, role, content, created_at) VALUES (?, 'therapist', ?, ?)",
                (job["panel_id"], utterance, now),
            )
            connection.execute(
                "UPDATE inference_jobs SET status = 'completed', completed_at = ?, error = NULL WHERE id = ?",
                (now, job["id"]),
            )
            if automatic_end:
                connection.execute(
                    """
                    UPDATE study_panels
                    SET ended_at = COALESCE(ended_at, ?),
                        termination_reason = COALESCE(termination_reason, 'therapist_farewell')
                    WHERE id = ?
                    """,
                    (now, job["panel_id"]),
                )
            connection.execute("UPDATE studies SET updated_at = ? WHERE id = ?", (now, job["study_id"]))
            connection.commit()
        return automatic_end

    def _fail_job(self, job: sqlite3.Row, exc: Exception) -> None:
        LOGGER.exception("Inference job %s failed for its hidden method.", job["id"], exc_info=exc)
        now = utc_now()
        stored_error = GPU_FULL_ERROR_CODE if isinstance(exc, GPUCapacityError) else str(exc)[:2000]
        with _database(self.database_path) as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                "UPDATE inference_jobs SET status = 'failed', completed_at = ?, error = ? WHERE id = ?",
                (now, stored_error, job["id"]),
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
                with self._workers_lock:
                    self._busy_panels.add(job["panel_id"])
                    if job["panel_id"] in self.workers:
                        self._last_activity[job["panel_id"]] = time.monotonic()
                utterance = self._generate(job["panel_id"], job["method_key"], history)
                automatic_end = self._complete_job(job, utterance)
                if automatic_end:
                    self.release_panel(job["panel_id"])
            except Exception as exc:  # A failed model must not stop later queued jobs.
                self._fail_job(job, exc)
                self.release_panel(job["panel_id"])
            finally:
                with self._workers_lock:
                    self._busy_panels.discard(job["panel_id"])
                    if job["panel_id"] in self.workers:
                        self._last_activity[job["panel_id"]] = time.monotonic()

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
            gpu_assignments = {
                str(worker_gpu): gpu_assignments_count
                for worker_gpu in sorted({worker.gpu for worker in self.workers.values()})
                if (
                    gpu_assignments_count := sum(
                        worker.running and worker.gpu == worker_gpu
                        for worker in self.workers.values()
                    )
                )
            }
        return {
            "mode": self.mode,
            "queue_depth": int(queued),
            "busy": bool(running),
            "loaded_workers": loaded_workers,
            "gpu_assignments": gpu_assignments,
            "idle_timeout_seconds": self.idle_timeout_seconds,
        }
