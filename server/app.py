from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import json
import logging
import os
import random
import re
import secrets
import sqlite3
import threading
import time
import uuid
from contextlib import asynccontextmanager, contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from fastapi import Depends, FastAPI, Header, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

if __package__:
    from .inference_manager import InferenceManager
    from .profiles import load_profiles, profile_card, public_profile
else:
    from inference_manager import InferenceManager
    from profiles import load_profiles, profile_card, public_profile


LOGGER = logging.getLogger("cbt_live_interaction")
SERVER_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SERVER_DIR.parent
FRONTEND_DIR = PROJECT_DIR / "frontend"
DEFAULT_DB_PATH = SERVER_DIR / "data" / "study.sqlite3"


def _discover_topas_root() -> Path:
    candidates = (SERVER_DIR.parent, *SERVER_DIR.parents)
    return next(
        (candidate for candidate in candidates if (candidate / "simulations" / "agents.py").is_file()),
        SERVER_DIR.parent,
    )


DEFAULT_TOPAS_ROOT = _discover_topas_root()

ACCESS_CODE = os.getenv("STUDY_ACCESS_CODE", "").strip()
TOKEN_SECRET = os.getenv("STUDY_TOKEN_SECRET", "").strip()
TOKEN_TTL_SECONDS = int(os.getenv("STUDY_TOKEN_TTL_SECONDS", str(12 * 60 * 60)))
DATABASE_PATH = Path(os.getenv("STUDY_DB_PATH", str(DEFAULT_DB_PATH))).expanduser().resolve()
TOPAS_PROJECT_ROOT = Path(os.getenv("TOPAS_PROJECT_ROOT", str(DEFAULT_TOPAS_ROOT))).expanduser().resolve()
INFERENCE_MODE = os.getenv("STUDY_INFERENCE_MODE", "real").strip().lower()
ALLOWED_ORIGINS = [
    origin.strip().rstrip("/")
    for origin in os.getenv("STUDY_ALLOWED_ORIGINS", "").split(",")
    if origin.strip()
]
SERVE_FRONTEND = os.getenv("STUDY_SERVE_FRONTEND", "false").strip().lower() in {"1", "true", "yes", "on"}

PARTICIPANT_CODE_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{1,63}$")
DATABASE_LOCK = threading.RLock()
METHOD_KEYS = ["prompting", "proact", "archer", "aria", "sweet_rl", "topas"]
PANEL_LABELS = ["Therapist A", "Therapist B", "Therapist C", "Therapist D", "Therapist E", "Therapist F"]
PROFILES = load_profiles()


class LoginRequest(BaseModel):
    participant_code: str
    access_code: str


class StudyRequest(BaseModel):
    profile_id: str


class MessageRequest(BaseModel):
    content: str
    client_message_id: str


class ActionRequest(BaseModel):
    client_request_id: str


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _urlsafe_encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).decode("ascii").rstrip("=")


def _urlsafe_decode(value: str) -> bytes:
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))


def create_token(participant_code: str) -> str:
    payload = {
        "participant_code": participant_code,
        "expires_at": int(time.time()) + TOKEN_TTL_SECONDS,
        "nonce": secrets.token_urlsafe(8),
    }
    encoded = _urlsafe_encode(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
    signature = hmac.new(TOKEN_SECRET.encode("utf-8"), encoded.encode("ascii"), hashlib.sha256).digest()
    return f"{encoded}.{_urlsafe_encode(signature)}"


def decode_token(token: str) -> str:
    try:
        encoded, supplied = token.split(".", maxsplit=1)
        expected = hmac.new(TOKEN_SECRET.encode("utf-8"), encoded.encode("ascii"), hashlib.sha256).digest()
        if not hmac.compare_digest(expected, _urlsafe_decode(supplied)):
            raise ValueError("invalid signature")
        payload = json.loads(_urlsafe_decode(encoded))
        if int(payload["expires_at"]) < int(time.time()):
            raise ValueError("expired token")
        participant_code = str(payload["participant_code"])
        if not PARTICIPANT_CODE_PATTERN.fullmatch(participant_code):
            raise ValueError("invalid participant")
        return participant_code
    except (binascii.Error, KeyError, TypeError, UnicodeDecodeError, ValueError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=401, detail="Your session has expired. Please sign in again.") from exc


def require_participant(authorization: str | None = Header(default=None)) -> str:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Authentication is required.")
    return decode_token(authorization.removeprefix("Bearer ").strip())


@contextmanager
def database() -> Iterator[sqlite3.Connection]:
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DATABASE_PATH, timeout=30, isolation_level=None)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA busy_timeout = 30000")
    try:
        yield connection
    finally:
        connection.close()


def initialize_database() -> None:
    with DATABASE_LOCK, database() as connection:
        connection.executescript(
            """
            PRAGMA journal_mode = WAL;
            PRAGMA synchronous = NORMAL;

            CREATE TABLE IF NOT EXISTS participants (
                participant_code TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                last_seen_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS studies (
                id TEXT PRIMARY KEY,
                participant_code TEXT NOT NULL,
                profile_id TEXT NOT NULL,
                status TEXT NOT NULL CHECK (status IN ('active', 'finished')),
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                finished_at TEXT,
                FOREIGN KEY (participant_code) REFERENCES participants(participant_code),
                UNIQUE (participant_code, profile_id)
            );

            CREATE TABLE IF NOT EXISTS study_panels (
                id TEXT PRIMARY KEY,
                study_id TEXT NOT NULL,
                label TEXT NOT NULL,
                method_key TEXT NOT NULL,
                display_order INTEGER NOT NULL,
                FOREIGN KEY (study_id) REFERENCES studies(id) ON DELETE CASCADE,
                UNIQUE (study_id, label),
                UNIQUE (study_id, method_key)
            );

            CREATE TABLE IF NOT EXISTS panel_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                panel_id TEXT NOT NULL,
                role TEXT NOT NULL CHECK (role IN ('patient', 'therapist')),
                content TEXT NOT NULL,
                client_message_id TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (panel_id) REFERENCES study_panels(id) ON DELETE CASCADE
            );

            CREATE UNIQUE INDEX IF NOT EXISTS panel_messages_client_id
            ON panel_messages(panel_id, client_message_id)
            WHERE client_message_id IS NOT NULL;

            CREATE TABLE IF NOT EXISTS inference_jobs (
                sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                id TEXT NOT NULL UNIQUE,
                panel_id TEXT NOT NULL,
                kind TEXT NOT NULL CHECK (kind IN ('start', 'message', 'retry')),
                client_request_id TEXT NOT NULL,
                status TEXT NOT NULL CHECK (status IN ('queued', 'running', 'completed', 'failed')),
                error TEXT,
                created_at TEXT NOT NULL,
                started_at TEXT,
                completed_at TEXT,
                FOREIGN KEY (panel_id) REFERENCES study_panels(id) ON DELETE CASCADE,
                UNIQUE (panel_id, client_request_id)
            );
            """
        )


def get_study(connection: sqlite3.Connection, study_id: str, participant_code: str) -> sqlite3.Row:
    study = connection.execute(
        "SELECT * FROM studies WHERE id = ? AND participant_code = ?",
        (study_id, participant_code),
    ).fetchone()
    if study is None:
        raise HTTPException(status_code=404, detail="Study not found.")
    return study


def get_panel(
    connection: sqlite3.Connection, study_id: str, panel_id: str, participant_code: str
) -> tuple[sqlite3.Row, sqlite3.Row]:
    study = get_study(connection, study_id, participant_code)
    panel = connection.execute(
        "SELECT * FROM study_panels WHERE id = ? AND study_id = ?",
        (panel_id, study_id),
    ).fetchone()
    if panel is None:
        raise HTTPException(status_code=404, detail="Conversation not found.")
    return study, panel


def _latest_job(connection: sqlite3.Connection, panel_id: str) -> sqlite3.Row | None:
    return connection.execute(
        "SELECT * FROM inference_jobs WHERE panel_id = ? ORDER BY sequence DESC LIMIT 1",
        (panel_id,),
    ).fetchone()


def _queue_position(connection: sqlite3.Connection, job: sqlite3.Row) -> int | None:
    if job["status"] == "running":
        return 0
    if job["status"] != "queued":
        return None
    ahead = connection.execute(
        "SELECT COUNT(*) FROM inference_jobs WHERE status IN ('running', 'queued') AND sequence < ?",
        (job["sequence"],),
    ).fetchone()[0]
    return int(ahead) + 1


def serialize_job(connection: sqlite3.Connection, job: sqlite3.Row | None) -> dict | None:
    if job is None:
        return None
    return {
        "id": job["id"],
        "status": job["status"],
        "queue_position": _queue_position(connection, job),
        "created_at": job["created_at"],
        "started_at": job["started_at"],
        "completed_at": job["completed_at"],
        "can_retry": job["status"] == "failed",
    }


def serialize_study(connection: sqlite3.Connection, study: sqlite3.Row) -> dict:
    profile = PROFILES[study["profile_id"]]
    panels = []
    for panel in connection.execute(
        "SELECT * FROM study_panels WHERE study_id = ? ORDER BY display_order",
        (study["id"],),
    ).fetchall():
        messages = [
            dict(row)
            for row in connection.execute(
                "SELECT id, role, content, created_at FROM panel_messages WHERE panel_id = ? ORDER BY id",
                (panel["id"],),
            ).fetchall()
        ]
        job = _latest_job(connection, panel["id"])
        pending = job is not None and job["status"] in {"queued", "running"}
        panels.append(
            {
                "id": panel["id"],
                "label": panel["label"],
                "messages": messages,
                "job": serialize_job(connection, job),
                "can_start": study["status"] == "active" and not messages and not pending,
                "can_send": (
                    study["status"] == "active"
                    and bool(messages)
                    and messages[-1]["role"] == "therapist"
                    and not pending
                ),
            }
        )
    return {
        "id": study["id"],
        "status": study["status"],
        "profile": public_profile(profile),
        "panels": panels,
        "created_at": study["created_at"],
        "updated_at": study["updated_at"],
        "finished_at": study["finished_at"],
    }


def _method_order(participant_code: str, profile_id: str) -> list[str]:
    digest = hmac.new(
        TOKEN_SECRET.encode("utf-8"),
        f"{participant_code}|{profile_id}|panel-order".encode("utf-8"),
        hashlib.sha256,
    ).digest()
    methods = METHOD_KEYS.copy()
    random.Random(int.from_bytes(digest, "big")).shuffle(methods)
    return methods


def _validate_request_id(value: str) -> str:
    value = value.strip()
    if not value or len(value) > 100:
        raise HTTPException(status_code=422, detail="Invalid request identifier.")
    return value


def _enqueue_job(
    connection: sqlite3.Connection, panel_id: str, kind: str, client_request_id: str
) -> sqlite3.Row:
    existing = connection.execute(
        "SELECT * FROM inference_jobs WHERE panel_id = ? AND client_request_id = ?",
        (panel_id, client_request_id),
    ).fetchone()
    if existing is not None:
        return existing
    job_id = str(uuid.uuid4())
    connection.execute(
        """
        INSERT INTO inference_jobs(id, panel_id, kind, client_request_id, status, created_at)
        VALUES (?, ?, ?, ?, 'queued', ?)
        """,
        (job_id, panel_id, kind, client_request_id, utc_now()),
    )
    return connection.execute("SELECT * FROM inference_jobs WHERE id = ?", (job_id,)).fetchone()


INFERENCE_MANAGER = InferenceManager(
    database_path=DATABASE_PATH,
    server_dir=SERVER_DIR,
    project_root=TOPAS_PROJECT_ROOT,
    mode=INFERENCE_MODE,
)


@asynccontextmanager
async def lifespan(_: FastAPI):
    initialize_database()
    if not ACCESS_CODE:
        raise RuntimeError("STUDY_ACCESS_CODE is required.")
    if len(TOKEN_SECRET) < 32:
        raise RuntimeError("STUDY_TOKEN_SECRET is required and must contain at least 32 characters.")
    if INFERENCE_MODE not in {"real", "static"}:
        raise RuntimeError("STUDY_INFERENCE_MODE must be 'real' or 'static'.")
    INFERENCE_MANAGER.start()
    try:
        yield
    finally:
        INFERENCE_MANAGER.stop()


app = FastAPI(
    title="CBT Live Interaction Study",
    version="1.0.0",
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
    lifespan=lifespan,
)

if ALLOWED_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=ALLOWED_ORIGINS,
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type", "ngrok-skip-browser-warning"],
    )


@app.middleware("http")
async def security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    if request.url.path.startswith("/api/"):
        response.headers["Cache-Control"] = "no-store"
    return response


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "inference": INFERENCE_MANAGER.health()}


@app.post("/api/auth/login")
def login(payload: LoginRequest) -> dict:
    participant_code = payload.participant_code.strip()
    if not PARTICIPANT_CODE_PATTERN.fullmatch(participant_code):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Participant code must contain 2-64 letters, numbers, dots, underscores, or hyphens.",
        )
    if not hmac.compare_digest(payload.access_code, ACCESS_CODE):
        raise HTTPException(status_code=401, detail="The access code is incorrect.")
    now = utc_now()
    with DATABASE_LOCK, database() as connection:
        connection.execute("BEGIN IMMEDIATE")
        connection.execute(
            """
            INSERT INTO participants(participant_code, created_at, last_seen_at)
            VALUES (?, ?, ?)
            ON CONFLICT(participant_code) DO UPDATE SET last_seen_at = excluded.last_seen_at
            """,
            (participant_code, now, now),
        )
        connection.commit()
    return {"token": create_token(participant_code), "participant_code": participant_code, "expires_in": TOKEN_TTL_SECONDS}


@app.get("/api/profiles")
def list_profiles(participant_code: str = Depends(require_participant)) -> dict:
    with database() as connection:
        existing = {
            row["profile_id"]: {"id": row["id"], "status": row["status"]}
            for row in connection.execute(
                "SELECT id, profile_id, status FROM studies WHERE participant_code = ?",
                (participant_code,),
            ).fetchall()
        }
    return {"profiles": [profile_card(profile, existing.get(public_id)) for public_id, profile in PROFILES.items()]}


@app.post("/api/studies")
def create_study(payload: StudyRequest, participant_code: str = Depends(require_participant)) -> dict:
    profile_id = payload.profile_id.strip()
    if profile_id not in PROFILES:
        raise HTTPException(status_code=404, detail="Patient profile not found.")
    now = utc_now()
    with DATABASE_LOCK, database() as connection:
        connection.execute("BEGIN IMMEDIATE")
        existing = connection.execute(
            "SELECT * FROM studies WHERE participant_code = ? AND profile_id = ?",
            (participant_code, profile_id),
        ).fetchone()
        if existing is None:
            study_id = str(uuid.uuid4())
            connection.execute(
                """
                INSERT INTO studies(id, participant_code, profile_id, status, created_at, updated_at)
                VALUES (?, ?, ?, 'active', ?, ?)
                """,
                (study_id, participant_code, profile_id, now, now),
            )
            for index, (label, method) in enumerate(zip(PANEL_LABELS, _method_order(participant_code, profile_id))):
                connection.execute(
                    """
                    INSERT INTO study_panels(id, study_id, label, method_key, display_order)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (str(uuid.uuid4()), study_id, label, method, index),
                )
            existing = connection.execute("SELECT * FROM studies WHERE id = ?", (study_id,)).fetchone()
        connection.commit()
        return {"study": serialize_study(connection, existing)}


@app.get("/api/studies/{study_id}")
def read_study(study_id: str, participant_code: str = Depends(require_participant)) -> dict:
    with database() as connection:
        study = get_study(connection, study_id, participant_code)
        return {"study": serialize_study(connection, study)}


@app.post("/api/studies/{study_id}/panels/{panel_id}/start")
def start_panel(
    study_id: str,
    panel_id: str,
    payload: ActionRequest,
    participant_code: str = Depends(require_participant),
) -> dict:
    request_id = _validate_request_id(payload.client_request_id)
    with DATABASE_LOCK, database() as connection:
        connection.execute("BEGIN IMMEDIATE")
        study, panel = get_panel(connection, study_id, panel_id, participant_code)
        if study["status"] != "active":
            connection.rollback()
            raise HTTPException(status_code=409, detail="This patient study is finished.")
        duplicate = connection.execute(
            "SELECT * FROM inference_jobs WHERE panel_id = ? AND client_request_id = ?",
            (panel_id, request_id),
        ).fetchone()
        if duplicate is not None:
            connection.commit()
            return {"job": serialize_job(connection, duplicate)}
        if connection.execute("SELECT 1 FROM panel_messages WHERE panel_id = ?", (panel_id,)).fetchone():
            connection.rollback()
            raise HTTPException(status_code=409, detail="This conversation has already started.")
        if connection.execute(
            "SELECT 1 FROM inference_jobs WHERE panel_id = ? AND status IN ('queued', 'running')", (panel_id,)
        ).fetchone():
            connection.rollback()
            raise HTTPException(status_code=409, detail="A response is already queued for this conversation.")
        job = _enqueue_job(connection, panel["id"], "start", request_id)
        connection.commit()
        result = serialize_job(connection, job)
    INFERENCE_MANAGER.notify()
    return {"job": result}


@app.post("/api/studies/{study_id}/panels/{panel_id}/messages")
def send_message(
    study_id: str,
    panel_id: str,
    payload: MessageRequest,
    participant_code: str = Depends(require_participant),
) -> dict:
    content = payload.content.strip()
    request_id = _validate_request_id(payload.client_message_id)
    if not content:
        raise HTTPException(status_code=422, detail="Please enter a message.")
    if len(content) > 4000:
        raise HTTPException(status_code=422, detail="Messages cannot exceed 4,000 characters.")
    with DATABASE_LOCK, database() as connection:
        connection.execute("BEGIN IMMEDIATE")
        study, panel = get_panel(connection, study_id, panel_id, participant_code)
        duplicate = connection.execute(
            "SELECT * FROM inference_jobs WHERE panel_id = ? AND client_request_id = ?",
            (panel_id, request_id),
        ).fetchone()
        if duplicate is not None:
            connection.commit()
            return {"job": serialize_job(connection, duplicate)}
        if study["status"] != "active":
            connection.rollback()
            raise HTTPException(status_code=409, detail="This patient study is finished.")
        if connection.execute(
            "SELECT 1 FROM inference_jobs WHERE panel_id = ? AND status IN ('queued', 'running')", (panel_id,)
        ).fetchone():
            connection.rollback()
            raise HTTPException(status_code=409, detail="Wait for the pending therapist response.")
        latest = connection.execute(
            "SELECT role FROM panel_messages WHERE panel_id = ? ORDER BY id DESC LIMIT 1", (panel_id,)
        ).fetchone()
        if latest is None or latest["role"] != "therapist":
            connection.rollback()
            raise HTTPException(status_code=409, detail="Start the conversation or wait for the therapist response.")
        now = utc_now()
        connection.execute(
            """
            INSERT INTO panel_messages(panel_id, role, content, client_message_id, created_at)
            VALUES (?, 'patient', ?, ?, ?)
            """,
            (panel_id, content, request_id, now),
        )
        job = _enqueue_job(connection, panel["id"], "message", request_id)
        connection.execute("UPDATE studies SET updated_at = ? WHERE id = ?", (now, study_id))
        connection.commit()
        result = serialize_job(connection, job)
    INFERENCE_MANAGER.notify()
    return {"job": result}


@app.post("/api/studies/{study_id}/panels/{panel_id}/retry")
def retry_panel(
    study_id: str,
    panel_id: str,
    payload: ActionRequest,
    participant_code: str = Depends(require_participant),
) -> dict:
    request_id = _validate_request_id(payload.client_request_id)
    with DATABASE_LOCK, database() as connection:
        connection.execute("BEGIN IMMEDIATE")
        study, panel = get_panel(connection, study_id, panel_id, participant_code)
        if study["status"] != "active":
            connection.rollback()
            raise HTTPException(status_code=409, detail="This patient study is finished.")
        latest_job = _latest_job(connection, panel_id)
        if latest_job is None or latest_job["status"] != "failed":
            connection.rollback()
            raise HTTPException(status_code=409, detail="There is no failed response to retry.")
        job = _enqueue_job(connection, panel["id"], "retry", request_id)
        connection.commit()
        result = serialize_job(connection, job)
    INFERENCE_MANAGER.notify()
    return {"job": result}


@app.post("/api/studies/{study_id}/finish")
def finish_study(study_id: str, participant_code: str = Depends(require_participant)) -> dict:
    now = utc_now()
    with DATABASE_LOCK, database() as connection:
        connection.execute("BEGIN IMMEDIATE")
        study = get_study(connection, study_id, participant_code)
        pending = connection.execute(
            """
            SELECT 1 FROM inference_jobs AS j
            JOIN study_panels AS p ON p.id = j.panel_id
            WHERE p.study_id = ? AND j.status IN ('queued', 'running')
            LIMIT 1
            """,
            (study_id,),
        ).fetchone()
        if pending:
            connection.rollback()
            raise HTTPException(status_code=409, detail="Wait for queued therapist responses before finishing.")
        if study["status"] == "active":
            connection.execute(
                "UPDATE studies SET status = 'finished', finished_at = ?, updated_at = ? WHERE id = ?",
                (now, now, study_id),
            )
        connection.commit()
        study = get_study(connection, study_id, participant_code)
        result = serialize_study(connection, study)
    return {"study": result}


if SERVE_FRONTEND:
    if not FRONTEND_DIR.is_dir():
        raise RuntimeError(f"Frontend directory is missing: {FRONTEND_DIR}")

    @app.get("/config.js", include_in_schema=False)
    def frontend_config() -> FileResponse:
        return FileResponse(FRONTEND_DIR / "config.js", media_type="application/javascript")

    app.mount("/assets", StaticFiles(directory=FRONTEND_DIR), name="assets")

    @app.get("/{path:path}", include_in_schema=False)
    def frontend(path: str) -> FileResponse:
        if path.startswith("api/"):
            raise HTTPException(status_code=404, detail="API endpoint not found.")
        requested = (FRONTEND_DIR / path).resolve()
        if path and requested.is_file() and FRONTEND_DIR.resolve() in requested.parents:
            return FileResponse(requested)
        return FileResponse(FRONTEND_DIR / "index.html")
else:
    @app.get("/", include_in_schema=False)
    def service_root() -> dict:
        return {"service": "CBT Live Interaction API", "health": "/api/health"}
