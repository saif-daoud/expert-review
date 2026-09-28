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

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, StrictInt

if __package__:
    from .inference_manager import InferenceManager
    from .profiles import load_profiles, profile_card, public_profile
    from .session_rules import farewell_phrase
else:
    from inference_manager import InferenceManager
    from profiles import load_profiles, profile_card, public_profile
    from session_rules import farewell_phrase


LOGGER = logging.getLogger("cbt_live_interaction")
SERVER_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SERVER_DIR.parent
FRONTEND_DIR = PROJECT_DIR / "frontend"
DEFAULT_DB_PATH = SERVER_DIR / "data" / "study.sqlite3"
LEGACY_ACCESS_CODE = os.getenv("STUDY_ACCESS_CODE", "").strip()
TOKEN_SECRET = os.getenv("STUDY_TOKEN_SECRET", "").strip()
TOKEN_TTL_SECONDS = int(os.getenv("STUDY_TOKEN_TTL_SECONDS", str(12 * 60 * 60)))
DATABASE_PATH = Path(os.getenv("STUDY_DB_PATH", str(DEFAULT_DB_PATH))).expanduser().resolve()
INFERENCE_MODE = os.getenv("STUDY_INFERENCE_MODE", "real").strip().lower()
MAX_SESSION_TURNS = int(os.getenv("STUDY_MAX_SESSION_TURNS", "50"))
ALLOWED_ORIGINS = [
    origin.strip().rstrip("/")
    for origin in os.getenv("STUDY_ALLOWED_ORIGINS", "").split(",")
    if origin.strip()
]
SERVE_FRONTEND = os.getenv("STUDY_SERVE_FRONTEND", "false").strip().lower() in {"1", "true", "yes", "on"}

PARTICIPANT_CODE_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{1,63}$")
EMAIL_PATTERN = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]+$")
DATABASE_LOCK = threading.RLock()
METHOD_KEYS = ["prompting", "proact", "archer", "aria", "sweet_rl", "topas"]
PANEL_LABELS = ["Therapist A", "Therapist B", "Therapist C", "Therapist D", "Therapist E", "Therapist F"]
CTRS_KEYS = (
    "agenda",
    "feedback",
    "understanding",
    "interpersonal_effectiveness",
    "collaboration",
    "pacing_time_use",
    "guided_discovery",
    "focusing_on_key_cognitions_behaviors",
    "strategy_for_change",
    "application_of_cbt_techniques",
    "homework",
)
PROFILES = load_profiles()
EXPERT_CODES = (
    os.getenv("STUDY_EXPERT_1_CODE", "EXPERT-5834").strip().upper(),
    os.getenv("STUDY_EXPERT_2_CODE", "EXPERT-9271").strip().upper(),
)
EXPERT_ACCESS_CODES = {
    EXPERT_CODES[0]: os.getenv("STUDY_EXPERT_1_ACCESS_CODE", LEGACY_ACCESS_CODE).strip(),
    EXPERT_CODES[1]: os.getenv("STUDY_EXPERT_2_ACCESS_CODE", "").strip(),
}
LEGACY_PARTICIPANT_PROFILE_IDS = {
    code: tuple(profile_id for profile_id, profile in PROFILES.items() if profile["assignment_group"] == group)
    for group, code in enumerate(EXPERT_CODES, start=1)
}
_ORDERED_PROFILE_IDS = tuple(
    sorted(PROFILES, key=lambda profile_id: int(profile_id.removeprefix("patient-")))
)
FIRST_SHARED_PROFILE_IDS = _ORDERED_PROFILE_IDS[:30]
SECOND_EXPERT_PROFILE_IDS = _ORDERED_PROFILE_IDS[30:]
LEENA_PROFILE_ID = os.getenv("STUDY_LEENA_PROFILE_ID", "patient-39").strip()
if len(FIRST_SHARED_PROFILE_IDS) != 30 or len(set(FIRST_SHARED_PROFILE_IDS)) != 30:
    raise RuntimeError("The first shared code must have 30 unique patient profiles.")
if len(SECOND_EXPERT_PROFILE_IDS) != 10 or len(set(SECOND_EXPERT_PROFILE_IDS)) != 10:
    raise RuntimeError("The second expert code must have 10 unique patient profiles.")
if set(FIRST_SHARED_PROFILE_IDS) & set(SECOND_EXPERT_PROFILE_IDS):
    raise RuntimeError("The two registration cohorts must use disjoint patient profiles.")
if LEENA_PROFILE_ID not in PROFILES:
    raise RuntimeError("STUDY_LEENA_PROFILE_ID must reference a configured patient profile.")


class LoginRequest(BaseModel):
    email: str
    access_code: str


class ParticipantProfileRequest(BaseModel):
    name: str
    role: str
    institution: str
    latest_degree: str
    years_experience: StrictInt


class StudyRequest(BaseModel):
    profile_id: str


class MessageRequest(BaseModel):
    content: str
    client_message_id: str


class ActionRequest(BaseModel):
    client_request_id: str


class RatingRequest(BaseModel):
    scores: dict[str, StrictInt]
    comments: str = ""


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


def require_registered_participant(authorization: str | None = Header(default=None)) -> str:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Authentication is required.")
    participant_code = decode_token(authorization.removeprefix("Bearer ").strip())
    with database() as connection:
        participant = connection.execute(
            "SELECT participant_code FROM participants WHERE participant_code = ?",
            (participant_code,),
        ).fetchone()
    if participant is None:
        raise HTTPException(status_code=401, detail="Your session has expired. Please sign in again.")
    return participant_code


def require_participant(
    participant_code: str = Depends(require_registered_participant),
) -> str:
    with database() as connection:
        participant = connection.execute(
            "SELECT profile_completed FROM participants WHERE participant_code = ?",
            (participant_code,),
        ).fetchone()
    if participant is None or int(participant["profile_completed"] or 0) != 1:
        raise HTTPException(status_code=403, detail="Complete your professional profile first.")
    return participant_code


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


def normalize_email(value: str) -> str:
    return value.strip().lower()[:254]


def participant_code_for_email(email: str) -> str:
    return f"P-{hashlib.sha256(email.encode('utf-8')).hexdigest()[:32]}"


def matches_leena(*values: str | None) -> bool:
    return any("leena" in str(value or "").casefold() for value in values)


def registration_profile_ids(cohort_code: str) -> tuple[str, ...]:
    if cohort_code == EXPERT_CODES[0]:
        return FIRST_SHARED_PROFILE_IDS
    if cohort_code == EXPERT_CODES[1]:
        return SECOND_EXPERT_PROFILE_IDS
    return ()


def cohort_for_access_code(access_code: str) -> str:
    for cohort_code, configured_code in EXPERT_ACCESS_CODES.items():
        if configured_code and hmac.compare_digest(access_code, configured_code):
            return cohort_code
    return ""


def assigned_profile_ids(connection: sqlite3.Connection, participant_code: str) -> tuple[str, ...]:
    participant = connection.execute(
        "SELECT cohort_code, assigned_profiles_json FROM participants WHERE participant_code = ?",
        (participant_code,),
    ).fetchone()
    if participant is None:
        return ()
    try:
        configured = tuple(json.loads(participant["assigned_profiles_json"] or "[]"))
    except (TypeError, json.JSONDecodeError):
        configured = ()
    valid = tuple(profile_id for profile_id in configured if profile_id in PROFILES)
    if valid:
        return valid
    return LEGACY_PARTICIPANT_PROFILE_IDS.get(str(participant["cohort_code"] or "").upper(), ())


def assign_leena_patient(connection: sqlite3.Connection, participant_code: str) -> None:
    participant = connection.execute(
        "SELECT cohort_code, assigned_profiles_json FROM participants WHERE participant_code = ?",
        (participant_code,),
    ).fetchone()
    if participant is None:
        raise HTTPException(status_code=401, detail="The registered expert account was not found.")

    cohort_code = str(participant["cohort_code"] or "").upper()
    if cohort_code == EXPERT_CODES[0]:
        for row in connection.execute(
            """
            SELECT participant_code, assigned_profiles_json FROM participants
            WHERE cohort_code = ? AND email IS NOT NULL AND email != '' AND participant_code != ?
            """,
            (cohort_code, participant_code),
        ).fetchall():
            try:
                other_profiles = json.loads(row["assigned_profiles_json"] or "[]")
            except (TypeError, json.JSONDecodeError):
                other_profiles = []
            if LEENA_PROFILE_ID in other_profiles:
                raise HTTPException(
                    status_code=409,
                    detail="The reserved patient is already assigned to another email address.",
                )
        connection.execute(
            "UPDATE participants SET assigned_profiles_json = ? WHERE participant_code = ?",
            (json.dumps((LEENA_PROFILE_ID,)), participant_code),
        )

    legacy_studies = connection.execute(
        "SELECT id FROM studies WHERE participant_code = ? AND profile_id = ?",
        (EXPERT_CODES[0], LEENA_PROFILE_ID),
    ).fetchall()
    if not legacy_studies:
        claimed = connection.execute(
            """
            SELECT participant_code FROM studies
            WHERE profile_id = ? AND participant_code NOT IN (?, ?)
            LIMIT 1
            """,
            (LEENA_PROFILE_ID, participant_code, EXPERT_CODES[1]),
        ).fetchone()
        if claimed is not None:
            raise HTTPException(
                status_code=409,
                detail="The reserved patient progress is already linked to another email address.",
            )
        return

    existing = connection.execute(
        "SELECT id FROM studies WHERE participant_code = ? AND profile_id = ? LIMIT 1",
        (participant_code, LEENA_PROFILE_ID),
    ).fetchone()
    if existing is not None:
        raise HTTPException(
            status_code=409,
            detail="This account already has a separate study for the reserved patient.",
        )

    study_ids = tuple(row["id"] for row in legacy_studies)
    placeholders = ",".join("?" for _ in study_ids)
    connection.execute(
        f"""
        UPDATE panel_ratings SET participant_code = ?
        WHERE panel_id IN (
            SELECT id FROM study_panels WHERE study_id IN ({placeholders})
        )
        """,
        (participant_code, *study_ids),
    )
    connection.execute(
        f"UPDATE studies SET participant_code = ? WHERE id IN ({placeholders})",
        (participant_code, *study_ids),
    )


def participant_profile_payload(participant: sqlite3.Row) -> dict:
    return {
        "email": participant["email"],
        "name": participant["name"],
        "role": participant["role"],
        "institution": participant["institution"],
        "latest_degree": participant["latest_degree"],
        "years_experience": participant["years_experience"],
    }


def validate_participant_profile(payload: ParticipantProfileRequest) -> dict:
    values = {
        "name": payload.name.strip()[:160],
        "role": payload.role.strip()[:200],
        "institution": payload.institution.strip()[:260],
        "latest_degree": payload.latest_degree.strip()[:160],
        "years_experience": int(payload.years_experience),
    }
    if not all(values[key] for key in ("name", "role", "institution", "latest_degree")):
        raise HTTPException(status_code=422, detail="Complete every professional profile field.")
    if values["years_experience"] < 0 or values["years_experience"] > 80:
        raise HTTPException(
            status_code=422,
            detail="Years of clinical experience must be between 0 and 80.",
        )
    return values


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
                ended_at TEXT,
                termination_reason TEXT,
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

            CREATE TABLE IF NOT EXISTS panel_ratings (
                id TEXT PRIMARY KEY,
                panel_id TEXT NOT NULL UNIQUE,
                participant_code TEXT NOT NULL,
                scores_json TEXT NOT NULL,
                total_score INTEGER NOT NULL CHECK (total_score BETWEEN 0 AND 66),
                comments TEXT NOT NULL DEFAULT '',
                submitted_at TEXT NOT NULL,
                FOREIGN KEY (panel_id) REFERENCES study_panels(id) ON DELETE CASCADE,
                FOREIGN KEY (participant_code) REFERENCES participants(participant_code)
            );
            """
        )
        participant_columns = {
            row["name"] for row in connection.execute("PRAGMA table_info(participants)").fetchall()
        }
        participant_migrations = {
            "email": "TEXT",
            "cohort_code": "TEXT",
            "assigned_profiles_json": "TEXT",
            "name": "TEXT",
            "role": "TEXT",
            "institution": "TEXT",
            "latest_degree": "TEXT",
            "years_experience": "INTEGER",
            "profile_completed": "INTEGER NOT NULL DEFAULT 0",
        }
        for column, declaration in participant_migrations.items():
            if column not in participant_columns:
                connection.execute(f"ALTER TABLE participants ADD COLUMN {column} {declaration}")
        connection.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS participants_email_unique
            ON participants(lower(email))
            WHERE email IS NOT NULL AND email != ''
            """
        )
        for legacy_code, profile_ids in LEGACY_PARTICIPANT_PROFILE_IDS.items():
            connection.execute(
                """
                UPDATE participants
                SET cohort_code = COALESCE(cohort_code, ?),
                    assigned_profiles_json = COALESCE(assigned_profiles_json, ?)
                WHERE participant_code = ?
                """,
                (legacy_code, json.dumps(profile_ids), legacy_code),
            )
        panel_columns = {
            row["name"] for row in connection.execute("PRAGMA table_info(study_panels)").fetchall()
        }
        if "ended_at" not in panel_columns:
            connection.execute("ALTER TABLE study_panels ADD COLUMN ended_at TEXT")
        if "termination_reason" not in panel_columns:
            connection.execute("ALTER TABLE study_panels ADD COLUMN termination_reason TEXT")
        # Studies completed by the earlier parallel-chat UI have no CTRS
        # ratings. Reopen them so the new sequential evaluation can resume.
        connection.execute(
            """
            UPDATE studies
            SET status = 'active', finished_at = NULL, updated_at = ?
            WHERE status = 'finished'
              AND EXISTS (
                  SELECT 1
                  FROM study_panels AS p
                  LEFT JOIN panel_ratings AS r ON r.panel_id = p.id
                  WHERE p.study_id = studies.id AND r.id IS NULL
              )
            """,
            (utc_now(),),
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


def _current_panel(connection: sqlite3.Connection, study_id: str) -> sqlite3.Row | None:
    return connection.execute(
        """
        SELECT p.*
        FROM study_panels AS p
        LEFT JOIN panel_ratings AS r ON r.panel_id = p.id
        WHERE p.study_id = ? AND r.id IS NULL
        ORDER BY p.display_order
        LIMIT 1
        """,
        (study_id,),
    ).fetchone()


def _require_current_panel(
    connection: sqlite3.Connection, study: sqlite3.Row, panel: sqlite3.Row
) -> None:
    current = _current_panel(connection, study["id"])
    if current is None or current["id"] != panel["id"]:
        raise HTTPException(
            status_code=409,
            detail="Complete the current therapist session and CTRS rating first.",
        )


def _validated_rating(payload: RatingRequest) -> tuple[dict[str, int], str]:
    if set(payload.scores) != set(CTRS_KEYS):
        raise HTTPException(status_code=422, detail="A score is required for all 11 CTRS items.")
    scores = {key: int(payload.scores[key]) for key in CTRS_KEYS}
    if any(score < 0 or score > 6 for score in scores.values()):
        raise HTTPException(status_code=422, detail="Each CTRS score must be between 0 and 6.")
    comments = payload.comments.strip()
    if len(comments) > 4000:
        raise HTTPException(status_code=422, detail="Comments cannot exceed 4,000 characters.")
    return scores, comments


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
    allowed_ids = assigned_profile_ids(connection, study["participant_code"])
    display_number = allowed_ids.index(study["profile_id"]) + 1 if study["profile_id"] in allowed_ids else profile["display_number"]
    current = _current_panel(connection, study["id"])
    current_panel_id = current["id"] if current is not None and study["status"] == "active" else None
    completed_sessions = int(
        connection.execute(
            """
            SELECT COUNT(*)
            FROM panel_ratings AS r
            JOIN study_panels AS p ON p.id = r.panel_id
            WHERE p.study_id = ?
            """,
            (study["id"],),
        ).fetchone()[0]
    )
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
        rating = connection.execute(
            "SELECT scores_json, total_score, comments, submitted_at FROM panel_ratings WHERE panel_id = ?",
            (panel["id"],),
        ).fetchone()
        is_current = panel["id"] == current_panel_id
        if rating is not None:
            panel_status = "completed"
        elif is_current and panel["ended_at"]:
            panel_status = "rating"
        elif is_current:
            panel_status = "active"
        else:
            panel_status = "locked"
        panels.append(
            {
                "id": panel["id"],
                "label": panel["label"],
                "display_order": panel["display_order"],
                "status": panel_status,
                "is_current": is_current,
                "ended_at": panel["ended_at"],
                "termination_reason": panel["termination_reason"],
                "messages": messages,
                "job": serialize_job(connection, job),
                "rating": (
                    {
                        "scores": json.loads(rating["scores_json"]),
                        "total_score": rating["total_score"],
                        "comments": rating["comments"],
                        "submitted_at": rating["submitted_at"],
                    }
                    if rating is not None
                    else None
                ),
                "can_start": (
                    study["status"] == "active"
                    and is_current
                    and not panel["ended_at"]
                    and not messages
                    and not pending
                ),
                "can_send": (
                    study["status"] == "active"
                    and is_current
                    and not panel["ended_at"]
                    and bool(messages)
                    and messages[-1]["role"] == "therapist"
                    and not pending
                ),
                "can_end": (
                    study["status"] == "active"
                    and is_current
                    and not panel["ended_at"]
                    and bool(messages)
                    and messages[-1]["role"] == "therapist"
                    and not pending
                ),
                "can_rate": study["status"] == "active" and is_current and bool(panel["ended_at"]),
            }
        )
    return {
        "id": study["id"],
        "status": study["status"],
        "profile": public_profile(profile, display_number=display_number),
        "panels": panels,
        "current_panel_id": current_panel_id,
        "completed_sessions": completed_sessions,
        "total_sessions": len(PANEL_LABELS),
        "max_session_turns": MAX_SESSION_TURNS,
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
    mode=INFERENCE_MODE,
)


@asynccontextmanager
async def lifespan(_: FastAPI):
    initialize_database()
    if not all(EXPERT_ACCESS_CODES.values()):
        raise RuntimeError(
            "STUDY_EXPERT_1_ACCESS_CODE and STUDY_EXPERT_2_ACCESS_CODE are required."
        )
    if len(set(EXPERT_ACCESS_CODES.values())) != 2:
        raise RuntimeError("The two study access codes must be different.")
    if len(TOKEN_SECRET) < 32:
        raise RuntimeError("STUDY_TOKEN_SECRET is required and must contain at least 32 characters.")
    if INFERENCE_MODE not in {"real", "static"}:
        raise RuntimeError("STUDY_INFERENCE_MODE must be 'real' or 'static'.")
    if MAX_SESSION_TURNS < 1:
        raise RuntimeError("STUDY_MAX_SESSION_TURNS must be at least 1.")
    if len(set(EXPERT_CODES)) != 2 or not all(PARTICIPANT_CODE_PATTERN.fullmatch(code) for code in EXPERT_CODES):
        raise RuntimeError("The two configured expert codes must be distinct valid participant codes.")
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
    email = normalize_email(payload.email)
    if not EMAIL_PATTERN.fullmatch(email):
        raise HTTPException(status_code=422, detail="Enter a valid email address.")
    cohort_code = cohort_for_access_code(payload.access_code)
    if not cohort_code:
        raise HTTPException(status_code=401, detail="The access code is incorrect.")
    now = utc_now()
    with DATABASE_LOCK, database() as connection:
        connection.execute("BEGIN IMMEDIATE")
        participant = connection.execute(
            "SELECT * FROM participants WHERE lower(email) = ?",
            (email,),
        ).fetchone()
        if participant is not None and participant["cohort_code"] != cohort_code:
            connection.rollback()
            raise HTTPException(
                status_code=403,
                detail="This email is registered with a different study access code.",
            )
        if participant is None:
            pool = registration_profile_ids(cohort_code)
            if cohort_code == EXPERT_CODES[0]:
                registered_count = int(
                    connection.execute(
                        """
                        SELECT COUNT(*) FROM participants
                        WHERE cohort_code = ? AND email IS NOT NULL AND email != ''
                        """,
                        (cohort_code,),
                    ).fetchone()[0]
                )
                if registered_count >= 30:
                    connection.rollback()
                    raise HTTPException(
                        status_code=409,
                        detail="All 30 registrations for this access code have already been claimed.",
                    )
                used: set[str] = set()
                for row in connection.execute(
                    """
                    SELECT assigned_profiles_json FROM participants
                    WHERE cohort_code = ? AND email IS NOT NULL AND email != ''
                    """,
                    (cohort_code,),
                ).fetchall():
                    try:
                        used.update(json.loads(row["assigned_profiles_json"] or "[]"))
                    except (TypeError, json.JSONDecodeError):
                        continue
                available = next((profile_id for profile_id in pool if profile_id not in used), None)
                if available is None and not matches_leena(email):
                    connection.rollback()
                    raise HTTPException(
                        status_code=409,
                        detail="All 30 registrations for this access code have already been claimed.",
                    )
                profile_ids = (LEENA_PROFILE_ID,) if matches_leena(email) else (available,)
            else:
                registered = connection.execute(
                    """
                    SELECT participant_code FROM participants
                    WHERE cohort_code = ? AND email IS NOT NULL AND email != '' LIMIT 1
                    """,
                    (cohort_code,),
                ).fetchone()
                if registered is not None:
                    connection.rollback()
                    raise HTTPException(
                        status_code=409,
                        detail="This access code is already registered to an email address.",
                    )
                profile_ids = pool
            participant_code = participant_code_for_email(email)
            connection.execute(
                """
                INSERT INTO participants(
                    participant_code, email, cohort_code, assigned_profiles_json,
                    profile_completed, created_at, last_seen_at
                ) VALUES (?, ?, ?, ?, 0, ?, ?)
                """,
                (participant_code, email, cohort_code, json.dumps(profile_ids), now, now),
            )
            participant = connection.execute(
                "SELECT * FROM participants WHERE participant_code = ?",
                (participant_code,),
            ).fetchone()
        else:
            connection.execute(
                "UPDATE participants SET last_seen_at = ? WHERE participant_code = ?",
                (now, participant["participant_code"]),
            )
            participant = connection.execute(
                "SELECT * FROM participants WHERE participant_code = ?",
                (participant["participant_code"],),
            ).fetchone()
        if matches_leena(email, participant["name"]):
            try:
                assign_leena_patient(connection, participant["participant_code"])
            except HTTPException:
                connection.rollback()
                raise
            participant = connection.execute(
                "SELECT * FROM participants WHERE participant_code = ?",
                (participant["participant_code"],),
            ).fetchone()
        connection.commit()
    profile_completed = int(participant["profile_completed"] or 0) == 1
    return {
        "token": create_token(participant["participant_code"]),
        "email": email,
        "profile_required": not profile_completed,
        "profile": participant_profile_payload(participant) if profile_completed else None,
        "expires_in": TOKEN_TTL_SECONDS,
    }


@app.get("/api/auth/me")
def auth_me(participant_code: str = Depends(require_registered_participant)) -> dict:
    with database() as connection:
        participant = connection.execute(
            "SELECT * FROM participants WHERE participant_code = ?",
            (participant_code,),
        ).fetchone()
    profile_completed = int(participant["profile_completed"] or 0) == 1
    return {
        "email": participant["email"],
        "profile_required": not profile_completed,
        "profile": participant_profile_payload(participant) if profile_completed else None,
    }


@app.post("/api/auth/profile")
def save_participant_profile(
    payload: ParticipantProfileRequest,
    participant_code: str = Depends(require_registered_participant),
) -> dict:
    values = validate_participant_profile(payload)
    with DATABASE_LOCK, database() as connection:
        connection.execute("BEGIN IMMEDIATE")
        try:
            connection.execute(
                """
                UPDATE participants
                SET name = ?, role = ?, institution = ?, latest_degree = ?,
                    years_experience = ?, profile_completed = 1, last_seen_at = ?
                WHERE participant_code = ?
                """,
                (
                    values["name"],
                    values["role"],
                    values["institution"],
                    values["latest_degree"],
                    values["years_experience"],
                    utc_now(),
                    participant_code,
                ),
            )
            participant = connection.execute(
                "SELECT * FROM participants WHERE participant_code = ?",
                (participant_code,),
            ).fetchone()
            if matches_leena(participant["email"], values["name"]):
                assign_leena_patient(connection, participant_code)
        except HTTPException:
            connection.rollback()
            raise
        participant = connection.execute(
            "SELECT * FROM participants WHERE participant_code = ?",
            (participant_code,),
        ).fetchone()
        connection.commit()
    return {"profile": participant_profile_payload(participant)}


@app.get("/api/profiles")
def list_profiles(participant_code: str = Depends(require_participant)) -> dict:
    with database() as connection:
        existing = {
            row["profile_id"]: {
                "id": row["id"],
                "status": row["status"],
                "completed_sessions": int(row["completed_sessions"]),
                "total_sessions": len(PANEL_LABELS),
            }
            for row in connection.execute(
                """
                SELECT s.id, s.profile_id, s.status, COUNT(r.id) AS completed_sessions
                FROM studies AS s
                LEFT JOIN study_panels AS p ON p.study_id = s.id
                LEFT JOIN panel_ratings AS r ON r.panel_id = p.id
                WHERE s.participant_code = ?
                GROUP BY s.id, s.profile_id, s.status
                """,
                (participant_code,),
            ).fetchall()
        }
        allowed_ids = assigned_profile_ids(connection, participant_code)
    return {
        "profiles": [
            profile_card(PROFILES[profile_id], existing.get(profile_id), display_number=index)
            for index, profile_id in enumerate(allowed_ids, start=1)
        ]
    }


@app.post("/api/studies")
def create_study(payload: StudyRequest, participant_code: str = Depends(require_participant)) -> dict:
    profile_id = payload.profile_id.strip()
    now = utc_now()
    with DATABASE_LOCK, database() as connection:
        connection.execute("BEGIN IMMEDIATE")
        allowed_ids = assigned_profile_ids(connection, participant_code)
        if profile_id not in allowed_ids:
            connection.rollback()
            raise HTTPException(status_code=404, detail="Patient profile not found.")
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
        _require_current_panel(connection, study, panel)
        if panel["ended_at"]:
            connection.rollback()
            raise HTTPException(status_code=409, detail="This session is ready for its CTRS rating.")
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
        previous_panel_ids = [
            row["id"]
            for row in connection.execute(
                """
                SELECT p.id
                FROM study_panels AS p
                JOIN studies AS s ON s.id = p.study_id
                WHERE s.participant_code = ?
                  AND p.id <> ?
                  AND p.ended_at IS NULL
                  AND (
                      EXISTS (SELECT 1 FROM panel_messages AS m WHERE m.panel_id = p.id)
                      OR EXISTS (SELECT 1 FROM inference_jobs AS j WHERE j.panel_id = p.id)
                  )
                """,
                (participant_code, panel_id),
            ).fetchall()
        ]
        for previous_panel_id in previous_panel_ids:
            connection.execute(
                """
                UPDATE inference_jobs
                SET status = 'failed', completed_at = ?, error = 'Session left before generation completed.'
                WHERE panel_id = ? AND status IN ('queued', 'running')
                """,
                (utc_now(), previous_panel_id),
            )
        job = _enqueue_job(connection, panel["id"], "start", request_id)
        connection.commit()
        result = serialize_job(connection, job)
    for previous_panel_id in previous_panel_ids:
        INFERENCE_MANAGER.pause_panel(previous_panel_id)
    INFERENCE_MANAGER.activate_panel(panel_id)
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
        _require_current_panel(connection, study, panel)
        if panel["ended_at"]:
            connection.rollback()
            raise HTTPException(status_code=409, detail="This session is ready for its CTRS rating.")
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
        matched_farewell = farewell_phrase(content)
        therapist_turns = int(
            connection.execute(
                "SELECT COUNT(*) FROM panel_messages WHERE panel_id = ? AND role = 'therapist'",
                (panel_id,),
            ).fetchone()[0]
        )
        automatic_reason = (
            "patient_farewell"
            if matched_farewell
            else "max_turns"
            if therapist_turns >= MAX_SESSION_TURNS
            else None
        )
        if automatic_reason:
            connection.execute(
                """
                UPDATE inference_jobs
                SET status = 'completed', started_at = ?, completed_at = ?, error = NULL
                WHERE id = ?
                """,
                (now, now, job["id"]),
            )
            connection.execute(
                "UPDATE study_panels SET ended_at = ?, termination_reason = ? WHERE id = ?",
                (now, automatic_reason, panel_id),
            )
        connection.execute("UPDATE studies SET updated_at = ? WHERE id = ?", (now, study_id))
        connection.commit()
        job = connection.execute("SELECT * FROM inference_jobs WHERE id = ?", (job["id"],)).fetchone()
        result = serialize_job(connection, job)
    if automatic_reason:
        INFERENCE_MANAGER.release_panel(panel_id)
        return {
            "job": result,
            "auto_ended": True,
            "termination_reason": automatic_reason,
        }
    INFERENCE_MANAGER.activate_panel(panel_id)
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
        _require_current_panel(connection, study, panel)
        if panel["ended_at"]:
            connection.rollback()
            raise HTTPException(status_code=409, detail="This session is ready for its CTRS rating.")
        latest_job = _latest_job(connection, panel_id)
        if latest_job is None or latest_job["status"] != "failed":
            connection.rollback()
            raise HTTPException(status_code=409, detail="There is no failed response to retry.")
        job = _enqueue_job(connection, panel["id"], "retry", request_id)
        connection.commit()
        result = serialize_job(connection, job)
    INFERENCE_MANAGER.activate_panel(panel_id)
    INFERENCE_MANAGER.notify()
    return {"job": result}


@app.post("/api/studies/{study_id}/panels/{panel_id}/leave")
def leave_panel(
    study_id: str,
    panel_id: str,
    payload: ActionRequest,
    participant_code: str = Depends(require_participant),
) -> dict:
    _validate_request_id(payload.client_request_id)
    now = utc_now()
    with DATABASE_LOCK, database() as connection:
        connection.execute("BEGIN IMMEDIATE")
        study, panel = get_panel(connection, study_id, panel_id, participant_code)
        panel_ended = bool(panel["ended_at"])
        if not panel_ended:
            connection.execute(
                """
                UPDATE inference_jobs
                SET status = 'failed', completed_at = ?, error = 'Session left before generation completed.'
                WHERE panel_id = ? AND status IN ('queued', 'running')
                """,
                (now, panel_id),
            )
        connection.commit()
    if panel_ended:
        INFERENCE_MANAGER.release_panel(panel_id)
    else:
        INFERENCE_MANAGER.pause_panel(panel_id)
    with database() as connection:
        study = get_study(connection, study_id, participant_code)
        result = serialize_study(connection, study)
    return {"study": result}


@app.post("/api/studies/{study_id}/panels/{panel_id}/end")
def end_panel(
    study_id: str,
    panel_id: str,
    payload: ActionRequest,
    participant_code: str = Depends(require_participant),
) -> dict:
    _validate_request_id(payload.client_request_id)
    now = utc_now()
    with DATABASE_LOCK, database() as connection:
        connection.execute("BEGIN IMMEDIATE")
        study, panel = get_panel(connection, study_id, panel_id, participant_code)
        if study["status"] != "active":
            connection.rollback()
            raise HTTPException(status_code=409, detail="This patient study is finished.")
        _require_current_panel(connection, study, panel)
        if not panel["ended_at"]:
            pending = connection.execute(
                "SELECT 1 FROM inference_jobs WHERE panel_id = ? AND status IN ('queued', 'running')",
                (panel_id,),
            ).fetchone()
            latest = connection.execute(
                "SELECT role FROM panel_messages WHERE panel_id = ? ORDER BY id DESC LIMIT 1",
                (panel_id,),
            ).fetchone()
            if pending:
                connection.rollback()
                raise HTTPException(status_code=409, detail="Wait for the therapist response before ending.")
            if latest is None or latest["role"] != "therapist":
                connection.rollback()
                raise HTTPException(status_code=409, detail="Start the session before ending it.")
            connection.execute(
                "UPDATE study_panels SET ended_at = ?, termination_reason = 'expert_ended' WHERE id = ?",
                (now, panel_id),
            )
            connection.execute("UPDATE studies SET updated_at = ? WHERE id = ?", (now, study_id))
        connection.commit()
        study = get_study(connection, study_id, participant_code)
        result = serialize_study(connection, study)
    # No inference is needed while the expert completes CTRS. Terminating the
    # panel-owned subprocess releases the model and its CUDA allocations.
    INFERENCE_MANAGER.release_panel(panel_id)
    return {"study": result}


@app.post("/api/studies/{study_id}/panels/{panel_id}/rating")
def rate_panel(
    study_id: str,
    panel_id: str,
    payload: RatingRequest,
    participant_code: str = Depends(require_participant),
) -> dict:
    scores, comments = _validated_rating(payload)
    now = utc_now()
    with DATABASE_LOCK, database() as connection:
        connection.execute("BEGIN IMMEDIATE")
        study, panel = get_panel(connection, study_id, panel_id, participant_code)
        existing = connection.execute(
            "SELECT total_score FROM panel_ratings WHERE panel_id = ?", (panel_id,)
        ).fetchone()
        if existing is not None:
            connection.commit()
            study = get_study(connection, study_id, participant_code)
            return {"study": serialize_study(connection, study), "total_score": existing["total_score"]}
        if study["status"] != "active":
            connection.rollback()
            raise HTTPException(status_code=409, detail="This patient study is finished.")
        _require_current_panel(connection, study, panel)
        if not panel["ended_at"]:
            connection.rollback()
            raise HTTPException(status_code=409, detail="End the therapist session before submitting its rating.")
        total_score = sum(scores.values())
        connection.execute(
            """
            INSERT INTO panel_ratings(
                id, panel_id, participant_code, scores_json, total_score, comments, submitted_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(uuid.uuid4()),
                panel_id,
                participant_code,
                json.dumps(scores, separators=(",", ":")),
                total_score,
                comments,
                now,
            ),
        )
        completed = int(
            connection.execute(
                """
                SELECT COUNT(*)
                FROM panel_ratings AS r
                JOIN study_panels AS p ON p.id = r.panel_id
                WHERE p.study_id = ?
                """,
                (study_id,),
            ).fetchone()[0]
        )
        if completed == len(PANEL_LABELS):
            connection.execute(
                "UPDATE studies SET status = 'finished', finished_at = ?, updated_at = ? WHERE id = ?",
                (now, now, study_id),
            )
        else:
            connection.execute("UPDATE studies SET updated_at = ? WHERE id = ?", (now, study_id))
        connection.commit()
        study = get_study(connection, study_id, participant_code)
        result = serialize_study(connection, study)
    # Idempotent fallback for sessions ended before an API restart or by an
    # older frontend that did not trigger cleanup at the end step.
    INFERENCE_MANAGER.release_panel(panel_id)
    return {"study": result, "total_score": total_score}


@app.post("/api/studies/{study_id}/finish")
def finish_study(study_id: str, participant_code: str = Depends(require_participant)) -> dict:
    now = utc_now()
    with DATABASE_LOCK, database() as connection:
        connection.execute("BEGIN IMMEDIATE")
        study = get_study(connection, study_id, participant_code)
        remaining = connection.execute(
            """
            SELECT COUNT(*)
            FROM study_panels AS p
            LEFT JOIN panel_ratings AS r ON r.panel_id = p.id
            WHERE p.study_id = ? AND r.id IS NULL
            """,
            (study_id,),
        ).fetchone()[0]
        if remaining:
            connection.rollback()
            raise HTTPException(
                status_code=409,
                detail="Complete and rate all six therapist sessions before finishing this patient.",
            )
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
