from __future__ import annotations

import importlib
import sqlite3
import sys


def test_selective_reset_keeps_three_rated_sessions_and_clears_everything_else(
    monkeypatch, tmp_path
):
    database_path = tmp_path / "study.sqlite3"
    monkeypatch.setenv("STUDY_DB_PATH", str(database_path))
    monkeypatch.setenv("STUDY_EXPERT_1_ACCESS_CODE", "first-access-code")
    monkeypatch.setenv("STUDY_EXPERT_2_ACCESS_CODE", "second-access-code")
    monkeypatch.setenv("STUDY_TOKEN_SECRET", "test-token-secret-that-is-at-least-32-characters")
    monkeypatch.setenv("STUDY_INFERENCE_MODE", "static")
    sys.modules.pop("server.app", None)
    app = importlib.import_module("server.app")
    app.initialize_database()

    now = "2026-09-28T12:00:00.000Z"
    connection = sqlite3.connect(database_path)
    connection.execute("PRAGMA foreign_keys = ON")
    for participant in ("KEEP-EXPERT", "REMOVE-EXPERT"):
        connection.execute(
            "INSERT INTO participants(participant_code, created_at, last_seen_at) VALUES (?, ?, ?)",
            (participant, now, now),
        )
    for study_id, participant in (("keep-study", "KEEP-EXPERT"), ("remove-study", "REMOVE-EXPERT")):
        connection.execute(
            """
            INSERT INTO studies(id, participant_code, profile_id, status, created_at, updated_at)
            VALUES (?, ?, 'patient-39', 'active', ?, ?)
            """,
            (study_id, participant, now, now),
        )
        for index in range(6):
            panel_id = f"{study_id}-panel-{index}"
            connection.execute(
                """
                INSERT INTO study_panels(
                    id, study_id, label, method_key, display_order, ended_at,
                    termination_reason
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    panel_id,
                    study_id,
                    f"Therapist {'ABCDEF'[index]}",
                    app.METHOD_KEYS[index],
                    index,
                    now,
                    "expert_ended",
                ),
            )
            connection.execute(
                """
                INSERT INTO panel_messages(panel_id, role, content, client_message_id, created_at)
                VALUES (?, 'therapist', ?, ?, ?)
                """,
                (panel_id, f"message-{index}", f"message-{study_id}-{index}", now),
            )
            connection.execute(
                """
                INSERT INTO inference_jobs(
                    id, panel_id, kind, client_request_id, status, created_at,
                    started_at, completed_at
                ) VALUES (?, ?, 'start', ?, 'completed', ?, ?, ?)
                """,
                (
                    f"job-{study_id}-{index}",
                    panel_id,
                    f"request-{study_id}-{index}",
                    now,
                    now,
                    now,
                ),
            )
            if study_id == "keep-study" and index < 3:
                connection.execute(
                    """
                    INSERT INTO panel_ratings(
                        id, panel_id, participant_code, scores_json, total_score,
                        comments, submitted_at
                    ) VALUES (?, ?, ?, '{}', 33, 'keep', ?)
                    """,
                    (
                        f"rating-{index}",
                        panel_id,
                        # Reproduce the live legacy inconsistency that caused
                        # the participant deletion foreign-key failure.
                        "REMOVE-EXPERT" if index == 2 else "KEEP-EXPERT",
                        now,
                    ),
                )
    connection.commit()
    connection.close()

    from server.reset_db import reset_database

    backup_path = tmp_path / "backup.sqlite3"
    before, after = reset_database(database_path, "keep-study", 3, backup_path)
    assert before["studies"] == 2
    assert after == {
        "participants": 1,
        "studies": 1,
        "study_panels": 6,
        "panel_messages": 3,
        "inference_jobs": 3,
        "panel_ratings": 3,
    }
    assert backup_path.is_file()

    connection = sqlite3.connect(database_path)
    connection.row_factory = sqlite3.Row
    study = connection.execute("SELECT * FROM studies").fetchone()
    unrated = connection.execute(
        """
        SELECT p.ended_at, p.termination_reason
        FROM study_panels AS p
        LEFT JOIN panel_ratings AS r ON r.panel_id = p.id
        WHERE p.study_id = 'keep-study' AND r.id IS NULL
        """
    ).fetchall()
    connection.close()
    assert study["id"] == "keep-study"
    assert study["status"] == "active"
    assert all(row["ended_at"] is None and row["termination_reason"] is None for row in unrated)

    connection = sqlite3.connect(database_path)
    rating_owners = {
        row[0] for row in connection.execute("SELECT participant_code FROM panel_ratings")
    }
    connection.close()
    assert rating_owners == {"KEEP-EXPERT"}
