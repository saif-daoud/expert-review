from __future__ import annotations

import argparse
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path


TABLES = (
    "participants",
    "studies",
    "study_panels",
    "panel_messages",
    "inference_jobs",
    "panel_ratings",
)


def table_counts(connection: sqlite3.Connection) -> dict[str, int]:
    return {
        table: int(connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
        for table in TABLES
    }


def inspect_preserved_study(
    connection: sqlite3.Connection, keep_study_id: str, expected_ratings: int
) -> sqlite3.Row:
    study = connection.execute(
        "SELECT * FROM studies WHERE id = ?", (keep_study_id,)
    ).fetchone()
    if study is None:
        raise RuntimeError(f"Preserved study not found: {keep_study_id}")
    rating_count = int(
        connection.execute(
            """
            SELECT COUNT(*)
            FROM panel_ratings AS r
            JOIN study_panels AS p ON p.id = r.panel_id
            WHERE p.study_id = ?
            """,
            (keep_study_id,),
        ).fetchone()[0]
    )
    if rating_count != expected_ratings:
        raise RuntimeError(
            f"Refusing reset: study {keep_study_id} has {rating_count} ratings, "
            f"not the expected {expected_ratings}."
        )
    return study


def reset_database(
    database_path: Path,
    keep_study_id: str,
    expected_ratings: int,
    backup_path: Path,
) -> tuple[dict[str, int], dict[str, int]]:
    connection = sqlite3.connect(database_path, timeout=30, isolation_level=None)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA busy_timeout = 30000")
    try:
        study = inspect_preserved_study(connection, keep_study_id, expected_ratings)
        owner = study["participant_code"]
        before = table_counts(connection)

        backup_path.parent.mkdir(parents=True, exist_ok=True)
        backup = sqlite3.connect(backup_path)
        try:
            connection.backup(backup)
        finally:
            backup.close()

        connection.execute("BEGIN IMMEDIATE")
        try:
            connection.execute("DELETE FROM studies WHERE id != ?", (keep_study_id,))
            unrated_panels = """
                SELECT p.id
                FROM study_panels AS p
                LEFT JOIN panel_ratings AS r ON r.panel_id = p.id
                WHERE p.study_id = ? AND r.id IS NULL
            """
            connection.execute(
                f"DELETE FROM inference_jobs WHERE panel_id IN ({unrated_panels})",
                (keep_study_id,),
            )
            connection.execute(
                f"DELETE FROM panel_messages WHERE panel_id IN ({unrated_panels})",
                (keep_study_id,),
            )
            connection.execute(
                f"""
                UPDATE study_panels
                SET ended_at = NULL, termination_reason = NULL
                WHERE id IN ({unrated_panels})
                """,
                (keep_study_id,),
            )
            connection.execute(
                """
                UPDATE studies
                SET status = 'active', finished_at = NULL, updated_at = ?
                WHERE id = ?
                """,
                (
                    datetime.now(timezone.utc)
                    .isoformat(timespec="milliseconds")
                    .replace("+00:00", "Z"),
                    keep_study_id,
                ),
            )
            connection.execute(
                "DELETE FROM participants WHERE participant_code != ?", (owner,)
            )
            connection.commit()
        except Exception:
            connection.rollback()
            raise

        inspect_preserved_study(connection, keep_study_id, expected_ratings)
        remaining_studies = int(
            connection.execute("SELECT COUNT(*) FROM studies").fetchone()[0]
        )
        if remaining_studies != 1:
            raise RuntimeError(
                f"Reset verification failed: {remaining_studies} studies remain. "
                f"Restore {backup_path}."
            )
        after = table_counts(connection)
        return before, after
    finally:
        connection.close()


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Reset the study database while preserving one completed/partially "
            "completed study. Stop the API before using --apply."
        )
    )
    parser.add_argument("--keep-study", required=True, help="Study UUID to preserve.")
    parser.add_argument("--expected-ratings", type=int, default=3)
    parser.add_argument(
        "--database",
        type=Path,
        default=None,
        help="Database path; defaults to STUDY_DB_PATH or server/data/study.sqlite3.",
    )
    parser.add_argument("--backup", type=Path, default=None)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Perform the reset. Without this flag the command only validates and previews.",
    )
    args = parser.parse_args()

    server_dir = Path(__file__).resolve().parent
    database_path = (
        args.database
        or Path(os.getenv("STUDY_DB_PATH", server_dir / "data" / "study.sqlite3"))
    ).expanduser().resolve()
    if not database_path.is_file():
        raise SystemExit(f"Database does not exist: {database_path}")

    connection = sqlite3.connect(database_path)
    connection.row_factory = sqlite3.Row
    try:
        study = inspect_preserved_study(
            connection, args.keep_study, args.expected_ratings
        )
        before = table_counts(connection)
    finally:
        connection.close()
    print(f"Database: {database_path}")
    print(
        f"Preserving study={study['id']} participant={study['participant_code']} "
        f"profile={study['profile_id']} ratings={args.expected_ratings}"
    )
    print("Before: " + " ".join(f"{key}={value}" for key, value in before.items()))
    if not args.apply:
        print("Dry run only. Stop the API and add --apply to perform the reset.")
        return 0

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_path = (
        args.backup
        or database_path.with_name(f"{database_path.stem}.before-reset-{timestamp}.sqlite3")
    ).expanduser().resolve()
    before, after = reset_database(
        database_path,
        args.keep_study,
        args.expected_ratings,
        backup_path,
    )
    print(f"Backup: {backup_path}")
    print("After:  " + " ".join(f"{key}={value}" for key, value in after.items()))
    print("Reset complete. The preserved study still has the expected ratings.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
