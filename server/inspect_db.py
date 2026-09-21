from __future__ import annotations

import argparse
import os
import sqlite3
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect the live-interaction SQLite database.")
    parser.add_argument("--study", help="Show one study and its panels.")
    parser.add_argument("--participant", help="Filter the study list by participant code.")
    parser.add_argument("--messages", action="store_true", help="Include transcript messages with --study.")
    parser.add_argument("--show-methods", action="store_true", help="Reveal the blinded server-side method mapping.")
    args = parser.parse_args()

    server_dir = Path(__file__).resolve().parent
    database_path = Path(os.getenv("STUDY_DB_PATH", server_dir / "data" / "study.sqlite3")).expanduser()
    if not database_path.exists():
        raise SystemExit(f"Database does not exist yet: {database_path}")

    connection = sqlite3.connect(database_path)
    connection.row_factory = sqlite3.Row
    print(f"Database: {database_path.resolve()}")

    totals = {
        table: connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        for table in ("participants", "studies", "study_panels", "panel_messages", "inference_jobs")
    }
    print("\nTotals")
    print("  " + "  ".join(f"{key}={value}" for key, value in totals.items()))

    if args.study:
        study = connection.execute("SELECT * FROM studies WHERE id = ?", (args.study,)).fetchone()
        if study is None:
            raise SystemExit(f"Study not found: {args.study}")
        print(
            f"\nStudy {study['id']}\n"
            f"  participant={study['participant_code']} profile={study['profile_id']} status={study['status']}\n"
            f"  created={study['created_at']} updated={study['updated_at']}"
        )
        panels = connection.execute(
            """
            SELECT p.*,
                   (SELECT COUNT(*) FROM panel_messages AS m WHERE m.panel_id = p.id) AS message_count,
                   (SELECT status FROM inference_jobs AS j WHERE j.panel_id = p.id ORDER BY sequence DESC LIMIT 1) AS job_status
            FROM study_panels AS p
            WHERE p.study_id = ?
            ORDER BY p.display_order
            """,
            (args.study,),
        ).fetchall()
        print("\nPanels")
        for panel in panels:
            method = f" method={panel['method_key']}" if args.show_methods else ""
            print(
                f"  {panel['label']}  id={panel['id']}{method} "
                f"messages={panel['message_count']} latest_job={panel['job_status'] or '-'}"
            )
            if args.messages:
                messages = connection.execute(
                    "SELECT role, content, created_at FROM panel_messages WHERE panel_id = ? ORDER BY id",
                    (panel["id"],),
                ).fetchall()
                for message in messages:
                    content = " ".join(message["content"].split())
                    print(f"    [{message['created_at']}] {message['role']}: {content}")
        return 0

    query = """
        SELECT s.id, s.participant_code, s.profile_id, s.status,
               COUNT(DISTINCT p.id) AS panels,
               COUNT(DISTINCT m.id) AS messages,
               s.created_at, s.updated_at
        FROM studies AS s
        LEFT JOIN study_panels AS p ON p.study_id = s.id
        LEFT JOIN panel_messages AS m ON m.panel_id = p.id
    """
    parameters: tuple[str, ...] = ()
    if args.participant:
        query += " WHERE s.participant_code = ?"
        parameters = (args.participant,)
    query += " GROUP BY s.id ORDER BY s.created_at DESC"
    rows = connection.execute(query, parameters).fetchall()
    print("\nStudies")
    if not rows:
        print("  No studies yet.")
    for row in rows:
        print(
            f"  {row['id']} participant={row['participant_code']} profile={row['profile_id']} "
            f"status={row['status']} panels={row['panels']} messages={row['messages']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
