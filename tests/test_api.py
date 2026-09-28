from __future__ import annotations

import importlib
import json
import sqlite3
import sys
import time

from fastapi.testclient import TestClient

TEST_ACCESS_CODES = {
    "EXPERT-5834": "shared-referral-access-code",
    "EXPERT-9271": "second-expert-access-code",
}


def load_client(monkeypatch, tmp_path, max_session_turns=50):
    monkeypatch.setenv("STUDY_DB_PATH", str(tmp_path / "study.sqlite3"))
    monkeypatch.setenv("STUDY_EXPERT_1_ACCESS_CODE", TEST_ACCESS_CODES["EXPERT-5834"])
    monkeypatch.setenv("STUDY_EXPERT_2_ACCESS_CODE", TEST_ACCESS_CODES["EXPERT-9271"])
    monkeypatch.setenv("STUDY_TOKEN_SECRET", "test-token-secret-that-is-not-used-in-production")
    monkeypatch.setenv("STUDY_ALLOWED_ORIGINS", "http://127.0.0.1:5500")
    monkeypatch.setenv("STUDY_SERVE_FRONTEND", "false")
    monkeypatch.setenv("STUDY_INFERENCE_MODE", "static")
    monkeypatch.setenv("STUDY_MAX_SESSION_TURNS", str(max_session_turns))
    sys.modules.pop("server.app", None)
    module = importlib.import_module("server.app")
    return TestClient(module.app)


def login(
    client: TestClient,
    participant_code: str,
    email: str | None = None,
) -> dict[str, str]:
    email = email or f"{participant_code.lower()}@example.org"
    response = client.post(
        "/api/auth/login",
        json={
            "email": email,
            "access_code": TEST_ACCESS_CODES[participant_code],
        },
    )
    assert response.status_code == 200
    payload = response.json()
    headers = {"Authorization": f"Bearer {payload['token']}"}
    if payload["profile_required"]:
        profile = client.post(
            "/api/auth/profile",
            headers=headers,
            json={
                "name": "Dr Test Expert",
                "role": "Clinical psychologist",
                "institution": "Test Institute",
                "latest_degree": "PhD",
                "years_experience": 12,
            },
        )
        assert profile.status_code == 200
    resumed = client.post(
        "/api/auth/login",
        json={
            "email": email,
            "access_code": TEST_ACCESS_CODES[participant_code],
        },
    )
    assert resumed.status_code == 200
    assert resumed.json()["profile_required"] is False
    return {"Authorization": f"Bearer {resumed.json()['token']}"}


def wait_for_panel(client: TestClient, headers: dict[str, str], study_id: str, panel_id: str) -> dict:
    deadline = time.monotonic() + 4
    while time.monotonic() < deadline:
        study = client.get(f"/api/studies/{study_id}", headers=headers).json()["study"]
        panel = next(item for item in study["panels"] if item["id"] == panel_id)
        if panel["job"]["status"] in {"completed", "failed"}:
            return panel
        time.sleep(0.03)
    raise AssertionError("Inference job did not finish")


def test_first_login_collects_professional_profile_and_returning_email_skips_it(monkeypatch, tmp_path):
    with load_client(monkeypatch, tmp_path) as client:
        credentials = {
            "email": "Therapist.Expert@example.org",
            "access_code": TEST_ACCESS_CODES["EXPERT-5834"],
        }
        first = client.post("/api/auth/login", json=credentials)
        assert first.status_code == 200
        payload = first.json()
        assert payload["email"] == "therapist.expert@example.org"
        assert payload["profile_required"] is True
        headers = {"Authorization": f"Bearer {payload['token']}"}
        assert client.get("/api/profiles", headers=headers).status_code == 403
        me = client.get("/api/auth/me", headers=headers)
        assert me.status_code == 200
        assert me.json()["profile_required"] is True

        saved = client.post(
            "/api/auth/profile",
            headers=headers,
            json={
                "name": "Dr Example",
                "role": "CBT therapist",
                "institution": "Example Clinic",
                "latest_degree": "PsyD",
                "years_experience": 9,
            },
        )
        assert saved.status_code == 200
        assert saved.json()["profile"]["role"] == "CBT therapist"

        returning = client.post("/api/auth/login", json=credentials)
        assert returning.status_code == 200
        assert returning.json()["profile_required"] is False
        assert returning.json()["profile"]["name"] == "Dr Example"


def test_sequential_six_session_ctrs_flow(monkeypatch, tmp_path):
    with load_client(monkeypatch, tmp_path) as client:
        health = client.get("/api/health").json()
        assert health["status"] == "ok"
        assert health["inference"]["mode"] == "static"

        cors = client.options(
            "/api/health",
            headers={
                "Origin": "http://127.0.0.1:5500",
                "Access-Control-Request-Method": "GET",
                "Access-Control-Request-Headers": "ngrok-skip-browser-warning",
            },
        )
        assert cors.status_code == 200
        assert cors.headers["access-control-allow-origin"] == "http://127.0.0.1:5500"

        headers = login(client, "EXPERT-5834")
        profiles = client.get("/api/profiles", headers=headers).json()["profiles"]
        assert len(profiles) == 1
        assert set(profiles[0]) == {
            "id",
            "display_number",
            "display_name",
            "condition",
            "short_description",
        }
        assert [profile["display_number"] for profile in profiles] == [1]

        created = client.post("/api/studies", headers=headers, json={"profile_id": profiles[0]["id"]})
        assert created.status_code == 200
        study = created.json()["study"]
        assert len(study["panels"]) == 6
        assert [panel["label"] for panel in study["panels"]] == [f"Therapist {letter}" for letter in "ABCDEF"]
        assert "method_key" not in str(study)
        assert study["completed_sessions"] == 0
        assert study["total_sessions"] == 6
        assert study["current_panel_id"] == study["panels"][0]["id"]
        assert study["panels"][0]["status"] == "active"
        assert all(panel["status"] == "locked" for panel in study["panels"][1:])

        locked = client.post(
            f"/api/studies/{study['id']}/panels/{study['panels'][1]['id']}/start",
            headers=headers,
            json={"client_request_id": "start-locked"},
        )
        assert locked.status_code == 409

        panel = study["panels"][0]
        started = client.post(
            f"/api/studies/{study['id']}/panels/{panel['id']}/start",
            headers=headers,
            json={"client_request_id": "start-one"},
        )
        assert started.status_code == 200
        panel = wait_for_panel(client, headers, study["id"], panel["id"])
        assert panel["job"]["status"] == "completed"
        assert [message["role"] for message in panel["messages"]] == ["therapist"]

        message_payload = {"content": "I have been feeling anxious.", "client_message_id": "message-one"}
        sent = client.post(
            f"/api/studies/{study['id']}/panels/{panel['id']}/messages",
            headers=headers,
            json=message_payload,
        )
        assert sent.status_code == 200
        panel = wait_for_panel(client, headers, study["id"], panel["id"])
        assert [message["role"] for message in panel["messages"]] == ["therapist", "patient", "therapist"]

        retried_request = client.post(
            f"/api/studies/{study['id']}/panels/{panel['id']}/messages",
            headers=headers,
            json=message_payload,
        )
        assert retried_request.status_code == 200
        panel = client.get(f"/api/studies/{study['id']}", headers=headers).json()["study"]["panels"][0]
        assert len(panel["messages"]) == 3

        premature_finish = client.post(f"/api/studies/{study['id']}/finish", headers=headers)
        assert premature_finish.status_code == 409

        ctrs_keys = {
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
        }

        for index in range(6):
            study = client.get(f"/api/studies/{study['id']}", headers=headers).json()["study"]
            panel = next(item for item in study["panels"] if item["id"] == study["current_panel_id"])
            if index > 0:
                started = client.post(
                    f"/api/studies/{study['id']}/panels/{panel['id']}/start",
                    headers=headers,
                    json={"client_request_id": f"start-{index}"},
                )
                assert started.status_code == 200
                panel = wait_for_panel(client, headers, study["id"], panel["id"])

            ended = client.post(
                f"/api/studies/{study['id']}/panels/{panel['id']}/end",
                headers=headers,
                json={"client_request_id": f"end-{index}"},
            )
            assert ended.status_code == 200
            ended_study = ended.json()["study"]
            ended_panel = next(item for item in ended_study["panels"] if item["id"] == panel["id"])
            assert ended_panel["status"] == "rating"
            assert ended_panel["can_rate"] is True

            if index == 0:
                incomplete = client.post(
                    f"/api/studies/{study['id']}/panels/{panel['id']}/rating",
                    headers=headers,
                    json={"scores": {"agenda": 4}, "comments": ""},
                )
                assert incomplete.status_code == 422

            rated = client.post(
                f"/api/studies/{study['id']}/panels/{panel['id']}/rating",
                headers=headers,
                json={"scores": {key: 3 for key in ctrs_keys}, "comments": "Reviewed."},
            )
            assert rated.status_code == 200
            assert rated.json()["total_score"] == 33
            study = rated.json()["study"]
            assert study["completed_sessions"] == index + 1

        assert study["status"] == "finished"
        assert study["current_panel_id"] is None
        assert all(panel["status"] == "completed" for panel in study["panels"])

        finished = client.post(f"/api/studies/{study['id']}/finish", headers=headers)
        assert finished.status_code == 200
        profiles = client.get("/api/profiles", headers=headers).json()["profiles"]
        assert profiles[0]["study"]["completed_sessions"] == 6


def test_patient_farewell_ends_without_another_therapist_response(monkeypatch, tmp_path):
    with load_client(monkeypatch, tmp_path) as client:
        headers = login(client, "EXPERT-5834")
        profile = client.get("/api/profiles", headers=headers).json()["profiles"][0]
        study = client.post("/api/studies", headers=headers, json={"profile_id": profile["id"]}).json()["study"]
        panel = study["panels"][0]
        client.post(
            f"/api/studies/{study['id']}/panels/{panel['id']}/start",
            headers=headers,
            json={"client_request_id": "farewell-start"},
        )
        panel = wait_for_panel(client, headers, study["id"], panel["id"])

        response = client.post(
            f"/api/studies/{study['id']}/panels/{panel['id']}/messages",
            headers=headers,
            json={"content": "Thank you. Bye-bye.", "client_message_id": "patient-farewell"},
        )
        assert response.status_code == 200
        assert response.json()["auto_ended"] is True
        assert response.json()["termination_reason"] == "patient_farewell"
        study = client.get(f"/api/studies/{study['id']}", headers=headers).json()["study"]
        panel = study["panels"][0]
        assert panel["status"] == "rating"
        assert panel["termination_reason"] == "patient_farewell"
        assert [message["role"] for message in panel["messages"]] == ["therapist", "patient"]


def test_leaving_unloads_without_ending_and_next_patient_pauses_previous(monkeypatch, tmp_path):
    with load_client(monkeypatch, tmp_path) as client:
        app_module = sys.modules["server.app"]
        paused_panels = []
        activated_panels = []
        original_pause = app_module.INFERENCE_MANAGER.pause_panel
        original_activate = app_module.INFERENCE_MANAGER.activate_panel

        def record_pause(panel_id):
            paused_panels.append(panel_id)
            original_pause(panel_id)

        def record_activate(panel_id):
            activated_panels.append(panel_id)
            original_activate(panel_id)

        monkeypatch.setattr(app_module.INFERENCE_MANAGER, "pause_panel", record_pause)
        monkeypatch.setattr(app_module.INFERENCE_MANAGER, "activate_panel", record_activate)

        headers = login(client, "EXPERT-9271")
        profiles = client.get("/api/profiles", headers=headers).json()["profiles"]
        first_study = client.post(
            "/api/studies", headers=headers, json={"profile_id": profiles[0]["id"]}
        ).json()["study"]
        first_panel = first_study["panels"][0]
        client.post(
            f"/api/studies/{first_study['id']}/panels/{first_panel['id']}/start",
            headers=headers,
            json={"client_request_id": "leave-start-one"},
        )
        first_panel = wait_for_panel(client, headers, first_study["id"], first_panel["id"])
        original_messages = first_panel["messages"]

        left = client.post(
            f"/api/studies/{first_study['id']}/panels/{first_panel['id']}/leave",
            headers=headers,
            json={"client_request_id": "leave-one"},
        )
        assert left.status_code == 200
        left_panel = left.json()["study"]["panels"][0]
        assert left_panel["ended_at"] is None
        assert left_panel["status"] == "active"
        assert left_panel["messages"] == original_messages
        assert paused_panels == [first_panel["id"]]

        second_study = client.post(
            "/api/studies", headers=headers, json={"profile_id": profiles[1]["id"]}
        ).json()["study"]
        second_panel = second_study["panels"][0]
        started = client.post(
            f"/api/studies/{second_study['id']}/panels/{second_panel['id']}/start",
            headers=headers,
            json={"client_request_id": "leave-start-two"},
        )
        assert started.status_code == 200
        assert paused_panels == [first_panel["id"], first_panel["id"]]
        assert activated_panels[-1] == second_panel["id"]


def test_therapist_farewell_ends_after_storing_response(monkeypatch, tmp_path):
    with load_client(monkeypatch, tmp_path) as client:
        app_module = sys.modules["server.app"]
        monkeypatch.setattr(
            app_module.INFERENCE_MANAGER,
            "_static_response",
            lambda _method, _history: "Take care. Goodbye.",
        )
        headers = login(client, "EXPERT-5834")
        profile = client.get("/api/profiles", headers=headers).json()["profiles"][0]
        study = client.post("/api/studies", headers=headers, json={"profile_id": profile["id"]}).json()["study"]
        panel = study["panels"][0]
        client.post(
            f"/api/studies/{study['id']}/panels/{panel['id']}/start",
            headers=headers,
            json={"client_request_id": "therapist-farewell"},
        )
        panel = wait_for_panel(client, headers, study["id"], panel["id"])
        assert panel["status"] == "rating"
        assert panel["termination_reason"] == "therapist_farewell"
        assert panel["messages"][-1]["content"] == "Take care. Goodbye."


def test_session_ends_after_configured_dialogue_turn_limit(monkeypatch, tmp_path):
    with load_client(monkeypatch, tmp_path, max_session_turns=2) as client:
        headers = login(client, "EXPERT-5834")
        profile = client.get("/api/profiles", headers=headers).json()["profiles"][0]
        study = client.post("/api/studies", headers=headers, json={"profile_id": profile["id"]}).json()["study"]
        panel = study["panels"][0]
        client.post(
            f"/api/studies/{study['id']}/panels/{panel['id']}/start",
            headers=headers,
            json={"client_request_id": "limit-start"},
        )
        panel = wait_for_panel(client, headers, study["id"], panel["id"])
        client.post(
            f"/api/studies/{study['id']}/panels/{panel['id']}/messages",
            headers=headers,
            json={"content": "First response.", "client_message_id": "limit-one"},
        )
        panel = wait_for_panel(client, headers, study["id"], panel["id"])
        response = client.post(
            f"/api/studies/{study['id']}/panels/{panel['id']}/messages",
            headers=headers,
            json={"content": "Second response.", "client_message_id": "limit-two"},
        )
        assert response.status_code == 200
        assert response.json()["termination_reason"] == "max_turns"
        study = client.get(f"/api/studies/{study['id']}", headers=headers).json()["study"]
        panel = study["panels"][0]
        assert study["max_session_turns"] == 2
        assert panel["status"] == "rating"
        assert panel["termination_reason"] == "max_turns"
        assert [message["role"] for message in panel["messages"]] == [
            "therapist",
            "patient",
            "therapist",
            "patient",
        ]


def test_studies_are_private_and_mapping_is_stable(monkeypatch, tmp_path):
    with load_client(monkeypatch, tmp_path) as client:
        first = login(client, "EXPERT-5834", "referral-01@example.org")
        other_shared = login(client, "EXPERT-5834", "referral-02@example.org")
        second = login(client, "EXPERT-9271", "second-expert@example.org")
        first_profiles = client.get("/api/profiles", headers=first).json()["profiles"]
        other_shared_profiles = client.get("/api/profiles", headers=other_shared).json()["profiles"]
        second_profiles = client.get("/api/profiles", headers=second).json()["profiles"]
        assert len(first_profiles) == len(other_shared_profiles) == 1
        assert len(second_profiles) == 10
        assert first_profiles[0]["id"] == "patient-1"
        assert other_shared_profiles[0]["id"] == "patient-2"
        assert [profile["id"] for profile in second_profiles] == [
            f"patient-{number}" for number in range(31, 41)
        ]
        assert [profile["display_number"] for profile in first_profiles] == [1]
        assert [profile["display_name"] for profile in first_profiles] == ["Patient 1"]
        assert [profile["display_number"] for profile in second_profiles] == list(range(1, 11))
        assert [profile["display_name"] for profile in second_profiles] == [
            f"Patient {number}" for number in range(1, 11)
        ]
        assert {profile["id"] for profile in first_profiles + other_shared_profiles}.isdisjoint(
            profile["id"] for profile in second_profiles
        )
        crossed_access_code = client.post(
            "/api/auth/login",
            json={
                "email": "referral-01@example.org",
                "access_code": TEST_ACCESS_CODES["EXPERT-9271"],
            },
        )
        assert crossed_access_code.status_code == 403
        app_module = sys.modules["server.app"]
        assert app_module.FIRST_SHARED_PROFILE_IDS == tuple(
            f"patient-{number}" for number in range(1, 31)
        )
        assert app_module.SECOND_EXPERT_PROFILE_IDS == tuple(
            f"patient-{number}" for number in range(31, 41)
        )
        assert set(app_module.FIRST_SHARED_PROFILE_IDS).isdisjoint(app_module.SECOND_EXPERT_PROFILE_IDS)
        for group in (1, 2):
            assigned = [
                profile for profile in app_module.PROFILES.values() if profile["assignment_group"] == group
            ]
            assert sum(profile["condition"] == "Anxiety disorder" for profile in assigned) == 10
            assert sum(profile["condition"] == "Depression" for profile in assigned) == 10

        # IDs from the original ten-profile deployment remain attached to the
        # same source cases, so existing studies do not change patient roles.
        stable_ids = {
            "patient-1": "patient_act_001",
            "patient-2": "patient_act_002",
            "patient-3": "patient_act_003",
            "patient-4": "patient_act_021",
            "patient-5": "patient_act_022",
            "patient-6": "patient_act_004",
            "patient-7": "patient_act_007",
            "patient-8": "patient_act_016",
            "patient-9": "patient_act_023",
            "patient-10": "patient_act_026",
        }
        assert {
            public_id: app_module.PROFILES[public_id]["source_id"] for public_id in stable_ids
        } == stable_ids
        profile_id = first_profiles[0]["id"]
        first_create = client.post("/api/studies", headers=first, json={"profile_id": profile_id}).json()["study"]
        second_create = client.post("/api/studies", headers=first, json={"profile_id": profile_id}).json()["study"]
        assert first_create["id"] == second_create["id"]

        forbidden = client.get(f"/api/studies/{first_create['id']}", headers=other_shared)
        assert forbidden.status_code == 404

        wrong_profile = client.post(
            "/api/studies", headers=second, json={"profile_id": profile_id}
        )
        assert wrong_profile.status_code == 404

        allocated_ids = {first_profiles[0]["id"], other_shared_profiles[0]["id"]}
        for number in range(3, 31):
            referral = login(client, "EXPERT-5834", f"referral-{number:02d}@example.org")
            profiles = client.get("/api/profiles", headers=referral).json()["profiles"]
            assert len(profiles) == 1
            allocated_ids.add(profiles[0]["id"])
        assert allocated_ids == {f"patient-{number}" for number in range(1, 31)}

        exhausted = client.post(
            "/api/auth/login",
            json={
                "email": "referral-31@example.org",
                "access_code": TEST_ACCESS_CODES["EXPERT-5834"],
            },
        )
        assert exhausted.status_code == 409
        assert "30 registrations" in exhausted.json()["detail"]

        second_claim = client.post(
            "/api/auth/login",
            json={
                "email": "another-second-expert@example.org",
                "access_code": TEST_ACCESS_CODES["EXPERT-9271"],
            },
        )
        assert second_claim.status_code == 409

        unknown_login = client.post(
            "/api/auth/login",
            json={
                "email": "unknown@example.org",
                "access_code": "unknown-access-code",
            },
        )
        assert unknown_login.status_code == 401


def test_leena_email_receives_reserved_patient_and_preserved_progress(monkeypatch, tmp_path):
    with load_client(monkeypatch, tmp_path) as client:
        app_module = sys.modules["server.app"]
        now = "2026-09-28T10:00:00.000Z"
        study_id = "legacy-leena-study"
        with app_module.database() as connection:
            connection.execute(
                """
                INSERT INTO participants(
                    participant_code, created_at, last_seen_at, cohort_code,
                    assigned_profiles_json, profile_completed
                ) VALUES (?, ?, ?, ?, ?, 0)
                """,
                (
                    app_module.EXPERT_CODES[0],
                    now,
                    now,
                    app_module.EXPERT_CODES[0],
                    json.dumps(app_module.LEGACY_PARTICIPANT_PROFILE_IDS[app_module.EXPERT_CODES[0]]),
                ),
            )
            connection.execute(
                """
                INSERT INTO studies(
                    id, participant_code, profile_id, status, created_at, updated_at
                ) VALUES (?, ?, ?, 'active', ?, ?)
                """,
                (
                    study_id,
                    app_module.EXPERT_CODES[0],
                    app_module.LEENA_PROFILE_ID,
                    now,
                    now,
                ),
            )
            for index, (label, method) in enumerate(
                zip(app_module.PANEL_LABELS, app_module.METHOD_KEYS)
            ):
                panel_id = f"legacy-leena-panel-{index}"
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
                        label,
                        method,
                        index,
                        now if index < 3 else None,
                        "expert_ended" if index < 3 else None,
                    ),
                )
                if index == 0:
                    connection.execute(
                        """
                        INSERT INTO panel_messages(
                            panel_id, role, content, client_message_id, created_at
                        ) VALUES (?, 'therapist', ?, ?, ?)
                        """,
                        (panel_id, "Preserved Leena transcript", "legacy-leena-message", now),
                    )
                if index < 3:
                    connection.execute(
                        """
                        INSERT INTO panel_ratings(
                            id, panel_id, participant_code, scores_json, total_score,
                            comments, submitted_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            f"legacy-leena-rating-{index}",
                            panel_id,
                            app_module.EXPERT_CODES[0],
                            "{}",
                            30 + index,
                            f"Preserved rating {index + 1}",
                            now,
                        ),
                    )

        headers = login(client, "EXPERT-5834", "Dr.LeEnA@example.org")
        profiles = client.get("/api/profiles", headers=headers).json()["profiles"]
        assert len(profiles) == 1
        assert profiles[0]["id"] == app_module.LEENA_PROFILE_ID
        assert profiles[0]["display_number"] == 1
        assert profiles[0]["display_name"] == "Patient 1"
        assert profiles[0]["study"]["completed_sessions"] == 3

        resumed = client.get(f"/api/studies/{study_id}", headers=headers)
        assert resumed.status_code == 200
        study = resumed.json()["study"]
        assert study["profile"]["display_number"] == 1
        assert study["completed_sessions"] == 3
        assert study["panels"][0]["messages"][0]["content"] == "Preserved Leena transcript"
        assert study["panels"][0]["rating"]["total_score"] == 30

        with app_module.database() as connection:
            participant = connection.execute(
                "SELECT participant_code FROM participants WHERE lower(email) = ?",
                ("dr.leena@example.org",),
            ).fetchone()
            migrated_study = connection.execute(
                "SELECT participant_code FROM studies WHERE id = ?", (study_id,)
            ).fetchone()
            migrated_ratings = connection.execute(
                """
                SELECT DISTINCT r.participant_code
                FROM panel_ratings AS r
                JOIN study_panels AS p ON p.id = r.panel_id
                WHERE p.study_id = ?
                """,
                (study_id,),
            ).fetchall()
        assert migrated_study["participant_code"] == participant["participant_code"]
        assert [row["participant_code"] for row in migrated_ratings] == [
            participant["participant_code"]
        ]


def test_leena_name_match_is_case_insensitive(monkeypatch, tmp_path):
    with load_client(monkeypatch, tmp_path) as client:
        first = client.post(
            "/api/auth/login",
            json={
                "email": "reserved-case@example.org",
                "access_code": TEST_ACCESS_CODES["EXPERT-5834"],
            },
        )
        assert first.status_code == 200
        headers = {"Authorization": f"Bearer {first.json()['token']}"}
        saved = client.post(
            "/api/auth/profile",
            headers=headers,
            json={
                "name": "Dr LEENA Example",
                "role": "Clinical psychologist",
                "institution": "Test Institute",
                "latest_degree": "PhD",
                "years_experience": 12,
            },
        )
        assert saved.status_code == 200
        profiles = client.get("/api/profiles", headers=headers).json()["profiles"]
        assert len(profiles) == 1
        assert profiles[0]["id"] == sys.modules["server.app"].LEENA_PROFILE_ID
        assert profiles[0]["display_name"] == "Patient 1"


def test_legacy_finished_study_is_migrated_for_sequential_ratings(monkeypatch, tmp_path):
    database_path = tmp_path / "study.sqlite3"
    connection = sqlite3.connect(database_path)
    connection.executescript(
        """
        CREATE TABLE participants (
            participant_code TEXT PRIMARY KEY,
            created_at TEXT NOT NULL,
            last_seen_at TEXT NOT NULL
        );
        CREATE TABLE studies (
            id TEXT PRIMARY KEY,
            participant_code TEXT NOT NULL,
            profile_id TEXT NOT NULL,
            status TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            finished_at TEXT,
            UNIQUE (participant_code, profile_id)
        );
        CREATE TABLE study_panels (
            id TEXT PRIMARY KEY,
            study_id TEXT NOT NULL,
            label TEXT NOT NULL,
            method_key TEXT NOT NULL,
            display_order INTEGER NOT NULL,
            UNIQUE (study_id, label),
            UNIQUE (study_id, method_key)
        );
        CREATE TABLE panel_messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            panel_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            client_message_id TEXT,
            created_at TEXT NOT NULL
        );
        CREATE TABLE panel_ratings (
            id TEXT PRIMARY KEY,
            panel_id TEXT NOT NULL UNIQUE,
            participant_code TEXT NOT NULL,
            scores_json TEXT NOT NULL,
            total_score INTEGER NOT NULL,
            comments TEXT NOT NULL DEFAULT '',
            submitted_at TEXT NOT NULL
        );
        INSERT INTO participants VALUES ('EXPERT-5834', 'old', 'old');
        INSERT INTO studies VALUES (
            'legacy-study', 'EXPERT-5834', 'patient-1', 'finished', 'old', 'old', 'old'
        );
        """
    )
    for index, (label, method) in enumerate(
        zip(
            [f"Therapist {letter}" for letter in "ABCDEF"],
            ["prompting", "proact", "archer", "aria", "sweet_rl", "topas"],
        )
    ):
        connection.execute(
            "INSERT INTO study_panels VALUES (?, 'legacy-study', ?, ?, ?)",
            (f"legacy-panel-{index}", label, method, index),
        )
    connection.execute(
        """
        INSERT INTO panel_messages(panel_id, role, content, client_message_id, created_at)
        VALUES ('legacy-panel-0', 'therapist', 'Stored legacy transcript', 'legacy-message', 'old')
        """
    )
    connection.execute(
        """
        INSERT INTO panel_ratings(
            id, panel_id, participant_code, scores_json, total_score, comments, submitted_at
        ) VALUES (
            'legacy-rating', 'legacy-panel-0', 'EXPERT-5834', '{}', 33, 'Stored rating', 'old'
        )
        """
    )
    connection.commit()
    connection.close()

    with load_client(monkeypatch, tmp_path):
        pass

    connection = sqlite3.connect(database_path)
    connection.row_factory = sqlite3.Row
    panel_columns = {row["name"] for row in connection.execute("PRAGMA table_info(study_panels)")}
    participant_columns = {row["name"] for row in connection.execute("PRAGMA table_info(participants)")}
    study = connection.execute("SELECT * FROM studies WHERE id = 'legacy-study'").fetchone()
    participant = connection.execute(
        "SELECT * FROM participants WHERE participant_code = 'EXPERT-5834'"
    ).fetchone()
    legacy_message = connection.execute(
        "SELECT content FROM panel_messages WHERE client_message_id = 'legacy-message'"
    ).fetchone()
    legacy_rating = connection.execute(
        "SELECT total_score, comments FROM panel_ratings WHERE id = 'legacy-rating'"
    ).fetchone()
    connection.close()
    assert "ended_at" in panel_columns
    assert {"email", "cohort_code", "assigned_profiles_json", "profile_completed"} <= participant_columns
    assert study["status"] == "active"
    assert study["finished_at"] is None
    assert participant["cohort_code"] == "EXPERT-5834"
    assert participant["email"] is None
    assert legacy_message["content"] == "Stored legacy transcript"
    assert legacy_rating["total_score"] == 33
    assert legacy_rating["comments"] == "Stored rating"


def test_legacy_five_profile_environment_is_expanded(monkeypatch):
    from server import profiles

    monkeypatch.setenv(
        "STUDY_EXPERT_1_PROFILE_IDS",
        "patient_act_001,patient_act_002,patient_act_003,patient_act_021,patient_act_022",
    )
    monkeypatch.setenv(
        "STUDY_EXPERT_2_PROFILE_IDS",
        "patient_act_004,patient_act_007,patient_act_016,patient_act_023,patient_act_026",
    )
    assert len(profiles._selected_group(1)) == 20
    assert len(profiles._selected_group(2)) == 20
