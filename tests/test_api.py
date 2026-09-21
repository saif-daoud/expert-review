from __future__ import annotations

import importlib
import sys
import time
from pathlib import Path

from fastapi.testclient import TestClient


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def load_client(monkeypatch, tmp_path):
    monkeypatch.setenv("STUDY_DB_PATH", str(tmp_path / "study.sqlite3"))
    monkeypatch.setenv("STUDY_ACCESS_CODE", "test-access-code")
    monkeypatch.setenv("STUDY_TOKEN_SECRET", "test-token-secret-that-is-not-used-in-production")
    monkeypatch.setenv("STUDY_ALLOWED_ORIGINS", "http://127.0.0.1:5500")
    monkeypatch.setenv("STUDY_SERVE_FRONTEND", "false")
    monkeypatch.setenv("STUDY_INFERENCE_MODE", "static")
    monkeypatch.setenv("TOPAS_PROJECT_ROOT", str(PROJECT_ROOT))
    sys.modules.pop("server.app", None)
    module = importlib.import_module("server.app")
    return TestClient(module.app)


def login(client: TestClient, participant_code: str) -> dict[str, str]:
    response = client.post(
        "/api/auth/login",
        json={"participant_code": participant_code, "access_code": "test-access-code"},
    )
    assert response.status_code == 200
    return {"Authorization": f"Bearer {response.json()['token']}"}


def wait_for_panel(client: TestClient, headers: dict[str, str], study_id: str, panel_id: str) -> dict:
    deadline = time.monotonic() + 4
    while time.monotonic() < deadline:
        study = client.get(f"/api/studies/{study_id}", headers=headers).json()["study"]
        panel = next(item for item in study["panels"] if item["id"] == panel_id)
        if panel["job"]["status"] in {"completed", "failed"}:
            return panel
        time.sleep(0.03)
    raise AssertionError("Inference job did not finish")


def test_full_six_panel_flow(monkeypatch, tmp_path):
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

        headers = login(client, "EXPERT-01")
        profiles = client.get("/api/profiles", headers=headers).json()["profiles"]
        assert len(profiles) == 5
        assert set(profiles[0]) == {"id", "display_name", "condition", "short_description"}

        created = client.post("/api/studies", headers=headers, json={"profile_id": profiles[0]["id"]})
        assert created.status_code == 200
        study = created.json()["study"]
        assert len(study["panels"]) == 6
        assert [panel["label"] for panel in study["panels"]] == [f"Therapist {letter}" for letter in "ABCDEF"]
        assert "method_key" not in str(study)

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

        finished = client.post(f"/api/studies/{study['id']}/finish", headers=headers)
        assert finished.status_code == 200
        assert finished.json()["study"]["status"] == "finished"


def test_studies_are_private_and_mapping_is_stable(monkeypatch, tmp_path):
    with load_client(monkeypatch, tmp_path) as client:
        first = login(client, "EXPERT-01")
        second = login(client, "EXPERT-02")
        profile_id = client.get("/api/profiles", headers=first).json()["profiles"][0]["id"]
        first_create = client.post("/api/studies", headers=first, json={"profile_id": profile_id}).json()["study"]
        second_create = client.post("/api/studies", headers=first, json={"profile_id": profile_id}).json()["study"]
        assert first_create["id"] == second_create["id"]

        forbidden = client.get(f"/api/studies/{first_create['id']}", headers=second)
        assert forbidden.status_code == 404
