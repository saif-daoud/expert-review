from __future__ import annotations

import importlib
import json
import sys

from fastapi.testclient import TestClient

from server.llm_relay import RelayUpstreamResponse


RELAY_TOKEN = "test-relay-token-that-is-at-least-thirty-two-characters"


def load_client(monkeypatch):
    monkeypatch.setenv("SIMULATOR_RELAY_TOKEN", RELAY_TOKEN)
    monkeypatch.setenv("SIMULATOR_RELAY_UPSTREAM_API_KEY", "test-provider-key")
    monkeypatch.setenv("SIMULATOR_RELAY_UPSTREAM_BASE_URL", "https://provider.example/v1")
    monkeypatch.setenv("SIMULATOR_RELAY_MODEL", "gpt-4.1")
    sys.modules.pop("server.relay_app", None)
    module = importlib.import_module("server.relay_app")
    return module, TestClient(module.app)


def request_payload(**updates):
    payload = {
        "model": "gpt-4.1",
        "instructions": "Return JSON.",
        "input": "Hello",
        "max_output_tokens": 500,
        "text": {"format": {"type": "json_schema", "name": "result", "strict": True, "schema": {}}},
        "safety_identifier": "anonymous-id",
        "store": True,
    }
    payload.update(updates)
    return payload


def test_relay_requires_its_own_bearer_token(monkeypatch):
    _, client = load_client(monkeypatch)
    with client:
        health = client.get("/api/health")
        missing = client.post("/api/responses", json=request_payload())
        wrong = client.post(
            "/api/responses",
            headers={"Authorization": "Bearer wrong-token"},
            json=request_payload(),
        )
    assert missing.status_code == 401
    assert wrong.status_code == 401
    assert health.json() == {
        "status": "ok",
        "service": "cbt-simulator-relay",
        "configured": True,
        "model": "gpt-4.1",
    }


def test_relay_allowlists_and_forwards_responses_request(monkeypatch):
    module, client = load_client(monkeypatch)
    captured = {}

    def fake_forward(payload, config):
        captured["payload"] = payload
        captured["config"] = config
        return RelayUpstreamResponse(
            status_code=200,
            body=json.dumps({"output_text": "{\"value\":\"ok\"}"}).encode(),
        )

    monkeypatch.setattr(module, "forward_response_payload", fake_forward)
    with client:
        response = client.post(
            "/api/responses",
            headers={"Authorization": f"Bearer {RELAY_TOKEN}"},
            json=request_payload(),
        )
    assert response.status_code == 200
    assert response.json()["output_text"] == '{"value":"ok"}'
    assert captured["payload"]["model"] == "gpt-4.1"
    assert captured["payload"]["store"] is False
    assert captured["config"].upstream_base_url == "https://provider.example/v1"


def test_relay_rejects_other_models_and_fields(monkeypatch):
    _, client = load_client(monkeypatch)
    headers = {"Authorization": f"Bearer {RELAY_TOKEN}"}
    with client:
        other_model = client.post(
            "/api/responses", headers=headers, json=request_payload(model="gpt-5.1")
        )
        unexpected = client.post(
            "/api/responses", headers=headers, json=request_payload(tools=[])
        )
    assert other_model.status_code == 422
    assert unexpected.status_code == 422


def test_relay_preserves_provider_error_shape(monkeypatch):
    module, client = load_client(monkeypatch)

    monkeypatch.setattr(
        module,
        "forward_response_payload",
        lambda _payload, _config: RelayUpstreamResponse(
            status_code=429,
            body=b'{"error":{"message":"Rate limited"}}',
        ),
    )
    with client:
        response = client.post(
            "/api/responses",
            headers={"Authorization": f"Bearer {RELAY_TOKEN}"},
            json=request_payload(),
        )
    assert response.status_code == 429
    assert response.json() == {"error": {"message": "Rate limited"}}
