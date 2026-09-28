from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
import uuid


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


class ApiError(RuntimeError):
    def __init__(self, status: int, payload: object):
        super().__init__(f"HTTP {status}: {payload}")
        self.status = status
        self.payload = payload


class StudyApi:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self.token = ""

    def request(self, method: str, path: str, body: dict | None = None) -> dict:
        headers = {
            "Accept": "application/json",
            "ngrok-skip-browser-warning": "true",
        }
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        data = None
        if body is not None:
            headers["Content-Type"] = "application/json"
            data = json.dumps(body).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}{path}", data=data, headers=headers, method=method
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return json.load(response)
        except urllib.error.HTTPError as exc:
            try:
                payload = json.load(exc)
            except (json.JSONDecodeError, UnicodeDecodeError):
                payload = exc.read().decode("utf-8", errors="replace")
            raise ApiError(exc.code, payload) from exc


def latest_panel(api: StudyApi, study_id: str, panel_id: str) -> tuple[dict, dict]:
    study = api.request("GET", f"/api/studies/{study_id}")["study"]
    panel = next(item for item in study["panels"] if item["id"] == panel_id)
    return study, panel


def wait_for_generation(
    api: StudyApi, study_id: str, panel_id: str, timeout_seconds: int
) -> tuple[dict, dict, float]:
    started = time.monotonic()
    retried = False
    while time.monotonic() - started < timeout_seconds:
        study, panel = latest_panel(api, study_id, panel_id)
        job = panel.get("job") or {}
        status = job.get("status")
        if status == "completed":
            return study, panel, time.monotonic() - started
        if status == "failed":
            if retried:
                raise RuntimeError(f"{panel['label']} failed after one retry")
            api.request(
                "POST",
                f"/api/studies/{study_id}/panels/{panel_id}/retry",
                {"client_request_id": f"smoke-retry-{uuid.uuid4()}"},
            )
            retried = True
        time.sleep(5)
    raise TimeoutError(f"{panel_id} did not finish within {timeout_seconds} seconds")


def login(api: StudyApi, args: argparse.Namespace) -> None:
    body = {"email": args.email, "access_code": args.access_code}
    try:
        result = api.request("POST", "/api/auth/login", body)
    except ApiError as exc:
        missing_old_participant = (
            exc.status == 422
            and args.participant_code
            and "participant_code" in json.dumps(exc.payload)
        )
        if not missing_old_participant:
            raise
        result = api.request(
            "POST",
            "/api/auth/login",
            {
                "participant_code": args.participant_code,
                "access_code": args.access_code,
            },
        )
    api.token = result["token"]
    if result.get("profile_required"):
        api.request(
            "POST",
            "/api/auth/profile",
            {
                "name": "Automated Method Smoke Test",
                "role": "Research test account",
                "institution": "QCRI",
                "latest_degree": "N/A",
                "years_experience": 0,
            },
        )


def select_profile(api: StudyApi, requested_id: str | None) -> dict:
    profiles = api.request("GET", "/api/profiles")["profiles"]
    if requested_id:
        return next(profile for profile in profiles if profile["id"] == requested_id)
    unused = next((profile for profile in profiles if not profile.get("study")), None)
    if unused is None:
        raise RuntimeError("No unused patient is available for the smoke test")
    return unused


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Exercise all six real therapist sessions through the public study API."
    )
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--access-code", required=True)
    parser.add_argument("--email", default="method-smoke-test@example.org")
    parser.add_argument(
        "--participant-code",
        help="Compatibility value for an older deployed API that still requires it.",
    )
    parser.add_argument("--profile-id")
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument(
        "--patient-message",
        default=(
            "I feel tense around authority figures, especially when they become angry. "
            "I avoid speaking up and my heart starts racing."
        ),
    )
    args = parser.parse_args()

    api = StudyApi(args.base_url)
    health = api.request("GET", "/api/health")
    if health.get("status") != "ok" or health.get("inference", {}).get("mode") != "real":
        raise RuntimeError(f"The live API is not ready for real inference: {health}")
    login(api, args)
    profile = select_profile(api, args.profile_id)
    study = api.request("POST", "/api/studies", {"profile_id": profile["id"]})["study"]
    print(f"study={study['id']} profile={profile['id']} panels={len(study['panels'])}", flush=True)

    results: list[dict] = []
    while study["completed_sessions"] < study["total_sessions"]:
        panel = next(item for item in study["panels"] if item["id"] == study["current_panel_id"])
        panel_id = panel["id"]
        before = len(panel["messages"])
        if not panel["messages"]:
            api.request(
                "POST",
                f"/api/studies/{study['id']}/panels/{panel_id}/start",
                {"client_request_id": f"smoke-start-{uuid.uuid4()}"},
            )
            study, panel, first_seconds = wait_for_generation(
                api, study["id"], panel_id, args.timeout
            )
        else:
            first_seconds = 0.0
        if not panel["messages"] or panel["messages"][-1]["role"] != "therapist":
            raise RuntimeError(f"{panel['label']} did not produce its opening response")

        api.request(
            "POST",
            f"/api/studies/{study['id']}/panels/{panel_id}/messages",
            {
                "content": args.patient_message,
                "client_message_id": f"smoke-message-{uuid.uuid4()}",
            },
        )
        study, panel, reply_seconds = wait_for_generation(api, study["id"], panel_id, args.timeout)
        therapist_messages = [
            item["content"] for item in panel["messages"] if item["role"] == "therapist"
        ]
        if len(panel["messages"]) < before + 3 or len(therapist_messages) < 2:
            raise RuntimeError(f"{panel['label']} did not complete a full response cycle")
        leaked_labels = [
            label
            for label in ("Assistant:", "Human:", "Patient:", "Therapist:")
            if label.casefold() in therapist_messages[-1].casefold()
        ]

        ended = api.request(
            "POST",
            f"/api/studies/{study['id']}/panels/{panel_id}/end",
            {"client_request_id": f"smoke-end-{uuid.uuid4()}"},
        )["study"]
        rated = api.request(
            "POST",
            f"/api/studies/{study['id']}/panels/{panel_id}/rating",
            {
                "scores": {key: 3 for key in CTRS_KEYS},
                "comments": "Automated live method smoke test; not research data.",
            },
        )["study"]
        results.append(
            {
                "label": panel["label"],
                "opening_seconds": round(first_seconds, 1),
                "reply_seconds": round(reply_seconds, 1),
                "message_count": len(panel["messages"]),
                "reply_preview": therapist_messages[-1][:160],
                "speaker_label_leaks": leaked_labels,
            }
        )
        study = rated
        print(json.dumps(results[-1], ensure_ascii=False), flush=True)
        if ended["id"] != study["id"]:
            raise RuntimeError("Study identity changed while rating a session")

    print(json.dumps({"study_id": study["id"], "status": study["status"], "results": results}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        raise
