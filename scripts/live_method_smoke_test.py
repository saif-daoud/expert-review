from __future__ import annotations

import argparse
import hashlib
import hmac
import http.client
import json
import os
import random
import socket
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
ABLATION_METHODS = (
    "topas_expert_components",
    "topas_chatgpt_components",
    "topas_no_rl",
    "topas_no_conv_state",
    "topas_full",
    "theramind",
)


class ApiError(RuntimeError):
    def __init__(self, status: int, payload: object):
        super().__init__(f"HTTP {status}: {payload}")
        self.status = status
        self.payload = payload


class StudyApi:
    def __init__(self, base_url: str, evaluation_track: str):
        self.base_url = base_url.rstrip("/")
        self.evaluation_track = evaluation_track
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
        for attempt in range(3):
            try:
                with urllib.request.urlopen(request, timeout=60) as response:
                    return json.load(response)
            except urllib.error.HTTPError as exc:
                try:
                    payload = json.load(exc)
                except (json.JSONDecodeError, UnicodeDecodeError):
                    payload = exc.read().decode("utf-8", errors="replace")
                raise ApiError(exc.code, payload) from exc
            except (urllib.error.URLError, http.client.RemoteDisconnected, socket.timeout):
                if attempt == 2:
                    raise
                time.sleep(2 ** attempt)
        raise RuntimeError("unreachable")


def latest_panel(api: StudyApi, study_id: str, panel_id: str) -> tuple[dict, dict]:
    study = api.request(
        "GET", f"/api/studies/{study_id}?evaluation_track={api.evaluation_track}"
    )["study"]
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


def load_env_file(path: str | None) -> None:
    if not path:
        return
    for raw_line in open(path, encoding="utf-8"):
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        os.environ.setdefault(key.strip(), value)


def participant_code_for_email(email: str) -> str:
    return f"P-{hashlib.sha256(email.strip().lower().encode('utf-8')).hexdigest()[:32]}"


def private_method_order(
    token_secret: str, participant_code: str, profile_id: str, evaluation_track: str
) -> list[str]:
    digest = hmac.new(
        token_secret.encode("utf-8"),
        f"{participant_code}|{profile_id}|{evaluation_track}|panel-order".encode("utf-8"),
        hashlib.sha256,
    ).digest()
    methods = list(ABLATION_METHODS)
    random.Random(int.from_bytes(digest, "big")).shuffle(methods)
    return methods


def login(api: StudyApi, args: argparse.Namespace, access_code: str) -> None:
    body = {"email": args.email, "access_code": access_code}
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
                "access_code": access_code,
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
        description="Exercise all six real focused therapist sessions through the public study API."
    )
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--access-code")
    parser.add_argument(
        "--env-file",
        help="Read STUDY_EXPERT_ACCESS_CODE and STUDY_TOKEN_SECRET without putting secrets on the command line.",
    )
    parser.add_argument(
        "--email",
        default=f"ablation-smoke-{int(time.time())}-{uuid.uuid4().hex[:8]}@example.org",
    )
    parser.add_argument(
        "--participant-code",
        help="Compatibility value for an older deployed API that still requires it.",
    )
    parser.add_argument("--profile-id")
    parser.add_argument("--evaluation-track", default="ablations", choices=("ablations",))
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument(
        "--patient-message",
        default=(
            "This problem has been affecting my daily life, and I am not sure how to handle it. "
            "I would like help understanding what I can do."
        ),
    )
    args = parser.parse_args()
    load_env_file(args.env_file)
    access_code = args.access_code or os.getenv("STUDY_EXPERT_ACCESS_CODE", "")
    token_secret = os.getenv("STUDY_TOKEN_SECRET", "")
    if not access_code:
        raise RuntimeError("Provide --access-code or STUDY_EXPERT_ACCESS_CODE in --env-file")
    if not token_secret:
        raise RuntimeError("STUDY_TOKEN_SECRET is required to report the private method mapping")

    api = StudyApi(args.base_url, args.evaluation_track)
    health = api.request("GET", "/api/health")
    if health.get("status") != "ok" or health.get("inference", {}).get("mode") != "real":
        raise RuntimeError(f"The live API is not ready for real inference: {health}")
    login(api, args, access_code)
    profile = select_profile(api, args.profile_id)
    study = api.request(
        "POST",
        "/api/studies",
        {"profile_id": profile["id"], "evaluation_track": args.evaluation_track},
    )["study"]
    participant_code = participant_code_for_email(args.email)
    method_order = private_method_order(
        token_secret, participant_code, profile["id"], args.evaluation_track
    )
    print(
        f"email={args.email} study={study['id']} profile={profile['id']} "
        f"track={args.evaluation_track} panels={len(study['panels'])}",
        flush=True,
    )

    results: list[dict] = []
    while study["completed_sessions"] < study["total_sessions"]:
        panel = next(item for item in study["panels"] if item["id"] == study["current_panel_id"])
        panel_id = panel["id"]
        method = method_order[int(panel["display_order"])]
        print(f"starting {panel['label']} method={method}", flush=True)
        if not panel["messages"]:
            if (panel.get("job") or {}).get("status") not in {"queued", "running"}:
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

        therapist_messages = [
            item["content"] for item in panel["messages"] if item["role"] == "therapist"
        ]
        if len(therapist_messages) < 2:
            api.request(
                "POST",
                f"/api/studies/{study['id']}/panels/{panel_id}/messages",
                {
                    "content": args.patient_message,
                    "client_message_id": f"smoke-message-{uuid.uuid4()}",
                },
            )
            study, panel, reply_seconds = wait_for_generation(
                api, study["id"], panel_id, args.timeout
            )
            therapist_messages = [
                item["content"] for item in panel["messages"] if item["role"] == "therapist"
            ]
        else:
            reply_seconds = 0.0
        if len(therapist_messages) < 2:
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
                "method": method,
                "opening": therapist_messages[0],
                "reply": therapist_messages[-1],
                "opening_seconds": round(first_seconds, 1),
                "reply_seconds": round(reply_seconds, 1),
                "message_count": len(panel["messages"]),
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
