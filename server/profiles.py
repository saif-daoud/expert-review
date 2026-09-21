"""PatientAct profile loading and browser-safe study views."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


SERVER_DIR = Path(__file__).resolve().parent


def _discover_project_root() -> Path:
    candidates = (SERVER_DIR.parent, *SERVER_DIR.parents)
    return next(
        (candidate for candidate in candidates if (candidate / "simulations" / "data" / "patient_act.json").is_file()),
        SERVER_DIR.parent,
    )


DEFAULT_PROJECT_ROOT = _discover_project_root()
DEFAULT_PROFILE_IDS = (
    "patient_act_001",
    "patient_act_002",
    "patient_act_003",
    "patient_act_021",
    "patient_act_022",
)

CARD_DESCRIPTIONS = {
    "patient_act_001": "Persistent fear that a serious illness has been missed despite reassuring tests.",
    "patient_act_002": "Longstanding fear of crossing bridges, with intense anticipatory anxiety.",
    "patient_act_003": "Recent, unexplained panic attacks amid academic and life-transition pressure.",
    "patient_act_021": "Low mood, withdrawal, and guilt about being a burden at home.",
    "patient_act_022": "Depression, poor sleep, and loss of direction after university graduation.",
}


def _project_root() -> Path:
    return Path(os.getenv("TOPAS_PROJECT_ROOT", str(DEFAULT_PROJECT_ROOT))).expanduser().resolve()


def _profile_path() -> Path:
    configured = os.getenv("STUDY_PROFILES_PATH")
    if configured:
        return Path(configured).expanduser().resolve()
    return _project_root() / "simulations" / "data" / "patient_act.json"


def _selected_ids() -> tuple[str, ...]:
    raw = os.getenv("STUDY_PROFILE_IDS", ",".join(DEFAULT_PROFILE_IDS))
    values = tuple(value.strip() for value in raw.split(",") if value.strip())
    if len(values) != 5 or len(set(values)) != 5:
        raise RuntimeError("STUDY_PROFILE_IDS must contain exactly five unique PatientAct profile IDs.")
    return values


def _condition_label(value: Any) -> str:
    if isinstance(value, list):
        value = value[0] if value else ""
    labels = {
        "anxiety_disorder": "Anxiety disorder",
        "depression": "Depression",
    }
    key = str(value or "").strip().lower()
    return labels.get(key, key.replace("_", " ").title() or "Mental health concern")


def _first_paragraph(value: Any) -> str:
    text = _clean_text(value)
    return text.split("\n\n", maxsplit=1)[0]


def _clean_text(value: Any) -> str:
    text = str(value or "").strip()
    replacements = {
        "â€™": "’",
        "â€˜": "‘",
        "â€œ": "“",
        "â€": "”",
        "â€“": "–",
        "â€”": "—",
        "Õ³Õ¡Õ¶Õ¡ÕºÕ¡Ö€Õ°": "route",
    }
    for broken, replacement in replacements.items():
        text = text.replace(broken, replacement)
    return text


def load_profiles() -> dict[str, dict[str, Any]]:
    path = _profile_path()
    if not path.is_file():
        raise RuntimeError(
            f"PatientAct data was not found at {path}. Set TOPAS_PROJECT_ROOT or STUDY_PROFILES_PATH."
        )
    with path.open("r", encoding="utf-8") as handle:
        raw_profiles = json.load(handle)
    by_id = {str(item.get("profile_id")): item for item in raw_profiles}

    selected: dict[str, dict[str, Any]] = {}
    for index, source_id in enumerate(_selected_ids(), start=1):
        if source_id not in by_id:
            raise RuntimeError(f"Configured PatientAct profile does not exist: {source_id}")
        raw = by_id[source_id]
        public_id = f"patient-{index}"
        selected[public_id] = {
            "id": public_id,
            "source_id": source_id,
            "display_name": f"Patient {index}",
            "condition": _condition_label(raw.get("disease_key") or raw.get("presenting_conditions")),
            "short_description": CARD_DESCRIPTIONS.get(
                source_id,
                str(raw.get("summary", "")).split(".", maxsplit=1)[0].strip() + ".",
            ),
            "summary": _clean_text(raw.get("summary")),
            "current_context": _first_paragraph(raw.get("current_stressor")),
            "relevant_history": _clean_text(raw.get("relevant_history")),
            "coping_strategies": [_clean_text(item) for item in raw.get("coping_strategies", []) if _clean_text(item)],
            "conversational_style": _clean_text(raw.get("conversational_style", "plain")),
            "role_guidance": [
                "Stay in character and answer naturally as this patient.",
                "Reveal details gradually when they fit the therapist's questions.",
                "Do not invent a real identity or share your own personal information.",
            ],
        }
    return selected


def profile_card(profile: dict[str, Any], study: dict[str, Any] | None = None) -> dict[str, Any]:
    result = {
        "id": profile["id"],
        "display_name": profile["display_name"],
        "condition": profile["condition"],
        "short_description": profile["short_description"],
    }
    if study:
        result["study"] = study
    return result


def public_profile(profile: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in profile.items() if key != "source_id"}
