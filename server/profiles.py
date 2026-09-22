"""PatientAct profile loading and browser-safe study views."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


SERVER_DIR = Path(__file__).resolve().parent
DEFAULT_PROFILE_GROUPS = (
    (
        "patient_act_001",
        "patient_act_002",
        "patient_act_003",
        "patient_act_021",
        "patient_act_022",
        "patient_act_005",
        "patient_act_008",
        "patient_act_010",
        "patient_act_012",
        "patient_act_014",
        "patient_act_017",
        "patient_act_019",
        "patient_act_024",
        "patient_act_027",
        "patient_act_029",
        "patient_act_031",
        "patient_act_033",
        "patient_act_035",
        "patient_act_037",
        "patient_act_039",
    ),
    (
        "patient_act_004",
        "patient_act_007",
        "patient_act_016",
        "patient_act_023",
        "patient_act_026",
        "patient_act_006",
        "patient_act_009",
        "patient_act_011",
        "patient_act_013",
        "patient_act_015",
        "patient_act_018",
        "patient_act_020",
        "patient_act_025",
        "patient_act_028",
        "patient_act_030",
        "patient_act_032",
        "patient_act_034",
        "patient_act_036",
        "patient_act_038",
        "patient_act_040",
    ),
)

# The first ten positions preserve the IDs used by the initial deployment.
# New profiles are appended so existing SQLite studies keep the same patient.
PUBLIC_PROFILE_ORDER = (
    "patient_act_001",
    "patient_act_002",
    "patient_act_003",
    "patient_act_021",
    "patient_act_022",
    "patient_act_004",
    "patient_act_007",
    "patient_act_016",
    "patient_act_023",
    "patient_act_026",
    "patient_act_005",
    "patient_act_006",
    "patient_act_008",
    "patient_act_009",
    "patient_act_010",
    "patient_act_011",
    "patient_act_012",
    "patient_act_013",
    "patient_act_014",
    "patient_act_015",
    "patient_act_017",
    "patient_act_018",
    "patient_act_019",
    "patient_act_020",
    "patient_act_024",
    "patient_act_025",
    "patient_act_027",
    "patient_act_028",
    "patient_act_029",
    "patient_act_030",
    "patient_act_031",
    "patient_act_032",
    "patient_act_033",
    "patient_act_034",
    "patient_act_035",
    "patient_act_036",
    "patient_act_037",
    "patient_act_038",
    "patient_act_039",
    "patient_act_040",
)
PUBLIC_PROFILE_IDS = {
    source_id: f"patient-{index}"
    for index, source_id in enumerate(PUBLIC_PROFILE_ORDER, start=1)
}

CARD_DESCRIPTIONS = {
    "patient_act_001": "Persistent fear that a serious illness has been missed despite reassuring tests.",
    "patient_act_002": "Longstanding fear of crossing bridges, with intense anticipatory anxiety.",
    "patient_act_003": "Recent, unexplained panic attacks amid academic and life-transition pressure.",
    "patient_act_021": "Low mood, withdrawal, and guilt about being a burden at home.",
    "patient_act_022": "Depression, poor sleep, and loss of direction after university graduation.",
    "patient_act_004": "Anxiety and strong physical reactions around emotionally explosive authority figures.",
    "patient_act_007": "Severe public-speaking anxiety linked to a humiliating childhood experience.",
    "patient_act_016": "Recurring panic attacks, fear of dying, and avoidance of being alone.",
    "patient_act_023": "Low mood, exhaustion, and unresolved grief following the death of a parent.",
    "patient_act_026": "Persistent grief, isolation, and difficulty resuming daily life after the loss of a longtime pet.",
}


def _profile_path() -> Path:
    configured = os.getenv("STUDY_PROFILES_PATH")
    if configured:
        return Path(configured).expanduser().resolve()
    return SERVER_DIR / "data" / "patient_act.json"


def _selected_group(group_number: int) -> tuple[str, ...]:
    variable = f"STUDY_EXPERT_{group_number}_PROFILE_IDS"
    raw = os.getenv(variable, ",".join(DEFAULT_PROFILE_GROUPS[group_number - 1]))
    values = tuple(value.strip() for value in raw.split(",") if value.strip())
    legacy_values = DEFAULT_PROFILE_GROUPS[group_number - 1][:5]
    if values == legacy_values:
        # Expand the exact five-profile configuration used by the first
        # deployment, allowing its existing private .env to upgrade safely.
        values = DEFAULT_PROFILE_GROUPS[group_number - 1]
    if len(values) != 20 or len(set(values)) != 20:
        raise RuntimeError(f"{variable} must contain exactly 20 unique PatientAct profile IDs.")
    return values


def _selected_groups() -> tuple[tuple[str, ...], tuple[str, ...]]:
    groups = (_selected_group(1), _selected_group(2))
    all_ids = groups[0] + groups[1]
    if len(set(all_ids)) != len(all_ids):
        raise RuntimeError("The two experts must have ten distinct PatientAct profile IDs.")
    return groups


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


def _card_description(source_id: str, summary: Any) -> str:
    if source_id in CARD_DESCRIPTIONS:
        return CARD_DESCRIPTIONS[source_id]
    sentence = _clean_text(summary).split(".", maxsplit=1)[0].strip()
    if len(sentence) > 155:
        sentence = sentence[:155].rsplit(" ", maxsplit=1)[0].rstrip(",;:") + "..."
        return sentence
    return sentence + "." if sentence else "Patient profile available."


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
            f"PatientAct data was not found at {path}. Set STUDY_PROFILES_PATH to override it."
        )
    with path.open("r", encoding="utf-8") as handle:
        raw_profiles = json.load(handle)
    by_id = {str(item.get("profile_id")): item for item in raw_profiles}

    selected: dict[str, dict[str, Any]] = {}
    for assignment_group, source_ids in enumerate(_selected_groups(), start=1):
        for display_number, source_id in enumerate(source_ids, start=1):
            if source_id not in by_id:
                raise RuntimeError(f"Configured PatientAct profile does not exist: {source_id}")
            if source_id not in PUBLIC_PROFILE_IDS:
                raise RuntimeError(f"PatientAct profile is missing a stable public ID: {source_id}")
            raw = by_id[source_id]
            public_id = PUBLIC_PROFILE_IDS[source_id]
            selected[public_id] = {
                "id": public_id,
                "source_id": source_id,
                "assignment_group": assignment_group,
                "display_number": display_number,
                "display_name": f"Patient {display_number}",
                "condition": _condition_label(raw.get("disease_key") or raw.get("presenting_conditions")),
                "short_description": _card_description(source_id, raw.get("summary")),
                "summary": _clean_text(raw.get("summary")),
                "current_context": _first_paragraph(raw.get("current_stressor")),
                "relevant_history": _clean_text(raw.get("relevant_history")),
                "coping_strategies": [
                    _clean_text(item) for item in raw.get("coping_strategies", []) if _clean_text(item)
                ],
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
        "display_number": profile["display_number"],
        "display_name": profile["display_name"],
        "condition": profile["condition"],
        "short_description": profile["short_description"],
    }
    if study:
        result["study"] = study
    return result


def public_profile(profile: dict[str, Any]) -> dict[str, Any]:
    private_keys = {"source_id", "assignment_group"}
    return {key: value for key, value in profile.items() if key not in private_keys}
