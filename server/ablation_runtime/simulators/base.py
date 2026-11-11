import re
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


SPEAKER_PREFIX = re.compile(
    r"^\s*(?:(?:therapist|patient|client|persuader|persuadee)\s*:\s*)+",
    re.IGNORECASE,
)
FAREWELL = re.compile(
    r"(?<![\w])(?:good(?:[\s-]+)?bye|bye(?:[\s-]+bye)?)(?![\w])",
    re.IGNORECASE,
)


def strip_speaker_prefixes(text: str) -> str:
    return SPEAKER_PREFIX.sub("", str(text or "")).strip()


def farewell_phrase(text: str) -> Optional[str]:
    cleaned = strip_speaker_prefixes(text)
    for match in FAREWELL.finditer(cleaned):
        before = cleaned[: match.start()]
        after = cleaned[match.end() :]
        if re.search(r"\b(?:say|says|said|saying|word|phrase)\W*$", before, re.IGNORECASE):
            continue
        if re.match(r"\W*to\b", after, re.IGNORECASE):
            continue
        return match.group(0)
    return None


def p4g_persona(profile: Dict[str, Any]) -> str:
    lines = ["<persona>"]
    for key, value in profile.items():
        if value is None:
            continue
        label = key.replace("_", " ").title()
        lines.append(f"- {label}: {value}")
    lines.append("</persona>")
    return "\n".join(lines)


class PatientSimulator(ABC):
    last_call: Dict[str, Any]

    @abstractmethod
    def respond(self, conversation: str) -> str:
        raise NotImplementedError

    @staticmethod
    def says_goodbye(text: str) -> bool:
        return farewell_phrase(text) is not None
