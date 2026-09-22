"""Conversation stopping rules shared by the API and inference queue."""

from __future__ import annotations

import re


# Kept identical to simulations/simulators/base.py so live sessions and
# offline simulations recognize the same farewell phrases and exclusions.
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


def farewell_phrase(text: str) -> str | None:
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
