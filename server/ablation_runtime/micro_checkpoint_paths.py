"""Resolve micro checkpoints without relying on historical folder spelling."""
from __future__ import annotations

import json
from pathlib import Path
import re


def canonical_name(value):
    value = " ".join(str(value).strip().split()).lower()
    return re.sub(r"[^a-z0-9._-]+", "_", value).strip("_")


def legacy_name(value):
    """Old sanitizer accidentally replaced uppercase letters instead of lowering them."""
    value = " ".join(str(value).strip().split())
    return re.sub(r"[^a-z0-9._-]+", "_", value).strip("_")


def saved_macro(folder):
    for name in ("model_meta.json", "label_map.json", "metrics.json"):
        path = folder / name
        if path.is_file():
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                meta = payload.get("meta", payload)
                if isinstance(meta, dict) and meta.get("macro_action"):
                    return str(meta["macro_action"])
    return None


def resolve_micro_policy_dir(root, macro, template="micro_{macro}"):
    """Prefer exact saved identity; support corrected and legacy directory names.

    Never guess between duplicate checkpoints or load a folder explicitly tagged
    with a different macro. Return the canonical expected path when none exists,
    so callers retain their normal missing-file/single-action behavior.
    """
    root, macro = Path(root), str(macro)
    expected = root / template.format(macro=canonical_name(macro))
    if not root.is_dir():
        return expected
    matches = [folder for folder in sorted(root.iterdir())
               if folder.is_dir() and (folder / "actor.pt").is_file() and saved_macro(folder) == macro]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError(f"Multiple micro checkpoints declare macro {macro!r}: " + ", ".join(map(str, matches)))
    candidates = []
    for slug in (canonical_name(macro), legacy_name(macro)):
        folder = root / template.format(macro=slug)
        if folder not in candidates and (folder / "actor.pt").is_file() and saved_macro(folder) in (None, macro):
            candidates.append(folder)
    if len(candidates) > 1:
        raise ValueError(f"Ambiguous micro checkpoint folders for {macro!r}: " + ", ".join(map(str, candidates)))
    return candidates[0] if candidates else expected
