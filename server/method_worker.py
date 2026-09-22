"""JSON-lines model worker launched by the API in a method-specific Conda env."""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any


PROTOCOL_STDOUT = sys.stdout
SERVER_DIR = Path(__file__).resolve().parent
# Imported simulation code uses stdout for diagnostics. Keep stdout reserved for
# machine-readable protocol messages and send all diagnostics to the model log.
sys.stdout = sys.stderr


def _configured_path(variable: str, default: Path) -> str:
    return str(Path(os.getenv(variable, str(default))).expanduser().resolve())


def configure_runtime(runtime: str) -> None:
    model_root = Path(os.getenv("STUDY_MODEL_ROOT", str(SERVER_DIR / "models"))).expanduser().resolve()
    prompt = SERVER_DIR / "assets" / "therapist_agent_prompt.txt"
    os.environ.setdefault("THERAPIST_MAX_NEW_TOKENS", "96")

    if runtime == "base":
        os.environ["THERAPIST_BACKEND"] = "hf"
        os.environ.setdefault(
            "THERAPIST_HF_MODEL",
            os.getenv("STUDY_BASE_MODEL", "Qwen/Qwen2.5-7B-Instruct"),
        )
        os.environ.setdefault("THERAPIST_HF_DEVICE", "cuda")
        os.environ.setdefault("THERAPIST_HF_TORCH_DTYPE", "bf16")
        os.environ.setdefault("THERAPIST_HF_MAX_INPUT_LENGTH", "512")
        os.environ.setdefault("THERAPIST_PROMPTING_SYSTEM_PROMPT_PATH", str(prompt))
        return
    if runtime == "archer":
        os.environ["THERAPIST_BACKEND"] = "archer"
        os.environ.setdefault(
            "THERAPIST_ARCHER_MODEL",
            os.getenv("STUDY_ARCHER_BASE_MODEL", "Qwen/Qwen2.5-7B-Instruct"),
        )
        os.environ.setdefault(
            "THERAPIST_ARCHER_CHECKPOINT",
            _configured_path(
                "STUDY_ARCHER_CHECKPOINT",
                model_root / "archer" / "epoch=9-step=13294.ckpt",
            ),
        )
        os.environ.setdefault("THERAPIST_ARCHER_DEVICE", "cuda")
        os.environ.setdefault("THERAPIST_ARCHER_TORCH_DTYPE", "bf16")
        os.environ.setdefault("THERAPIST_ARCHER_MAX_INPUT_LENGTH", "512")
        os.environ.setdefault("THERAPIST_ARCHER_USE_LORA", "True")
        os.environ.setdefault("THERAPIST_ARCHER_USE_SWEET_RL_PROMPT", "True")
        os.environ.setdefault("THERAPIST_ARCHER_SYSTEM_PROMPT_PATH", str(prompt))
        return
    if runtime == "aria":
        os.environ["THERAPIST_BACKEND"] = "aria"
        os.environ.setdefault(
            "THERAPIST_ARIA_MODEL",
            os.getenv("STUDY_ARIA_BASE_MODEL", "Qwen/Qwen2.5-7B-Instruct"),
        )
        os.environ.setdefault(
            "THERAPIST_ARIA_CHECKPOINT",
            _configured_path(
                "STUDY_ARIA_CHECKPOINT",
                model_root / "aria" / "trainer.pt",
            ),
        )
        os.environ.setdefault("THERAPIST_ARIA_DEVICE", "cuda")
        os.environ.setdefault("THERAPIST_ARIA_TORCH_DTYPE", "bf16")
        os.environ.setdefault("THERAPIST_ARIA_MAX_INPUT_LENGTH", "4096")
        os.environ.setdefault("THERAPIST_ARIA_USE_LORA", "True")
        os.environ.setdefault("THERAPIST_ARIA_SYSTEM_PROMPT_PATH", str(prompt))
        return
    if runtime == "sweet_rl":
        os.environ["THERAPIST_BACKEND"] = "sweet_rl"
        os.environ.setdefault(
            "THERAPIST_SWEET_RL_MODEL",
            _configured_path(
                "STUDY_SWEET_RL_MODEL",
                model_root / "sweet_rl" / "actor_no_tti",
            ),
        )
        os.environ.setdefault("THERAPIST_SWEET_RL_DEVICE", "cuda")
        os.environ.setdefault("THERAPIST_SWEET_RL_DEVICE_MAP", "auto")
        os.environ.setdefault("THERAPIST_SWEET_RL_TORCH_DTYPE", "bf16")
        os.environ.setdefault("THERAPIST_SWEET_RL_TRUST_REMOTE_CODE", "True")
        os.environ.setdefault("THERAPIST_SWEET_RL_MAX_INPUT_LENGTH", "16384")
        os.environ.setdefault("THERAPIST_SWEET_RL_TEMPERATURE", "0.7")
        os.environ.setdefault("THERAPIST_SWEET_RL_TOP_P", "0.8")
        os.environ.setdefault("THERAPIST_SWEET_RL_PROMPT_PATH", str(prompt))
        return
    raise ValueError(f"Unknown worker runtime: {runtime}")


def transcript_from_history(history: list[dict[str, Any]]) -> str:
    labels = {"therapist": "Therapist", "patient": "Patient"}
    lines = []
    for message in history:
        role = str(message.get("role", ""))
        content = " ".join(str(message.get("content", "")).split()).strip()
        if role in labels and content:
            lines.append(f"{labels[role]}: {content}")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", choices=("base", "archer", "aria", "sweet_rl"), required=True)
    args = parser.parse_args()

    configure_runtime(args.runtime)

    with redirect_stdout(sys.stderr):
        from model_runtime.llm import build_generator_from_env
        from model_runtime.policies import CBT_DOMAIN, build_policy

        generator = build_generator_from_env("THERAPIST")
        policies: dict[str, Any] = {}

    allowed_methods = {"prompting", "proact"} if args.runtime == "base" else {args.runtime}
    for line in sys.stdin:
        request: dict[str, Any] = {}
        try:
            request = json.loads(line)
            method = str(request["method"])
            if method not in allowed_methods:
                raise ValueError(f"Method {method!r} cannot run in runtime {args.runtime!r}.")
            if method not in policies:
                with redirect_stdout(sys.stderr):
                    policies[method] = build_policy(
                        generator,
                        method,
                        CBT_DOMAIN,
                    )
            # Prompting was evaluated with the raw training prompt, while
            # ProAct uses Qwen's chat template. Both policies share the same
            # loaded base model, so switch only its lightweight prompt mode.
            if args.runtime == "base" and hasattr(generator, "prompt_style"):
                generator.prompt_style = "raw" if method == "prompting" else "chat"
            transcript = transcript_from_history(list(request.get("history") or []))
            generator.reset_turn_stats()
            with redirect_stdout(sys.stderr):
                utterance, _metadata = policies[method].respond(transcript)
            response = {"id": request.get("id"), "ok": True, "utterance": utterance}
            del _metadata
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            response = {"id": request.get("id"), "ok": False, "error": f"{type(exc).__name__}: {exc}"}
        PROTOCOL_STDOUT.write(json.dumps(response, ensure_ascii=False) + "\n")
        PROTOCOL_STDOUT.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
