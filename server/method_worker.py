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
ABLATION_METHODS = {
    "topas_expert_components",
    "topas_chatgpt_components",
    "topas_no_rl",
    "topas_no_conv_state",
    "topas_full",
}
ABLATION_POLICY_METHODS = {
    "topas_expert_components": "iql_policy_term_intra_with_conv",
    "topas_chatgpt_components": "iql_policy_term_intra_with_conv",
    "topas_no_rl": "topas_no_rl",
    "topas_no_conv_state": "iql_policy_term_intra_no_conv",
    "topas_full": "iql_policy_term_intra_with_conv",
}
# Imported simulation code uses stdout for diagnostics. Keep stdout reserved for
# machine-readable protocol messages and send all diagnostics to the model log.
sys.stdout = sys.stderr


def _configured_path(variable: str, default: Path) -> str:
    return str(Path(os.getenv(variable, str(default))).expanduser().resolve())


def _configure_ablation_method(method: str, model_root: Path) -> None:
    """Reproduce the ablation simulation launch configurations for human input."""
    component_root = SERVER_DIR / "assets" / "ablation_components"
    runs_root = Path(
        os.getenv(
            "STUDY_TOPAS_ABLATION_RUNS_DIR",
            str(model_root / "topas" / "ablation_runs"),
        )
    ).expanduser().resolve()

    os.environ["SIMULATION_ACTIVATION_LAYER"] = "22"
    os.environ["SIMULATION_FILTER_MACROS_BY_TERMINATION"] = "false"
    os.environ["SIMULATION_MACRO_POLICY_DETERMINISTIC"] = "false"
    os.environ["SIMULATION_TERMINATION_DETERMINISTIC"] = "false"
    os.environ["SIMULATION_MICRO_POLICY_DETERMINISTIC"] = "false"
    os.environ["SIMULATION_TERMINATION_THRESHOLD"] = "0.5"
    os.environ["SIMULATION_CONTEXT_TURNS"] = "5"
    os.environ["SIMULATION_CONV_STATE_UPDATE_INTERVAL"] = "1"
    os.environ["SIMULATION_MAX_MACRO_TURNS"] = "-1"
    os.environ["SIMULATION_ALLOW_TERMINATED_MACRO_RESELECTION"] = "false"
    os.environ["SIMULATION_USE_MEMORY"] = "false"
    os.environ["SIMULATION_GENERATION_MEMORY"] = "false"
    os.environ["SIMULATION_DEVICE"] = "cuda"

    if method == "topas_full":
        full_runs = Path(
            _configured_path(
                "STUDY_TOPAS_FULL_RUNS_DIR",
                model_root / "topas" / "full" / "old_runs",
            )
        )
        os.environ["TOPA_COMPONENTS_DIR"] = str(component_root / "no_rl")
        os.environ["SIMULATION_RUNS_DIR"] = str(full_runs)
        os.environ["SIMULATION_MACRO_POLICY_DIR"] = _configured_path(
            "STUDY_TOPAS_FULL_MACRO_POLICY_DIR",
            full_runs / "TOPAS_Options_withmemory22",
        )
        os.environ["SIMULATION_MICRO_POLICY_DIR"] = _configured_path(
            "STUDY_TOPAS_FULL_MICRO_POLICY_DIR",
            full_runs / "TOPAS_Primitive_withmemory22",
        )
        os.environ["SIMULATION_CONV_STATE_DIR"] = _configured_path(
            "STUDY_TOPAS_FULL_CONV_STATE_DIR",
            model_root / "topas" / "full" / "runs" / "sft_conv_state_acts_L22",
        )
        os.environ["SIMULATION_MEMORY_RETRIEVER_DIR"] = _configured_path(
            "STUDY_TOPAS_FULL_MEMORY_RETRIEVER_DIR",
            full_runs / "TOPAS_Options_withmemory22_memory_retrieval",
        )
        os.environ["SIMULATION_KG_SCHEMA_PATH"] = _configured_path(
            "STUDY_TOPAS_FULL_KG_SCHEMA_PATH",
            model_root / "topas" / "full" / "components" / "knowledge_graph.json",
        )
        os.environ["SIMULATION_POLICY_CONV_STATE_ONLY"] = "auto"
        os.environ["SIMULATION_ACTIVATION_WINDOW"] = "5"
        os.environ["SIMULATION_MEMORY_MODE"] = "retrieved"
        os.environ["SIMULATION_GENERATION_MEMORY"] = "true"
        os.environ["SIMULATION_MEMORY_RETRIEVAL_DETERMINISTIC"] = "false"
        os.environ["SIMULATION_MEMORY_RETRIEVAL_AVAILABLE_ONLY"] = "true"
        os.environ["SIMULATION_MEMORY_RETRIEVAL_TOP_K"] = "3"
        os.environ["SIMULATION_REQUIRE_FULL_GENERATION_MEMORY"] = "true"
        os.environ["SIMULATION_MEMORY_MAX_TOKENS"] = "0"
        os.environ["SIMULATION_SESSION_IDX"] = "0"
        os.environ["TOPA_LLM_PROVIDER"] = os.getenv(
            "STUDY_TOPAS_FULL_KG_PROVIDER", "openai"
        )
        os.environ["SIMULATION_KG_MODEL"] = os.getenv(
            "STUDY_TOPAS_FULL_KG_MODEL", "gpt-4.1"
        )
        os.environ["THERAPIST_TEMPERATURE"] = "0.7"
        os.environ["THERAPIST_TOP_P"] = "0.9"
        os.environ["THERAPIST_HF_TORCH_DTYPE"] = "float16"
        return

    if method == "topas_no_rl":
        os.environ["TOPA_COMPONENTS_DIR"] = str(component_root / "no_rl")
        os.environ["THERAPIST_HF_TORCH_DTYPE"] = "bf16"
        # Retain the launcher's values for provenance. In the human study the
        # patient-simulator validation loop is absent, so these do not change
        # therapist generation.
        os.environ["TOPAS_VALIDATION_MAX_RETRIES"] = "5"
        os.environ["TOPAS_NO_RL_STATE_MAX_NEW_TOKENS"] = "2048"
        return

    os.environ["THERAPIST_HF_TORCH_DTYPE"] = "float16"
    if method == "topas_expert_components":
        os.environ["TOPA_COMPONENTS_DIR"] = str(component_root / "expert")
        os.environ["SIMULATION_RUNS_DIR"] = _configured_path(
            "STUDY_TOPAS_EXPERT_RUNS_DIR", runs_root / "expert"
        )
        os.environ["SIMULATION_CONV_STATE_DIR"] = _configured_path(
            "STUDY_TOPAS_EXPERT_CONV_STATE_DIR",
            runs_root / "sft_conv_state_acts_L22_Expert",
        )
        os.environ["SIMULATION_MACRO_POLICY_DIR"] = _configured_path(
            "STUDY_TOPAS_EXPERT_MACRO_POLICY_DIR", runs_root / "expert_O_22"
        )
        os.environ["SIMULATION_MICRO_POLICY_DIR"] = _configured_path(
            "STUDY_TOPAS_EXPERT_MICRO_POLICY_DIR", runs_root / "expert_P_22"
        )
        os.environ["SIMULATION_POLICY_CONV_STATE_ONLY"] = "true"
        return

    if method == "topas_chatgpt_components":
        os.environ["TOPA_COMPONENTS_DIR"] = str(component_root / "chatgpt")
        os.environ["SIMULATION_RUNS_DIR"] = _configured_path(
            "STUDY_TOPAS_CHAT_RUNS_DIR", runs_root / "chat"
        )
        os.environ["SIMULATION_CONV_STATE_DIR"] = _configured_path(
            "STUDY_TOPAS_CHAT_CONV_STATE_DIR",
            runs_root / "sft_conv_state_acts_L22_Chat",
        )
        os.environ["SIMULATION_MACRO_POLICY_DIR"] = _configured_path(
            "STUDY_TOPAS_CHAT_MACRO_POLICY_DIR", runs_root / "chat_O_22"
        )
        os.environ["SIMULATION_MICRO_POLICY_DIR"] = _configured_path(
            "STUDY_TOPAS_CHAT_MICRO_POLICY_DIR", runs_root / "chat_P_22"
        )
        os.environ["SIMULATION_POLICY_CONV_STATE_ONLY"] = "true"
        return

    if method == "topas_no_conv_state":
        no_conv_runs = Path(
            _configured_path("STUDY_TOPAS_NO_CONV_RUNS_DIR", runs_root / "no_conv")
        )
        macro_dir = Path(
            _configured_path(
                "STUDY_TOPAS_NO_CONV_MACRO_POLICY_DIR",
                no_conv_runs / "iql_policy_over_options_acts_L22",
            )
        )
        os.environ["SIMULATION_RUNS_DIR"] = str(no_conv_runs)
        os.environ["SIMULATION_MACRO_POLICY_DIR"] = str(macro_dir)
        os.environ["SIMULATION_MICRO_POLICY_DIR"] = _configured_path(
            "STUDY_TOPAS_NO_CONV_MICRO_POLICY_DIR",
            no_conv_runs / "iql_micro_policies_acts_L22",
        )
        # The no-conversation-state launcher reads the action space saved with
        # its macro checkpoint rather than using a separate component folder.
        os.environ["TOPA_COMPONENTS_DIR"] = _configured_path(
            "STUDY_TOPAS_NO_CONV_COMPONENTS_PATH",
            macro_dir / "used_action_space.json",
        )
        os.environ["SIMULATION_POLICY_CONV_STATE_ONLY"] = "auto"
        return

    raise ValueError(f"Unknown ablation method: {method}")


def configure_runtime(runtime: str, method: str) -> None:
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
        os.environ.setdefault(
            "THERAPIST_HF_TORCH_DTYPE",
            "float16" if method == "topas" else "bf16",
        )
        os.environ.setdefault("THERAPIST_HF_MAX_INPUT_LENGTH", "512")
        os.environ.setdefault("THERAPIST_PROMPTING_SYSTEM_PROMPT_PATH", str(prompt))
        if method in ABLATION_METHODS:
            _configure_ablation_method(method, model_root)
            return
        if method == "topas":
            topas_runs = Path(
                os.getenv(
                    "STUDY_TOPAS_RUNS_DIR",
                    str(model_root / "topas" / "runs"),
                )
            ).expanduser().resolve()
            os.environ.setdefault("STUDY_TOPAS_VARIANT", "iql_policy_term_intra_with_conv")
            os.environ.setdefault("SIMULATION_RUNS_DIR", str(topas_runs))
            os.environ.setdefault(
                "SIMULATION_CONV_STATE_DIR",
                os.getenv(
                    "STUDY_TOPAS_CONV_STATE_DIR",
                    str(topas_runs / "sft_conv_state_acts_L22"),
                ),
            )
            os.environ.setdefault("SIMULATION_FILTER_MACROS_BY_TERMINATION", "false")
            os.environ.setdefault("SIMULATION_MACRO_POLICY_DETERMINISTIC", "false")
            os.environ.setdefault("SIMULATION_TERMINATION_DETERMINISTIC", "false")
            os.environ.setdefault("SIMULATION_MICRO_POLICY_DETERMINISTIC", "false")
            os.environ.setdefault("SIMULATION_TERMINATION_THRESHOLD", "0.5")
            os.environ.setdefault("SIMULATION_POLICY_CONV_STATE_ONLY", "true")
            os.environ.setdefault("SIMULATION_CONTEXT_TURNS", "5")
            os.environ.setdefault("SIMULATION_CONV_STATE_UPDATE_INTERVAL", "1")
            os.environ.setdefault("SIMULATION_MAX_MACRO_TURNS", "-1")
            os.environ.setdefault("SIMULATION_DEVICE", "cuda")
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
    # The CBT simulation begins every case with this domain-level utterance.
    # It stays implicit in the browser while keeping live inference aligned
    # with the exact simulation context used by all six therapist methods.
    lines = ["Patient: Hello"]
    for message in history:
        role = str(message.get("role", ""))
        content = " ".join(str(message.get("content", "")).split()).strip()
        if role in labels and content:
            lines.append(f"{labels[role]}: {content}")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", choices=("base", "archer", "aria", "sweet_rl"), required=True)
    parser.add_argument(
        "--method",
        choices=(
            "prompting",
            "proact",
            "archer",
            "aria",
            "sweet_rl",
            "topas",
            *sorted(ABLATION_METHODS),
        ),
        required=True,
    )
    args = parser.parse_args()

    configure_runtime(args.runtime, args.method)

    with redirect_stdout(sys.stderr):
        from model_runtime.llm import build_generator_from_env

        generator = build_generator_from_env("THERAPIST")
        policies: dict[str, Any] = {}

    allowed_methods = {args.method}
    for line in sys.stdin:
        request: dict[str, Any] = {}
        try:
            request = json.loads(line)
            method = str(request["method"])
            if method not in allowed_methods:
                raise ValueError(f"Method {method!r} cannot run in runtime {args.runtime!r}.")
            if method not in policies:
                with redirect_stdout(sys.stderr):
                    if method in ABLATION_METHODS:
                        from ablation_runtime.agents import build_policy as build_ablation_policy
                        from ablation_runtime.config import CBT_DOMAIN as ABLATION_CBT_DOMAIN

                        policies[method] = build_ablation_policy(
                            generator,
                            ABLATION_POLICY_METHODS[method],
                            Path(os.environ["TOPA_COMPONENTS_DIR"]),
                            ABLATION_CBT_DOMAIN,
                        )
                        reset_session = getattr(policies[method], "reset_session", None)
                        if reset_session is not None:
                            reset_session(user_idx=0, session_idx=0, profile_id="human")
                    else:
                        from model_runtime.policies import CBT_DOMAIN, build_policy

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
