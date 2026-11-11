"""Validate a rootless GPU-server deployment without loading any checkpoints."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


SERVER_DIR = Path(__file__).resolve().parent


MODEL_ROOT = Path(os.getenv("STUDY_MODEL_ROOT", str(SERVER_DIR / "models"))).expanduser().resolve()

ENVIRONMENTS = {
    "base": os.getenv("STUDY_CONDA_ENV_BASE", "base"),
    "archer": os.getenv("STUDY_CONDA_ENV_ARCHER", "archer_env"),
    "aria": os.getenv("STUDY_CONDA_ENV_ARIA", "aria_env"),
    "sweet_rl": os.getenv("STUDY_CONDA_ENV_SWEET_RL", "sweet_rl"),
}

try:
    GPU_ORDER = tuple(
        int(value.strip())
        for value in os.getenv("STUDY_GPU_ORDER", "0,1,2,3").split(",")
        if value.strip()
    )
except ValueError:
    GPU_ORDER = ()

ARCHER_CHECKPOINT = Path(
    os.getenv("STUDY_ARCHER_CHECKPOINT", str(MODEL_ROOT / "archer" / "epoch=9-step=13294.ckpt"))
).expanduser().resolve()
ARIA_CHECKPOINT = Path(
    os.getenv("STUDY_ARIA_CHECKPOINT", str(MODEL_ROOT / "aria" / "trainer.pt"))
).expanduser().resolve()
SWEET_RL_MODEL = Path(
    os.getenv("STUDY_SWEET_RL_MODEL", str(MODEL_ROOT / "sweet_rl" / "actor_no_tti"))
).expanduser().resolve()
TOPAS_RUNS_DIR = Path(
    os.getenv("STUDY_TOPAS_RUNS_DIR", str(MODEL_ROOT / "topas" / "runs"))
).expanduser().resolve()
TOPAS_CONV_STATE_DIR = Path(
    os.getenv(
        "STUDY_TOPAS_CONV_STATE_DIR",
        str(TOPAS_RUNS_DIR / "sft_conv_state_acts_L22"),
    )
).expanduser().resolve()
ABLATION_RUNS_DIR = Path(
    os.getenv("STUDY_TOPAS_ABLATION_RUNS_DIR", str(MODEL_ROOT / "topas" / "ablation_runs"))
).expanduser().resolve()


def configured_path(variable: str, default: Path) -> Path:
    return Path(os.getenv(variable, str(default))).expanduser().resolve()


EXPERT_CONV_STATE_DIR = configured_path(
    "STUDY_TOPAS_EXPERT_CONV_STATE_DIR",
    ABLATION_RUNS_DIR / "sft_conv_state_acts_L22_Expert",
)
EXPERT_MACRO_POLICY_DIR = configured_path(
    "STUDY_TOPAS_EXPERT_MACRO_POLICY_DIR", ABLATION_RUNS_DIR / "expert_O_22"
)
EXPERT_MICRO_POLICY_DIR = configured_path(
    "STUDY_TOPAS_EXPERT_MICRO_POLICY_DIR", ABLATION_RUNS_DIR / "expert_P_22"
)
CHAT_CONV_STATE_DIR = configured_path(
    "STUDY_TOPAS_CHAT_CONV_STATE_DIR",
    ABLATION_RUNS_DIR / "sft_conv_state_acts_L22_CHATGPT",
)
CHAT_MACRO_POLICY_DIR = configured_path(
    "STUDY_TOPAS_CHAT_MACRO_POLICY_DIR", ABLATION_RUNS_DIR / "chat_O_22"
)
CHAT_MICRO_POLICY_DIR = configured_path(
    "STUDY_TOPAS_CHAT_MICRO_POLICY_DIR", ABLATION_RUNS_DIR / "chat_P_22"
)
NO_CONV_RUNS_DIR = configured_path(
    "STUDY_TOPAS_NO_CONV_RUNS_DIR", ABLATION_RUNS_DIR / "no_conv"
)
NO_CONV_MACRO_POLICY_DIR = configured_path(
    "STUDY_TOPAS_NO_CONV_MACRO_POLICY_DIR",
    NO_CONV_RUNS_DIR / "TOPAS_Options_wo_cvstates22",
)
NO_CONV_MICRO_POLICY_DIR = configured_path(
    "STUDY_TOPAS_NO_CONV_MICRO_POLICY_DIR",
    NO_CONV_RUNS_DIR / "TOPAS_Primitive_wo_cvstates22",
)
NO_CONV_COMPONENTS_PATH = configured_path(
    "STUDY_TOPAS_NO_CONV_COMPONENTS_PATH",
    NO_CONV_MACRO_POLICY_DIR / "used_action_space.json",
)
FULL_RUNS_DIR = configured_path(
    "STUDY_TOPAS_FULL_RUNS_DIR", MODEL_ROOT / "topas" / "full" / "old_runs"
)
FULL_MACRO_POLICY_DIR = configured_path(
    "STUDY_TOPAS_FULL_MACRO_POLICY_DIR",
    FULL_RUNS_DIR / "TOPAS_Options_withmemory22",
)
FULL_MICRO_POLICY_DIR = configured_path(
    "STUDY_TOPAS_FULL_MICRO_POLICY_DIR",
    FULL_RUNS_DIR / "TOPAS_Primitive_withmemory22",
)
FULL_CONV_STATE_DIR = configured_path(
    "STUDY_TOPAS_FULL_CONV_STATE_DIR",
    FULL_RUNS_DIR / "sft_conv_state_acts_L22",
)
FULL_MEMORY_RETRIEVER_DIR = configured_path(
    "STUDY_TOPAS_FULL_MEMORY_RETRIEVER_DIR",
    FULL_RUNS_DIR / "TOPAS_Options_withmemory22_memory_retrieval",
)
FULL_KG_SCHEMA_PATH = configured_path(
    "STUDY_TOPAS_FULL_KG_SCHEMA_PATH",
    MODEL_ROOT / "topas" / "full" / "components" / "knowledge_graph.json",
)


def optional_directory(variable: str, default: Path) -> Path:
    value = os.getenv(variable)
    if value is None or value.strip().lower() == "none":
        return default
    return Path(value).expanduser().resolve()


TOPAS_MACRO_POLICY_DIR = optional_directory(
    "SIMULATION_MACRO_POLICY_DIR",
    TOPAS_RUNS_DIR / "iql_policy_over_options_with_conv_acts_L22",
)
TOPAS_MICRO_POLICY_DIR = optional_directory(
    "SIMULATION_MICRO_POLICY_DIR",
    TOPAS_RUNS_DIR / "iql_intra_option_policies_acts_L22",
)

REQUIRED_PATHS = {
    "bundled PatientAct profiles": SERVER_DIR / "data" / "patient_act.json",
    "bundled model runtime": SERVER_DIR / "model_runtime" / "llm.py",
    "bundled policy runtime": SERVER_DIR / "model_runtime" / "policies.py",
    "bundled session stopping rules": SERVER_DIR / "session_rules.py",
    "bundled Archer actor": SERVER_DIR / "model_runtime" / "archer_model.py",
    "bundled TOPAS policy": SERVER_DIR / "model_runtime" / "topa_agent.py",
    "bundled TOPAS models": SERVER_DIR / "model_runtime" / "topa_models.py",
    "bundled therapist prompt": SERVER_DIR / "assets" / "therapist_agent_prompt.txt",
    "bundled TOPAS macro actions": SERVER_DIR / "assets" / "topa_components" / "macro_actions.json",
    "bundled TOPAS micro actions": SERVER_DIR / "assets" / "topa_components" / "micro_actions.json",
    "bundled TOPAS conversation states": SERVER_DIR / "assets" / "topa_components" / "conversation_states.json",
    "bundled ablation policy runtime": SERVER_DIR / "ablation_runtime" / "agents.py",
    "bundled latest TOPAS ablation agent": SERVER_DIR / "ablation_runtime" / "topa_agent.py",
    "bundled expert components": SERVER_DIR / "assets" / "ablation_components" / "expert" / "macro_actions.json",
    "bundled ChatGPT components": SERVER_DIR / "assets" / "ablation_components" / "chatgpt" / "macro_actions.json",
    "bundled prompt-only components": SERVER_DIR / "assets" / "ablation_components" / "no_rl" / "macro_actions.json",
    "bundled standalone KG annotator": SERVER_DIR / "topa" / "annotation" / "annotators" / "knowledge_graph.py",
    "Archer checkpoint": ARCHER_CHECKPOINT,
    "ARIA checkpoint": ARIA_CHECKPOINT,
    "Sweet-RL checkpoint": SWEET_RL_MODEL,
    "Sweet-RL model weights": SWEET_RL_MODEL / "model.safetensors",
    "TOPAS macro actor": TOPAS_MACRO_POLICY_DIR / "actor.pt",
    "TOPAS termination actor": TOPAS_MACRO_POLICY_DIR / "termination.pt",
    "TOPAS macro metadata": TOPAS_MACRO_POLICY_DIR / "metrics.json",
    "TOPAS intra-option policies": TOPAS_MICRO_POLICY_DIR,
    "TOPAS conversation-state actor": TOPAS_CONV_STATE_DIR / "actor.pt",
    "TOPAS conversation-state labels": TOPAS_CONV_STATE_DIR / "label_map.json",
    "expert-components macro actor": EXPERT_MACRO_POLICY_DIR / "actor.pt",
    "expert-components termination actor": EXPERT_MACRO_POLICY_DIR / "termination.pt",
    "expert-components macro metadata": EXPERT_MACRO_POLICY_DIR / "metrics.json",
    "expert-components micro policies": EXPERT_MICRO_POLICY_DIR,
    "expert-components conversation-state actor": EXPERT_CONV_STATE_DIR / "actor.pt",
    "expert-components conversation-state labels": EXPERT_CONV_STATE_DIR / "label_map.json",
    "ChatGPT-components macro actor": CHAT_MACRO_POLICY_DIR / "actor.pt",
    "ChatGPT-components termination actor": CHAT_MACRO_POLICY_DIR / "termination.pt",
    "ChatGPT-components macro metadata": CHAT_MACRO_POLICY_DIR / "metrics.json",
    "ChatGPT-components micro policies": CHAT_MICRO_POLICY_DIR,
    "ChatGPT-components conversation-state actor": CHAT_CONV_STATE_DIR / "actor.pt",
    "ChatGPT-components conversation-state labels": CHAT_CONV_STATE_DIR / "label_map.json",
    "no-conversation-state macro actor": NO_CONV_MACRO_POLICY_DIR / "actor.pt",
    "no-conversation-state termination actor": NO_CONV_MACRO_POLICY_DIR / "termination.pt",
    "no-conversation-state macro metadata": NO_CONV_MACRO_POLICY_DIR / "model_meta.json",
    "no-conversation-state action space": NO_CONV_COMPONENTS_PATH,
    "no-conversation-state micro policies": NO_CONV_MICRO_POLICY_DIR,
    "full TOPAS macro actor": FULL_MACRO_POLICY_DIR / "actor.pt",
    "full TOPAS termination actor": FULL_MACRO_POLICY_DIR / "termination.pt",
    "full TOPAS macro metadata": FULL_MACRO_POLICY_DIR / "model_meta.json",
    "full TOPAS saved action space": FULL_MACRO_POLICY_DIR / "used_action_space.json",
    "full TOPAS micro policies": FULL_MICRO_POLICY_DIR,
    "full TOPAS conversation-state actor": FULL_CONV_STATE_DIR / "actor.pt",
    "full TOPAS conversation-state labels": FULL_CONV_STATE_DIR / "label_map.json",
    "full TOPAS memory retriever": FULL_MEMORY_RETRIEVER_DIR / "retriever.pt",
    "full TOPAS memory retriever config": FULL_MEMORY_RETRIEVER_DIR / "run_config.json",
    "full TOPAS memory candidate masks": FULL_MEMORY_RETRIEVER_DIR / "candidate_masks.npy",
    "full TOPAS KG schema": FULL_KG_SCHEMA_PATH,
}


def conda_executable() -> str | None:
    return os.getenv("STUDY_CONDA_EXE") or os.getenv("CONDA_EXE") or shutil.which("conda")


def main() -> int:
    failures: list[str] = []
    print(f"Server directory: {SERVER_DIR}")
    print(f"Model root: {MODEL_ROOT}")
    print(f"TOPAS runs: {TOPAS_RUNS_DIR}")
    for label, path in REQUIRED_PATHS.items():
        exists = path.is_file() if path.suffix else path.is_dir()
        print(f"[{'OK' if exists else 'MISSING'}] {label}: {path}")
        if not exists:
            failures.append(label)

    for variable in ("AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_KEY"):
        value = os.getenv(variable, "").strip()
        configured = bool(value) and not value.startswith("replace-with-")
        print(f"[{'OK' if configured else 'MISSING'}] full TOPAS {variable}")
        if not configured:
            failures.append(f"full TOPAS {variable}")

    conda = conda_executable()
    if not conda:
        print("[MISSING] Conda executable (set STUDY_CONDA_EXE)")
        failures.append("Conda executable")
    else:
        try:
            result = subprocess.run(
                [conda, "env", "list", "--json"],
                check=True,
                capture_output=True,
                text=True,
                timeout=30,
            )
            paths = [Path(value) for value in json.loads(result.stdout).get("envs", [])]
            info = subprocess.run(
                [conda, "info", "--json"],
                check=True,
                capture_output=True,
                text=True,
                timeout=30,
            )
            root_prefix = Path(json.loads(info.stdout)["root_prefix"]).resolve()
            names = {path.name for path in paths}
            if any(path.resolve() == root_prefix for path in paths):
                names.add("base")
            for runtime, name in ENVIRONMENTS.items():
                exists = name in names or any(str(path) == name for path in paths)
                print(f"[{'OK' if exists else 'MISSING'}] {runtime} Conda environment: {name}")
                if not exists:
                    failures.append(f"Conda environment {name}")
        except (OSError, subprocess.SubprocessError, ValueError, json.JSONDecodeError) as exc:
            print(f"[FAILED] Could not inspect Conda environments: {exc}")
            failures.append("Conda environment check")

    gpu_command = shutil.which("nvidia-smi")
    if gpu_command:
        result = subprocess.run(
            [
                gpu_command,
                "--query-gpu=index,name,memory.total,memory.free",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=15,
        )
        print("\nVisible GPUs (index, name, total MiB, free MiB):")
        print(result.stdout.strip() or result.stderr.strip() or "None")
        visible_indices: set[int] = set()
        for line in result.stdout.splitlines():
            try:
                visible_indices.add(int(line.split(",", maxsplit=1)[0].strip()))
            except (ValueError, IndexError):
                continue
        if not GPU_ORDER or len(set(GPU_ORDER)) != len(GPU_ORDER):
            print("[INVALID] STUDY_GPU_ORDER must contain unique numeric GPU indices")
            failures.append("GPU order")
        else:
            missing_gpus = [gpu for gpu in GPU_ORDER if gpu not in visible_indices]
            print(f"Configured GPU order: {','.join(map(str, GPU_ORDER))}")
            if missing_gpus:
                print(f"[MISSING] Configured GPU indices: {missing_gpus}")
                failures.append("Configured GPUs")
    else:
        print("[MISSING] nvidia-smi")
        failures.append("nvidia-smi")

    if failures:
        print("\nPreflight failed: " + ", ".join(failures))
        return 1
    print("\nPreflight passed. No model checkpoint was loaded.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
