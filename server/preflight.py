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
    "bundled Archer actor": SERVER_DIR / "model_runtime" / "archer_model.py",
    "bundled TOPAS policy": SERVER_DIR / "model_runtime" / "topa_agent.py",
    "bundled TOPAS models": SERVER_DIR / "model_runtime" / "topa_models.py",
    "bundled therapist prompt": SERVER_DIR / "assets" / "therapist_agent_prompt.txt",
    "bundled TOPAS macro actions": SERVER_DIR / "assets" / "topa_components" / "macro_actions.json",
    "bundled TOPAS micro actions": SERVER_DIR / "assets" / "topa_components" / "micro_actions.json",
    "bundled TOPAS conversation states": SERVER_DIR / "assets" / "topa_components" / "conversation_states.json",
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
        result = subprocess.run([gpu_command, "-L"], capture_output=True, text=True, timeout=15)
        print("\nVisible GPUs:")
        print(result.stdout.strip() or result.stderr.strip() or "None")
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
