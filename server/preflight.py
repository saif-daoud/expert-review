"""Validate a rootless GPU-server deployment without loading any checkpoints."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


SERVER_DIR = Path(__file__).resolve().parent


def discover_project_root() -> Path:
    candidates = (SERVER_DIR.parent, *SERVER_DIR.parents)
    return next(
        (candidate for candidate in candidates if (candidate / "simulations" / "agents.py").is_file()),
        SERVER_DIR.parent,
    )


PROJECT_ROOT = Path(os.getenv("TOPAS_PROJECT_ROOT", str(discover_project_root()))).expanduser().resolve()

ENVIRONMENTS = {
    "base": os.getenv("STUDY_CONDA_ENV_BASE", "base"),
    "archer": os.getenv("STUDY_CONDA_ENV_ARCHER", "archer_env"),
    "aria": os.getenv("STUDY_CONDA_ENV_ARIA", "aria_env"),
    "sweet_rl": os.getenv("STUDY_CONDA_ENV_SWEET_RL", "sweet_rl"),
}

REQUIRED_PATHS = {
    "PatientAct profiles": PROJECT_ROOT / "simulations" / "data" / "patient_act.json",
    "simulation package": PROJECT_ROOT / "simulations" / "agents.py",
    "therapist prompt": PROJECT_ROOT / "baselines" / "sweet_rl_cbt" / "prompts" / "therapist_agent_prompt.txt",
    "Archer checkpoint": (
        PROJECT_ROOT
        / "baselines"
        / "outputs"
        / "cbt_offlinearcher_qwen25_7b"
        / "csv_logs"
        / "version_0"
        / "checkpoints"
        / "epoch=9-step=13294.ckpt"
    ),
    "OfflineArcher code": PROJECT_ROOT / "baselines" / "OfflineArcher-main" / "Algorithms.py",
    "ARIA checkpoint": PROJECT_ROOT / "baselines" / "outputs" / "cbt_aria_qwen25_7b" / "checkpoints" / "trainer.pt",
    "Sweet-RL checkpoint": (
        PROJECT_ROOT
        / "baselines"
        / "runs"
        / "sweet_rl_cbt_qwen25_7b_full"
        / "checkpoints"
        / "actor_no_tti"
    ),
    "Sweet-RL model weights": (
        PROJECT_ROOT
        / "baselines"
        / "runs"
        / "sweet_rl_cbt_qwen25_7b_full"
        / "checkpoints"
        / "actor_no_tti"
        / "model.safetensors"
    ),
}


def conda_executable() -> str | None:
    return os.getenv("STUDY_CONDA_EXE") or os.getenv("CONDA_EXE") or shutil.which("conda")


def main() -> int:
    failures: list[str] = []
    print(f"Project root: {PROJECT_ROOT}")
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
