from __future__ import annotations

import sys
from pathlib import Path

import pytest

from server.model_runtime.policies import CBT_DOMAIN as STANDALONE_DOMAIN
from server.model_runtime.policies import build_policy as build_standalone_policy


class RecordingGenerator:
    def __init__(self, outputs: list[str]) -> None:
        self.outputs = iter(outputs)
        self.calls: list[dict] = []
        self.last_prompt_diagnostics: dict = {}
        self.save_prompt_diagnostics = False

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        self.last_prompt_diagnostics = {
            "system_prompt": kwargs["system"],
            "user_prompt": kwargs["user"],
        }
        return next(self.outputs)

    def reset_turn_stats(self) -> None:
        return None

    def turn_stats(self) -> dict:
        return {"calls": len(self.calls)}


@pytest.mark.parametrize(
    ("method", "outputs"),
    [
        ("prompting", ["A therapeutic response."]),
        ("proact", ["Hidden plan.", "A therapeutic response."]),
        ("archer", ["A therapeutic response."]),
        ("aria", ["A therapeutic response."]),
        ("sweet_rl", ["A therapeutic response."]),
    ],
)
def test_bundled_policies_match_simulation_prompts(method, outputs):
    topas_root = Path(__file__).resolve().parents[3]
    if not (topas_root / "simulations" / "agents.py").is_file():
        pytest.skip("The source simulation package is not present for comparison.")
    sys.path.insert(0, str(topas_root))
    from simulations.agents import build_policy as build_simulation_policy
    from simulations.config import CBT_DOMAIN as simulation_domain

    transcript = "Patient: Hello\nTherapist: Welcome.\nPatient: I feel anxious."
    simulation_generator = RecordingGenerator(outputs.copy())
    standalone_generator = RecordingGenerator(outputs.copy())

    simulation_policy = build_simulation_policy(
        simulation_generator,
        method,
        Path("unused"),
        simulation_domain,
    )
    standalone_policy = build_standalone_policy(
        standalone_generator,
        method,
        STANDALONE_DOMAIN,
    )

    simulation_response, _ = simulation_policy.respond(transcript)
    standalone_response, _ = standalone_policy.respond(transcript)

    assert standalone_response == simulation_response
    assert standalone_generator.calls == simulation_generator.calls


def test_worker_has_no_simulation_or_baseline_code_imports():
    server_dir = Path(__file__).resolve().parents[1] / "server"
    worker_source = (server_dir / "method_worker.py").read_text(encoding="utf-8")
    assert "from simulations" not in worker_source
    assert "import simulations" not in worker_source
    assert "from baselines" not in worker_source
    assert "import baselines" not in worker_source
