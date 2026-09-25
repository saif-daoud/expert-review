from __future__ import annotations

import sys
from pathlib import Path

import pytest

from server.model_runtime.policies import CBT_DOMAIN as STANDALONE_DOMAIN
from server.model_runtime.policies import _clean
from server.model_runtime.policies import build_policy as build_standalone_policy
from server.model_runtime.llm import TransformersChatGenerator


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


class CharacterTokenizer:
    def __call__(self, text: str, **_kwargs):
        return {"input_ids": [ord(character) for character in text]}

    def decode(self, token_ids, **_kwargs):
        return "".join(chr(token_id) for token_id in token_ids)


def test_context_limit_truncates_only_conversation_history():
    generator = object.__new__(TransformersChatGenerator)
    generator.prompt_style = "chat"
    generator.max_input_length = 180
    generator.tokenizer = CharacterTokenizer()
    system = "SYSTEM INSTRUCTION MUST REMAIN COMPLETE"
    prefix = "Conversation so far:\n"
    suffix = "\nWrite only the next therapist response."
    history = "\n".join(
        f"Patient: historical turn {index}" for index in range(20)
    )

    user = generator.build_user_with_history(
        system=system,
        prefix=prefix,
        history=history,
        suffix=suffix,
    )
    rendered = generator._build_prompt(system, user)

    assert len(generator.tokenizer(rendered)["input_ids"]) <= 180
    assert system in rendered
    assert suffix.strip() in rendered
    assert "historical turn 19" in rendered
    assert "historical turn 0" not in rendered


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
    assert len(standalone_generator.calls) == len(simulation_generator.calls)
    for standalone_call, simulation_call in zip(
        standalone_generator.calls,
        simulation_generator.calls,
    ):
        assert standalone_call["system"] == simulation_call["system"]
        assert standalone_call["user"] == simulation_call["user"]
        # Live inference adds the requested boundary for leaked continuation
        # text while otherwise retaining the simulation's stop list.
        assert standalone_call["stop"] == [
            "\nAssistant:",
            "\nHuman:",
            *simulation_call["stop"],
        ]


def test_worker_has_no_simulation_or_baseline_code_imports():
    server_dir = Path(__file__).resolve().parents[1] / "server"
    worker_source = (server_dir / "method_worker.py").read_text(encoding="utf-8")
    assert "from simulations" not in worker_source
    assert "import simulations" not in worker_source
    assert "from baselines" not in worker_source
    assert "import baselines" not in worker_source


def test_topas_uses_the_live_simulation_configuration(monkeypatch, tmp_path):
    from server.model_runtime import topa_agent

    captured: dict = {}

    class FakeAgent:
        def begin_turn(self):
            return None

        def next_system_utterance(self, dialogue, session_metadata):
            captured["dialogue"] = dialogue
            captured["session_metadata"] = session_metadata
            return "A TOPAS response."

        def end_turn(self):
            return {"calls": 1}

        def get_last_turn_metadata(self):
            return {"raw_output": "A TOPAS response.", "tensor_payload": {}}

    def fake_build_agent(**kwargs):
        captured.update(kwargs)
        return FakeAgent()

    monkeypatch.setattr(topa_agent, "build_agent", fake_build_agent)
    monkeypatch.setenv("SIMULATION_RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setenv("SIMULATION_CONV_STATE_DIR", str(tmp_path / "conv"))
    monkeypatch.setenv("SIMULATION_FILTER_MACROS_BY_TERMINATION", "false")
    monkeypatch.setenv("SIMULATION_MACRO_POLICY_DETERMINISTIC", "false")
    monkeypatch.setenv("SIMULATION_TERMINATION_DETERMINISTIC", "false")
    monkeypatch.setenv("SIMULATION_MICRO_POLICY_DETERMINISTIC", "false")
    monkeypatch.setenv("SIMULATION_TERMINATION_THRESHOLD", "0.5")
    monkeypatch.setenv("SIMULATION_POLICY_CONV_STATE_ONLY", "true")
    monkeypatch.setenv("SIMULATION_CONTEXT_TURNS", "5")
    monkeypatch.setenv("SIMULATION_CONV_STATE_UPDATE_INTERVAL", "1")
    monkeypatch.setenv("SIMULATION_MAX_MACRO_TURNS", "-1")

    generator = RecordingGenerator(["unused"])
    components = tmp_path / "components"
    policy = build_standalone_policy(
        generator,
        "topas",
        STANDALONE_DOMAIN,
        components,
    )
    response, _ = policy.respond("Patient: Hello\nTherapist: Welcome.\nPatient: I feel anxious.")

    assert response == "A TOPAS response."
    assert captured["variant_name"] == "iql_policy_term_intra_with_conv"
    assert captured["action_space_path"] == components
    assert captured["context_turns"] == 5
    assert captured["max_macro_turns"] == -1
    assert captured["conv_state_update_interval"] == 1
    assert captured["termination_threshold"] == 0.5
    assert captured["macro_policy_deterministic"] is False
    assert captured["termination_deterministic"] is False
    assert captured["filter_macros_by_termination"] is False
    assert captured["micro_policy_deterministic"] is False
    assert captured["policy_conv_state_only"] is True
    assert captured["dialogue"][-1] == {"speaker": "Patient", "text": "I feel anxious."}


def test_topas_runtime_is_bundled_without_external_code_imports():
    server_dir = Path(__file__).resolve().parents[1] / "server"
    for name in ("topa_agent.py", "topa_models.py"):
        source = (server_dir / "model_runtime" / name).read_text(encoding="utf-8")
        assert "from simulations" not in source
        assert "import simulations" not in source
        assert "from baselines" not in source
        assert "import baselines" not in source


def test_bundled_topas_sources_match_the_simulation_runtime():
    project_root = Path(__file__).resolve().parents[3]
    simulation_dir = project_root / "simulations"
    if not (simulation_dir / "topa_agent.py").is_file():
        pytest.skip("The source simulation package is not present for comparison.")
    server_dir = Path(__file__).resolve().parents[1] / "server"
    relative_files = (
        Path("topa_agent.py"),
        Path("topa_models.py"),
        Path("topa_components/macro_actions.json"),
        Path("topa_components/micro_actions.json"),
        Path("topa_components/conversation_states.json"),
    )
    for relative in relative_files:
        bundled_root = (
            server_dir / "assets"
            if relative.parts[0] == "topa_components"
            else server_dir / "model_runtime"
        )
        assert (bundled_root / relative).read_bytes() == (simulation_dir / relative).read_bytes()


def test_leaked_speaker_markers_are_generation_boundaries():
    raw = (
        "Hello, it's okay to share. What comes to mind first?"
        "Assistant: It sounds like you're having a tough day."
    )
    assert _clean(raw) == "Hello, it's okay to share. What comes to mind first?"
    assert _clean("Assistant: Hello, how are you feeling?") == "Hello, how are you feeling?"
    assert _clean("Could you tell me more?Human: I feel anxious.") == "Could you tell me more?"
    assert _clean("That sounds difficult. Patient: Yes, it is.") == "That sounds difficult."
