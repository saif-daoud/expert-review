from __future__ import annotations

import os
from pathlib import Path

from server import method_worker
from server.ablation_runtime.agents import PromptActionPolicy, build_policy
from server.ablation_runtime.config import CBT_DOMAIN


class FakeGenerator:
    def reset_turn_stats(self):
        return None


def test_expert_component_worker_matches_simulation_flags(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("STUDY_MODEL_ROOT", str(tmp_path / "models"))
    monkeypatch.setenv("STUDY_TOPAS_EXPERT_RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setenv("STUDY_TOPAS_EXPERT_CONV_STATE_DIR", str(tmp_path / "conv"))
    monkeypatch.setenv("STUDY_TOPAS_EXPERT_MACRO_POLICY_DIR", str(tmp_path / "macro"))
    monkeypatch.setenv("STUDY_TOPAS_EXPERT_MICRO_POLICY_DIR", str(tmp_path / "micro"))

    method_worker.configure_runtime("base", "topas_expert_components")

    assert os.environ["SIMULATION_CONV_STATE_DIR"] == str((tmp_path / "conv").resolve())
    assert os.environ["SIMULATION_MACRO_POLICY_DIR"] == str((tmp_path / "macro").resolve())
    assert os.environ["SIMULATION_MICRO_POLICY_DIR"] == str((tmp_path / "micro").resolve())
    assert os.environ["SIMULATION_POLICY_CONV_STATE_ONLY"] == "true"
    assert os.environ["SIMULATION_ALLOW_TERMINATED_MACRO_RESELECTION"] == "false"
    assert os.environ["SIMULATION_FILTER_MACROS_BY_TERMINATION"] == "false"
    assert os.environ["SIMULATION_MAX_MACRO_TURNS"] == "-1"
    assert os.environ["THERAPIST_HF_TORCH_DTYPE"] == "float16"
    assert Path(os.environ["TOPA_COMPONENTS_DIR"]).name == "expert"


def test_prompt_only_ablation_uses_bundled_latest_components(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("STUDY_MODEL_ROOT", str(tmp_path / "models"))
    method_worker.configure_runtime("base", "topas_no_rl")

    components = Path(os.environ["TOPA_COMPONENTS_DIR"])
    policy = build_policy(FakeGenerator(), "topas_no_rl", components, CBT_DOMAIN)

    assert isinstance(policy, PromptActionPolicy)
    assert components.is_dir()
    assert len(policy.macro_actions) == len(policy.micro_action_groups) == 12
    assert os.environ["THERAPIST_HF_TORCH_DTYPE"] == "bf16"
    assert os.environ["TOPAS_VALIDATION_MAX_RETRIES"] == "5"
    assert os.environ["TOPAS_NO_RL_STATE_MAX_NEW_TOKENS"] == "2048"


def test_chatgpt_component_worker_uses_its_own_components_and_checkpoints(
    monkeypatch, tmp_path: Path
):
    monkeypatch.setenv("STUDY_MODEL_ROOT", str(tmp_path / "models"))
    monkeypatch.setenv("STUDY_TOPAS_CHAT_CONV_STATE_DIR", str(tmp_path / "chat-conv"))
    monkeypatch.setenv("STUDY_TOPAS_CHAT_MACRO_POLICY_DIR", str(tmp_path / "chat-macro"))
    monkeypatch.setenv("STUDY_TOPAS_CHAT_MICRO_POLICY_DIR", str(tmp_path / "chat-micro"))

    method_worker.configure_runtime("base", "topas_chatgpt_components")

    assert Path(os.environ["TOPA_COMPONENTS_DIR"]).name == "chatgpt"
    assert os.environ["SIMULATION_CONV_STATE_DIR"] == str((tmp_path / "chat-conv").resolve())
    assert os.environ["SIMULATION_MACRO_POLICY_DIR"] == str((tmp_path / "chat-macro").resolve())
    assert os.environ["SIMULATION_MICRO_POLICY_DIR"] == str((tmp_path / "chat-micro").resolve())
    assert os.environ["SIMULATION_POLICY_CONV_STATE_ONLY"] == "true"
    assert os.environ["THERAPIST_HF_TORCH_DTYPE"] == "float16"


def test_no_conversation_state_ablation_uses_saved_checkpoint_action_space(
    monkeypatch, tmp_path: Path
):
    monkeypatch.setenv("STUDY_MODEL_ROOT", str(tmp_path / "models"))
    monkeypatch.setenv("STUDY_TOPAS_NO_CONV_RUNS_DIR", str(tmp_path / "no-conv"))
    method_worker.configure_runtime("base", "topas_no_conv_state")

    macro = (tmp_path / "no-conv" / "iql_policy_over_options_acts_L22").resolve()
    assert os.environ["SIMULATION_MACRO_POLICY_DIR"] == str(macro)
    assert os.environ["TOPA_COMPONENTS_DIR"] == str(macro / "used_action_space.json")
    assert os.environ["SIMULATION_POLICY_CONV_STATE_ONLY"] == "auto"
    assert os.environ["SIMULATION_USE_MEMORY"] == "false"
    assert os.environ["SIMULATION_GENERATION_MEMORY"] == "false"


def test_topas_full_matches_the_full_memory_launcher(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("STUDY_MODEL_ROOT", str(tmp_path / "models"))
    monkeypatch.setenv("STUDY_TOPAS_FULL_RUNS_DIR", str(tmp_path / "old-runs"))
    monkeypatch.setenv("STUDY_TOPAS_FULL_MACRO_POLICY_DIR", str(tmp_path / "macro"))
    monkeypatch.setenv("STUDY_TOPAS_FULL_MICRO_POLICY_DIR", str(tmp_path / "micro"))
    monkeypatch.setenv("STUDY_TOPAS_FULL_CONV_STATE_DIR", str(tmp_path / "conv"))
    monkeypatch.setenv(
        "STUDY_TOPAS_FULL_MEMORY_RETRIEVER_DIR", str(tmp_path / "retriever")
    )
    monkeypatch.setenv("STUDY_TOPAS_FULL_KG_SCHEMA_PATH", str(tmp_path / "kg.json"))

    method_worker.configure_runtime("base", "topas_full")

    assert os.environ["SIMULATION_RUNS_DIR"] == str((tmp_path / "old-runs").resolve())
    assert os.environ["SIMULATION_MACRO_POLICY_DIR"] == str((tmp_path / "macro").resolve())
    assert os.environ["SIMULATION_MICRO_POLICY_DIR"] == str((tmp_path / "micro").resolve())
    assert os.environ["SIMULATION_CONV_STATE_DIR"] == str((tmp_path / "conv").resolve())
    assert os.environ["SIMULATION_MEMORY_RETRIEVER_DIR"] == str(
        (tmp_path / "retriever").resolve()
    )
    assert os.environ["SIMULATION_KG_SCHEMA_PATH"] == str((tmp_path / "kg.json").resolve())
    assert os.environ["SIMULATION_POLICY_CONV_STATE_ONLY"] == "auto"
    assert os.environ["SIMULATION_MEMORY_MODE"] == "retrieved"
    assert os.environ["SIMULATION_USE_MEMORY"] == "false"
    assert os.environ["SIMULATION_GENERATION_MEMORY"] == "true"
    assert os.environ["SIMULATION_MEMORY_RETRIEVAL_DETERMINISTIC"] == "false"
    assert os.environ["SIMULATION_MEMORY_RETRIEVAL_AVAILABLE_ONLY"] == "true"
    assert os.environ["SIMULATION_MEMORY_RETRIEVAL_TOP_K"] == "3"
    assert os.environ["SIMULATION_REQUIRE_FULL_GENERATION_MEMORY"] == "true"
    assert os.environ["SIMULATION_MEMORY_MAX_TOKENS"] == "0"
    assert os.environ["THERAPIST_HF_TORCH_DTYPE"] == "float16"
    assert os.environ["THERAPIST_TEMPERATURE"] == "0.7"
    assert os.environ["THERAPIST_TOP_P"] == "0.9"
    assert Path(os.environ["TOPA_COMPONENTS_DIR"]).name == "no_rl"


def test_all_ablation_methods_use_the_base_conda_runtime():
    from server.inference_manager import RUNTIME_BY_METHOD

    assert {
        method: RUNTIME_BY_METHOD[method]
        for method in method_worker.ABLATION_METHODS
    } == {method: "base" for method in method_worker.ABLATION_METHODS}
