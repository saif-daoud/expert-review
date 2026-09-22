"""CBT therapist policies used by the live interaction workers."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .llm import TextGenerator, finish_utterance


SPEAKER_PREFIX = re.compile(
    r"^\s*(?:(?:assistant|therapist|patient|client|persuader|persuadee)\s*:\s*)+",
    re.IGNORECASE,
)
ASSISTANT_BOUNDARY = re.compile(r"\bassistant\s*:", re.IGNORECASE)


@dataclass(frozen=True)
class DomainSpec:
    name: str
    system_role: str
    user_role: str
    system_prompt: str
    dialogue_heading: str
    policy_prompt_filename: str
    initial_user_utterance: str = ""
    user_role_aliases: tuple[str, ...] = ()

    @property
    def stop_tokens(self) -> list[str]:
        return [
            "\nAssistant:",
            f"\n{self.user_role}:",
            f"\n{self.system_role}:",
            *(f"\n{role}:" for role in self.user_role_aliases),
            "\nSystem:",
        ]


CBT_DOMAIN = DomainSpec(
    name="cbt",
    system_role="Therapist",
    user_role="Patient",
    system_prompt=(
        "You are a CBT therapist in a realistic first session. Write ONLY the next therapist utterance in natural language. "
        "On the first turn, sound like a real therapist opening the session: greet the patient warmly, briefly orient the session, and invite them to begin. "
        "Do not include reasoning, labels, JSON, or role prefixes."
    ),
    dialogue_heading="CBT therapy dialogue",
    policy_prompt_filename="therapist_agent_prompt.txt",
    initial_user_utterance="Hello",
    user_role_aliases=("Client",),
)


def _clean(text: str) -> str:
    cleaned = SPEAKER_PREFIX.sub("", str(text or "")).strip()
    boundary = ASSISTANT_BOUNDARY.search(cleaned)
    if boundary is not None:
        cleaned = cleaned[: boundary.start()].strip()
    return finish_utterance(cleaned)


def _dialogue(transcript: str, domain: DomainSpec) -> list[dict[str, str]]:
    turns: list[dict[str, str]] = []
    roles = (domain.system_role, domain.user_role, *domain.user_role_aliases)
    role_pattern = "|".join(re.escape(role) for role in roles)
    pattern = re.compile(
        rf"({role_pattern}):\s*(.*?)"
        rf"(?=(?:\n)?(?:{role_pattern}):|\Z)",
        re.S,
    )
    for match in pattern.finditer(transcript):
        speaker = (
            domain.system_role
            if match.group(1).lower() == domain.system_role.lower()
            else domain.user_role
        )
        text = " ".join(match.group(2).split()).strip()
        if text:
            turns.append({"speaker": speaker, "text": text})
    return turns


def _raw_policy_prompt(transcript: str, system_prompt: str, domain: DomainSpec) -> str:
    lines = [system_prompt, "", domain.dialogue_heading]
    lines.extend(f"{turn['speaker']}: {turn['text']}" for turn in _dialogue(transcript, domain))
    lines.append(f"{domain.system_role}:")
    return "\n".join(lines)


def _add_prompt_diagnostics(call: dict[str, Any], generator: TextGenerator) -> dict[str, Any]:
    if generator.save_prompt_diagnostics:
        call["prompt_diagnostics"] = generator.last_prompt_diagnostics
    return call


class ProActPolicy:
    def __init__(self, generator: TextGenerator, domain: DomainSpec = CBT_DOMAIN) -> None:
        self.generator = generator
        self.domain = domain
        self.reasoning_system_prompt = (
            "You are a CBT therapist planning the next turn in a realistic first session. "
            "Think step by step about the patient's state, the likely risk trajectory, and the best next therapeutic move. "
            "Write concise reasoning only; do not write the final therapist utterance."
        )
        self.utterance_system_prompt = (
            "You are a CBT therapist in a realistic first session. Given the conversation and the hidden planning notes, "
            "write ONLY the next therapist utterance in natural language. "
            "On the first turn, sound like a real therapist opening the session: greet the patient warmly, briefly orient the session, and invite them to begin. "
            "Do not reveal the planning notes. Do not include reasoning, labels, JSON, or role prefixes."
        )

    def respond(self, transcript: str) -> tuple[str, dict[str, Any]]:
        self.generator.reset_turn_stats()
        reasoning_task = (
            "First, think through the patient's current state, any warning signs, and the best next CBT move. "
            "Produce reasoning only, not the therapist utterance."
        )
        utterance_task = (
            "Now write only the next therapist utterance. If this is the first therapist turn, "
            "begin naturally with a greeting such as hi or hello before inviting the patient to share what brought them in."
        )
        reasoning_user = f"""Conversation so far:
        {transcript}

        {reasoning_task}"""
        reasoning_raw = self.generator.generate(
            system=self.reasoning_system_prompt,
            user=reasoning_user,
            stop=self.domain.stop_tokens,
        ).strip()
        reasoning_diagnostics = dict(self.generator.last_prompt_diagnostics)
        utterance_user = f"""Conversation so far:
        {transcript}

        Hidden planning notes (do not reveal these notes):
        {reasoning_raw}

        {utterance_task}"""
        utterance_raw = self.generator.generate(
            system=self.utterance_system_prompt,
            user=utterance_user,
            stop=self.domain.stop_tokens + ["Reasoning:"],
        ).strip()
        call = {
            "reasoning_system_prompt": self.reasoning_system_prompt,
            "reasoning_user_prompt": reasoning_user,
            "reasoning_raw_output": reasoning_raw,
            "reasoning_prompt_diagnostics": reasoning_diagnostics,
            "utterance_system_prompt": self.utterance_system_prompt,
            "utterance_user_prompt": utterance_user,
            "utterance_raw_output": utterance_raw,
            "utterance_prompt_diagnostics": dict(self.generator.last_prompt_diagnostics),
            "turn_stats": self.generator.turn_stats(),
        }
        return _clean(utterance_raw), call


class PromptingPolicy:
    def __init__(self, generator: TextGenerator, domain: DomainSpec = CBT_DOMAIN) -> None:
        self.generator = generator
        self.domain = domain
        default_prompt = Path(__file__).resolve().parents[1] / "assets" / domain.policy_prompt_filename
        prompt_path = Path(os.environ.get("THERAPIST_PROMPTING_SYSTEM_PROMPT_PATH", default_prompt))
        self.system_prompt = prompt_path.read_text(encoding="utf-8").strip()

    def respond(self, transcript: str) -> tuple[str, dict[str, Any]]:
        full_prompt = _raw_policy_prompt(transcript, self.system_prompt, self.domain)
        raw = self.generator.generate(system="", user=full_prompt, stop=self.domain.stop_tokens)
        call = {
            "utterance_system_prompt": self.system_prompt,
            "utterance_user_prompt": full_prompt,
            "utterance_raw_output": raw,
        }
        return _clean(raw), _add_prompt_diagnostics(call, self.generator)


class GeneratorPolicy:
    def __init__(self, generator: TextGenerator, domain: DomainSpec = CBT_DOMAIN) -> None:
        self.generator = generator
        self.domain = domain

    def respond(self, transcript: str) -> tuple[str, dict[str, Any]]:
        raw = self.generator.generate(
            system=self.domain.system_prompt,
            user=transcript,
            stop=self.domain.stop_tokens,
        )
        diagnostics = self.generator.last_prompt_diagnostics
        system_prompt = (
            getattr(self.generator, "policy_system_prompt", None)
            or getattr(self.generator, "system_instruction", None)
            or self.domain.system_prompt
        )
        call = {
            "utterance_system_prompt": diagnostics.get("system_prompt", system_prompt),
            "utterance_user_prompt": diagnostics.get("user_prompt", transcript),
            "utterance_raw_output": raw,
        }
        return _clean(raw), _add_prompt_diagnostics(call, self.generator)


def _boolean_environment(name: str, default: str) -> bool:
    value = os.environ.get(name, default).strip().lower()
    if value not in {"true", "false"}:
        raise ValueError(f"{name} must be true or false")
    return value == "true"


def _optional_directory_environment(name: str) -> str | None:
    value = os.environ.get(name)
    if value is None or value.strip().lower() == "none":
        return None
    return value


class TOPASPolicy:
    """The live-study wrapper for the simulation's TOPAS activation policy."""

    def __init__(
        self,
        generator: TextGenerator,
        components_dir: Path,
        domain: DomainSpec = CBT_DOMAIN,
    ) -> None:
        # TOPAS extracts layer-22 activations itself. Do not truncate that
        # routing input using the ordinary generation input limit.
        if hasattr(generator, "max_input_length"):
            generator.max_input_length = None

        from .topa_agent import CONV_STATE_DIR, RUNS_DIR, build_agent

        self.generator = generator
        self.domain = domain
        self.variant_name = os.environ.get(
            "STUDY_TOPAS_VARIANT",
            "iql_policy_term_intra_with_conv",
        )
        self.agent = build_agent(
            variant_name=self.variant_name,
            generator=generator,
            action_space_path=components_dir,
            runs_dir=os.environ.get("SIMULATION_RUNS_DIR") or RUNS_DIR,
            conv_state_dir=os.environ.get("SIMULATION_CONV_STATE_DIR") or CONV_STATE_DIR,
            macro_policy_dir=_optional_directory_environment("SIMULATION_MACRO_POLICY_DIR"),
            micro_policy_dir=_optional_directory_environment("SIMULATION_MICRO_POLICY_DIR"),
            context_turns=int(os.environ.get("SIMULATION_CONTEXT_TURNS", "5")),
            max_macro_turns=int(os.environ.get("SIMULATION_MAX_MACRO_TURNS", "-1")),
            conv_state_update_interval=int(
                os.environ.get("SIMULATION_CONV_STATE_UPDATE_INTERVAL", "1")
            ),
            termination_threshold=float(
                os.environ.get("SIMULATION_TERMINATION_THRESHOLD", "0.5")
            ),
            macro_policy_deterministic=_boolean_environment(
                "SIMULATION_MACRO_POLICY_DETERMINISTIC", "false"
            ),
            termination_deterministic=_boolean_environment(
                "SIMULATION_TERMINATION_DETERMINISTIC", "false"
            ),
            filter_macros_by_termination=_boolean_environment(
                "SIMULATION_FILTER_MACROS_BY_TERMINATION", "false"
            ),
            micro_policy_deterministic=_boolean_environment(
                "SIMULATION_MICRO_POLICY_DETERMINISTIC", "false"
            ),
            policy_conv_state_only=_boolean_environment(
                "SIMULATION_POLICY_CONV_STATE_ONLY", "true"
            ),
            system_prompt=domain.system_prompt,
            device=os.environ.get("SIMULATION_DEVICE"),
        )

    def respond(self, transcript: str) -> tuple[str, dict[str, Any]]:
        self.agent.begin_turn()
        raw = self.agent.next_system_utterance(
            _dialogue(transcript, self.domain),
            session_metadata={},
        )
        stats = self.agent.end_turn()
        metadata = self.agent.get_last_turn_metadata()
        metadata["turn_stats"] = stats
        tensor_payload = metadata.pop("tensor_payload", {})
        call = {
            "utterance_system_prompt": metadata.get("utterance_prompt_system", ""),
            "utterance_user_prompt": metadata.get("utterance_prompt_user", ""),
            "utterance_raw_output": metadata.get("raw_output", raw),
            "activation_policy_metadata": metadata,
            "tensor_payload": tensor_payload,
        }
        if metadata.get("terminate_session"):
            call["terminate_session"] = True
            call["termination_reason"] = metadata.get("termination_reason", "")
        return _clean(raw), _add_prompt_diagnostics(call, self.generator)


def build_policy(
    generator: TextGenerator,
    method: str,
    domain: DomainSpec = CBT_DOMAIN,
    components_dir: Path | None = None,
) -> PromptingPolicy | GeneratorPolicy | ProActPolicy | TOPASPolicy:
    if method in {"archer", "sweet_rl", "aria"}:
        return GeneratorPolicy(generator, domain)
    if method == "prompting":
        return PromptingPolicy(generator, domain)
    if method == "proact":
        return ProActPolicy(generator, domain)
    if method == "topas":
        default_components = Path(__file__).resolve().parents[1] / "assets" / "topa_components"
        configured_components = components_dir or Path(
            os.environ.get("STUDY_TOPAS_COMPONENTS_DIR", default_components)
        )
        return TOPASPolicy(generator, configured_components, domain)
    raise ValueError(f"Unknown live-study policy method: {method}")
