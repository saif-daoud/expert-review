import json
import os
import re
from dataclasses import asdict
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Tuple

from .config import DOMAIN, DomainSpec
from .llm import TextGenerator, finish_utterance
from .simulators.base import strip_speaker_prefixes


def _topa_module() -> Any:
    """Load TOPAS only when an activation-policy method is requested."""
    from . import topa_agent

    return topa_agent


def build_agent(*args: Any, **kwargs: Any) -> Any:
    """Compatibility wrapper that keeps TOPAS imports lazy."""
    return _topa_module().build_agent(*args, **kwargs)


@lru_cache(maxsize=4)
def _activation_reference(path: str, modified_ns: int) -> Dict[str, Any]:
    import torch
    obj = torch.load(path, map_location="cpu")
    keys = ("indexing", "layers", "system_tag", "user_tag", "system_token_id", "model_name",
            "window_utterances", "use_memory", "memory_max_tokens", "memory_alignment")
    return {**{k: obj[k] for k in keys if k in obj},
            "dtype": obj["activations"].dtype, "hidden_dim": int(obj["activations"].shape[-1])}


def _kg_backend() -> Dict[str, str]:
    endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT", "")
    provider = (os.environ.get("TOPA_LLM_PROVIDER") or ("openai" if "/openai/v1" in endpoint else "azure")).strip().lower()
    model_var = "OPENAI_MODEL" if provider == "openai" else "AZURE_OPENAI_DEPLOYMENT"
    model = os.environ.get("SIMULATION_KG_MODEL") or os.environ.get(model_var) or "gpt-4.1"
    return {"provider": provider, "model": model, "endpoint": endpoint}


def _activation_directories(domain: DomainSpec):
    topa = _topa_module()
    runs = Path(os.environ.get("SIMULATION_RUNS_DIR") or os.environ.get("TOPA_RL_RUNS_DIR")
                or (topa.SIMULATIONS_DIR / "checkpoints" / f"{domain.name}_topa"))
    conv = Path(os.environ.get("SIMULATION_CONV_STATE_DIR") or runs / "sft_conv_state_acts_L22")
    return runs, conv


def activation_config(domain: DomainSpec):
    """Actual HF layer and input settings, independent of a saved tensor's row index."""
    topa = _topa_module()
    use_memory = os.environ.get("SIMULATION_USE_MEMORY", "false").strip().lower()
    if use_memory not in {"true", "false"}:
        raise ValueError("SIMULATION_USE_MEMORY must be true or false")
    generation_memory = os.environ.get("SIMULATION_GENERATION_MEMORY", "false").strip().lower()
    if generation_memory not in {"true", "false"}:
        raise ValueError("SIMULATION_GENERATION_MEMORY must be true or false")
    require_full = os.environ.get("SIMULATION_REQUIRE_FULL_GENERATION_MEMORY", "false").lower()
    if require_full not in {"true", "false"}:
        raise ValueError("SIMULATION_REQUIRE_FULL_GENERATION_MEMORY must be true or false")
    memory_deterministic = os.environ.get("SIMULATION_MEMORY_RETRIEVAL_DETERMINISTIC", "true").strip().lower()
    if memory_deterministic not in {"true", "false"}:
        raise ValueError("SIMULATION_MEMORY_RETRIEVAL_DETERMINISTIC must be true or false")
    memory_available_only = os.environ.get("SIMULATION_MEMORY_RETRIEVAL_AVAILABLE_ONLY", "true").strip().lower()
    if memory_available_only not in {"true", "false"}:
        raise ValueError("SIMULATION_MEMORY_RETRIEVAL_AVAILABLE_ONLY must be true or false")
    memory_top_k = int(os.environ.get("SIMULATION_MEMORY_RETRIEVAL_TOP_K", "1"))
    if memory_top_k < 1 or (memory_top_k > 1 and memory_available_only != "true"):
        raise ValueError("SIMULATION_MEMORY_RETRIEVAL_TOP_K must be positive; values above 1 require available-only retrieval")
    cfg = topa.LatentConfig(
        layer=int(os.environ.get("SIMULATION_ACTIVATION_LAYER", "22")),
        system_tag=f"<|{domain.system_role.lower()}|>",
        user_tag=f"<|{domain.user_role.lower()}|>",
        prefix=os.environ.get("SIMULATION_ACTIVATION_PREFIX", f"Act as the {domain.system_role.lower()} agent and continue the conversation: "),
        window_utterances=int(os.environ.get("SIMULATION_ACTIVATION_WINDOW", "5")),
        use_memory=use_memory == "true",
        memory_max_tokens=int(os.environ.get("SIMULATION_MEMORY_MAX_TOKENS", "8192")),
        use_generation_memory=generation_memory == "true",
        memory_mode=os.environ.get("SIMULATION_MEMORY_MODE", "legacy"),
        memory_retriever_dir=os.environ.get("SIMULATION_MEMORY_RETRIEVER_DIR", ""),
        require_full_generation_memory=require_full == "true",
        memory_retrieval_deterministic=memory_deterministic == "true",
        memory_retrieval_available_only=memory_available_only == "true",
        memory_retrieval_top_k=memory_top_k,
    )
    reference_path = os.environ.get("SIMULATION_ACTIVATION_REFERENCE")
    if reference_path:
        reference = _activation_reference(reference_path, Path(reference_path).stat().st_mtime_ns)
        if reference.get("indexing") != "system_turn_idx":
            raise ValueError("Activation reference must contain one vector per system turn")
        layers = reference["layers"]
        if len(layers) == 1 and "SIMULATION_ACTIVATION_LAYER" not in os.environ:
            cfg.layer = int(layers[0])
        for env, field in [("SIMULATION_USE_MEMORY", "use_memory"),
                           ("SIMULATION_ACTIVATION_WINDOW", "window_utterances"),
                           ("SIMULATION_MEMORY_MAX_TOKENS", "memory_max_tokens")]:
            if field not in reference:
                continue
            if env not in os.environ:
                setattr(cfg, field, reference[field])
            elif getattr(cfg, field) != reference[field]:
                raise ValueError(f"{env} disagrees with the training activation reference")
        if cfg.system_tag != reference["system_tag"] or cfg.user_tag != reference["user_tag"]:
            raise ValueError("Training activation role tags disagree with the simulation domain")
        if cfg.layer not in layers:
            raise ValueError("SIMULATION_ACTIVATION_LAYER is absent from the training reference's layers")
        if cfg.use_memory and reference.get("memory_alignment") != "latest_completed_phase_v1":
            raise ValueError("Training reference does not use completed-phase KG memory")
    return cfg


def activation_run_settings(domain: DomainSpec, variant: str, components_dir: Path) -> Dict[str, Any]:
    """Small provenance record to prevent resuming incompatible policy runs."""
    topa = _topa_module()
    runs, conv = _activation_directories(domain)
    paths = topa.variant_paths(
        variant, runs_dir=runs,
        action_space_path=components_dir,
        conv_state_dir=conv,
        macro_policy_dir=(None if os.environ.get("SIMULATION_MACRO_POLICY_DIR", "none").lower() == "none" else os.environ["SIMULATION_MACRO_POLICY_DIR"]),
        micro_policy_dir=(None if os.environ.get("SIMULATION_MICRO_POLICY_DIR", "none").lower() == "none" else os.environ["SIMULATION_MICRO_POLICY_DIR"]),
    )
    files = [Path(p) for p in paths.values()]
    files += [Path(__file__).parent / name for name in
              ("agents.py", "topa_agent.py", "topa_models.py", "memory_policy.py", "session.py", "llm.py")]
    retriever_dir = os.environ.get("SIMULATION_MEMORY_RETRIEVER_DIR")
    if retriever_dir:
        files += [Path(retriever_dir) / name for name in ("retriever.pt", "run_config.json", "candidate_masks.npy")]
    if "intra_dir" in paths:
        files += list(Path(paths["intra_dir"]).glob("micro_*/actor.pt"))
        files += list(Path(paths["intra_dir"]).glob("micro_*/model_meta.json"))
    if "conv_state_dir" in paths:
        files += [Path(paths["conv_state_dir"]) / n for n in ("actor.pt", "label_map.json")]
    files += [Path(paths["macro_actor"]).parent / "used_action_space.json"]
    if components_dir.is_dir():
        files += [components_dir / name for name in ("macro_actions.json", "micro_actions.json", "conversation_states.json", "knowledge_graph.json")]
    schema = os.environ.get("SIMULATION_KG_SCHEMA_PATH")
    if schema:
        files.append(Path(schema))
    names = ("SIMULATION_MAX_MACRO_TURNS", "SIMULATION_TERMINATION_THRESHOLD",
             "SIMULATION_POLICY_CONV_STATE_ONLY", "SIMULATION_MACRO_POLICY_DETERMINISTIC",
             "SIMULATION_TERMINATION_DETERMINISTIC", "SIMULATION_MICRO_POLICY_DETERMINISTIC",
             "SIMULATION_FILTER_MACROS_BY_TERMINATION", "SIMULATION_ALLOW_TERMINATED_MACRO_RESELECTION",
             "SIMULATION_CONTEXT_TURNS", "SIMULATION_CONV_STATE_UPDATE_INTERVAL",
             "SIMULATION_KG_SCHEMA_PATH", "SIMULATION_KG_INITIAL_GRAPH", "SIMULATION_SESSION_IDX",
             "SIMULATION_KG_MODEL", "SIMULATION_ACTIVATION_REFERENCE",
             "SIMULATION_MEMORY_MODE", "SIMULATION_MEMORY_RETRIEVER_DIR", "SIMULATION_REQUIRE_FULL_GENERATION_MEMORY",
             "SIMULATION_MEMORY_RETRIEVAL_DETERMINISTIC",
             "SIMULATION_MEMORY_RETRIEVAL_AVAILABLE_ONLY",
             "THERAPIST_BACKEND", "THERAPIST_HF_MODEL", "THERAPIST_HF_TORCH_DTYPE",
             "THERAPIST_HF_ATTN_IMPLEMENTATION")
    latent = asdict(activation_config(domain))
    # Old records already describe top-1. Keep its signature unchanged, while
    # making a different cardinality an explicit resume incompatibility.
    if latent.get("memory_retrieval_top_k") == 1:
        latent.pop("memory_retrieval_top_k")
    return {
        "alignment_version": "kg_type_mask_comparison_v3", "variant": variant,
        "latent": latent, "paths": paths,
        "kg_backend": _kg_backend() if (activation_config(domain).use_memory
                                       or activation_config(domain).use_generation_memory) else None,
        "options": {n: os.environ.get(n) for n in names},
        "files": {str(p.resolve()): [p.stat().st_size, p.stat().st_mtime_ns]
                  for p in sorted(set(files)) if p.is_file()},
    }


def _clean(text: str) -> str:
    return finish_utterance(strip_speaker_prefixes(text))


def _dialogue(transcript: str, domain: DomainSpec, *, preserve_text: bool = False) -> List[Dict[str, str]]:
    turns: List[Dict[str, str]] = []
    roles = (domain.system_role, domain.user_role, *domain.user_role_aliases)
    role_pattern = "|".join(re.escape(role) for role in roles)
    if preserve_text:
        boundaries = list(re.finditer(rf"^({role_pattern}): ?", transcript, re.M))
        for index, match in enumerate(boundaries):
            end = boundaries[index + 1].start() if index + 1 < len(boundaries) else len(transcript)
            text = transcript[match.end():end]
            if text.endswith("\n"):
                text = text[:-1]
            speaker = domain.system_role if match.group(1).lower() == domain.system_role.lower() else domain.user_role
            turns.append({"speaker": speaker, "text": text})
        return turns
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


def _raw_policy_prompt(
    transcript: str,
    system_prompt: str,
    domain: DomainSpec,
) -> str:
    lines = [system_prompt, "", domain.dialogue_heading]
    lines.extend(
        f"{turn['speaker']}: {turn['text']}" for turn in _dialogue(transcript, domain)
    )
    lines.append(f"{domain.system_role}:")
    return "\n".join(lines)


def generator_info(generator: TextGenerator) -> Dict[str, str]:
    return {
        "backend": generator.__class__.__name__,
        "model": str(
            getattr(generator, "model_label", None)
            or getattr(generator, "deployment", None)
            or getattr(generator, "model_name_or_path", generator.__class__.__name__)
        ),
    }


def _add_prompt_diagnostics(
    call: Dict[str, Any], generator: TextGenerator
) -> Dict[str, Any]:
    if generator.save_prompt_diagnostics:
        call["prompt_diagnostics"] = generator.last_prompt_diagnostics
    return call


def _json_from_text(raw: str) -> Dict[str, Any]:
    text = (raw or "").strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end <= start:
        return {}
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return {}


class ProActPolicy:
    """Generate hidden planning first, then the next system response."""

    def __init__(self, generator: TextGenerator, domain: DomainSpec = DOMAIN) -> None:
        self.generator = generator
        self.domain = domain
        if domain.name == "p4g":
            self.reasoning_system_prompt = (
                "You are planning the next persuader turn in a realistic donation conversation. "
                "Think step by step about the persuadee's concern, the likely obstacle trajectory, and the best respectful persuasive move. "
                "Write concise reasoning only; do not write the final persuader response."
            )
            self.utterance_system_prompt = domain.system_prompt
        else:
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

    def respond(self, transcript: str) -> Tuple[str, Dict[str, Any]]:
        self.generator.reset_turn_stats()
        if self.domain.name == "p4g":
            reasoning_task = (
                "First, think through the persuadee's current stance, any warning signs, and the best respectful persuasion move. "
                "Produce reasoning only, not the persuader response."
            )
            utterance_task = (
                "Now write only the next persuader response. If this is the first persuader turn, "
                "briefly greet the persuadee and introduce the donation conversation naturally."
            )
        else:
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


class TheraMindPolicy:
    """CBT-only, first-session integration of the official TheraMind pipeline."""

    COUNSELOR_SYSTEM_PROMPT = (
        "You are an experienced and empathetic psychological counselor."
    )
    MAX_NEW_TOKENS = 96
    EMOTIONS = {
        "joy",
        "sadness",
        "anger",
        "fear",
        "disgust",
        "surprise",
        "trust",
        "anticipation",
    }
    COOPERATIVE_STRATEGIES = (
        "Interpretation",
        "Confrontation",
        "Invite to Take New Perspectives",
        "Invite to Explore New Actions",
    )
    RESISTANT_STRATEGIES = (
        "Restatement",
        "Reflection of Feelings",
        "Self-disclosure",
        "Inquiring Subjective Information",
        "Inquiring Objective Information",
        "Affirmation and Reassurance",
        "Minimal Encouragement",
        "Answer",
    )

    def __init__(
        self,
        generator: TextGenerator,
        profile: Dict[str, Any],
        domain: DomainSpec = DOMAIN,
    ) -> None:
        if domain.name != "cbt":
            raise ValueError("TheraMind is integrated only for the CBT domain")
        self.generator = generator
        self.domain = domain
        self.profile = profile
        self.patient_record = self._patient_record(profile)
        self.current_therapy: str | None = None
        self.used_strategies: List[str] = []
        self.turn_index = 0

    @staticmethod
    def _patient_record(profile: Dict[str, Any]) -> Dict[str, Any]:
        detailed = profile.get("profile", {})
        demographics = detailed.get("demographics", {})
        return {
            "patient pseudonym": profile.get("name", ""),
            "patient age": demographics.get("age_group", ""),
            "mental health history": profile.get("relevant_history", ""),
            "physical health history": detailed.get("physical_health_history", ""),
            "current problems and symptoms": profile.get("summary", ""),
        }

    def _generate(
        self,
        calls: List[Dict[str, Any]],
        module: str,
        prompt: str,
        *,
        temperature: float,
        stop: List[str] | None = None,
    ) -> str:
        temperature_key = "THERAPIST_TEMPERATURE"
        top_p_key = "THERAPIST_TOP_P"
        previous_temperature = os.environ.get(temperature_key)
        previous_top_p = os.environ.get(top_p_key)
        previous_max_tokens = self.generator.max_new_tokens
        previous_generator_temperature = getattr(self.generator, "temperature", None)
        previous_generator_top_p = getattr(self.generator, "top_p", None)
        try:
            os.environ[temperature_key] = str(temperature)
            os.environ[top_p_key] = "0.75"
            self.generator.max_new_tokens = self.MAX_NEW_TOKENS
            if previous_generator_temperature is not None:
                self.generator.temperature = temperature
            if previous_generator_top_p is not None:
                self.generator.top_p = 0.75
            raw = self.generator.generate(
                system=self.COUNSELOR_SYSTEM_PROMPT,
                user=prompt,
                stop=stop,
            ).strip()
            calls.append(
                {
                    "module": module,
                    "system_prompt": self.COUNSELOR_SYSTEM_PROMPT,
                    "user_prompt": prompt,
                    "raw_output": raw,
                    "temperature": temperature,
                    "top_p": 0.75,
                    "max_new_tokens": self.MAX_NEW_TOKENS,
                    "prompt_diagnostics": dict(
                        self.generator.last_prompt_diagnostics
                    ),
                }
            )
            return raw
        finally:
            self.generator.max_new_tokens = previous_max_tokens
            if previous_generator_temperature is not None:
                self.generator.temperature = previous_generator_temperature
            if previous_generator_top_p is not None:
                self.generator.top_p = previous_generator_top_p
            if previous_temperature is None:
                os.environ.pop(temperature_key, None)
            else:
                os.environ[temperature_key] = previous_temperature
            if previous_top_p is None:
                os.environ.pop(top_p_key, None)
            else:
                os.environ[top_p_key] = previous_top_p

    def _select_initial_therapy(self, calls: List[Dict[str, Any]]) -> str:
        prompt = f"""Select a suitable psychological treatment for this patient record.
You may choose one therapy or combine at most two therapies with " + ".
Return only the professional therapy name, without explanation.

Patient medical record:
{json.dumps(self.patient_record, ensure_ascii=False, indent=2)}"""
        raw = self._generate(
            calls,
            "initial_therapy_selection",
            prompt,
            temperature=0.3,
        )
        therapy = raw.strip().strip('"').rstrip(". ")
        return therapy or "cognitive-behavioral therapy"

    def _should_end(self, patient_input: str, calls: List[Dict[str, Any]]) -> bool:
        prompt = f"""Decide whether the patient clearly intends to end the current session.
Examples include saying goodbye, saying that is all for today, or asking to continue next time.
Return only True or False.

Patient's current words:
{patient_input}"""
        raw = self._generate(
            calls,
            "session_end_detection",
            prompt,
            temperature=0.0,
        )
        return raw.strip().lower() == "true"

    def _assess_resistance(
        self, patient_input: str, calls: List[Dict[str, Any]]
    ) -> bool:
        prompt = f"""Determine whether the patient's current words show resistance or a substantial departure from the counseling topic.
Resistance includes directly rejecting the question or intervention, refusing to continue, marked impatience, or shifting to an unrelated topic.
Return only True or False.

Patient's current words:
{patient_input}"""
        raw = self._generate(
            calls,
            "reaction_classification",
            prompt,
            temperature=0.0,
        )
        return raw.strip().lower() == "true"

    def _assess_emotion(
        self, patient_input: str, calls: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        prompt = f"""Identify the patient's strongest current emotion and its intensity.
The emotion must be exactly one of: joy, sadness, anger, fear, disgust, surprise, trust, anticipation.
Intensity must be a number from 0 to 1 rounded to one decimal place.

Patient's current words:
{patient_input}

Return JSON only:
{{"primary_emotion": "emotion", "emotional_intensity": 0.0}}"""
        raw = self._generate(
            calls,
            "emotion_perception",
            prompt,
            temperature=0.3,
        )
        parsed = _json_from_text(raw)
        emotion = str(parsed.get("primary_emotion", "trust")).strip().lower()
        if emotion not in self.EMOTIONS:
            emotion = "trust"
        try:
            intensity = min(1.0, max(0.0, float(parsed.get("emotional_intensity", 0.0))))
        except (TypeError, ValueError):
            intensity = 0.0
        return {"primary_emotion": emotion, "emotional_intensity": intensity}

    def _treatment_stage(
        self, transcript: str, calls: List[Dict[str, Any]]
    ) -> str:
        prompt = f"""Analyze the current stage of treatment in no more than 80 words.
Summarize what has been addressed and state how treatment should proceed next.
Return one concise paragraph only.

Current therapy:
{self.current_therapy}

Current-session dialogue:
{transcript}"""
        raw = self._generate(
            calls,
            "treatment_stage",
            prompt,
            temperature=0.3,
        )
        return raw or "Initial assessment and therapeutic engagement."

    def _memory(
        self,
        transcript: str,
        patient_input: str,
        calls: List[Dict[str, Any]],
    ) -> str:
        prompt = f"""Determine whether earlier dialogue is needed to respond to the patient's current words.
Use earlier dialogue only when it contains clearly related, reasonably recent information.
If it is useful, return an accurate summary of no more than 50 words.
Otherwise, return exactly: No need to consider historical conversation memory

Current-session dialogue:
{transcript}

Patient's current words:
{patient_input}

Return only the summary or the specified sentence, without analysis or explanation."""
        raw = self._generate(
            calls,
            "memory_retrieval",
            prompt,
            temperature=0.3,
        )
        return raw or "No need to consider historical conversation memory"

    def _response_strategy(
        self,
        patient_input: str,
        emotion: Dict[str, Any],
        resistant: bool,
        calls: List[Dict[str, Any]],
    ) -> Dict[str, str]:
        allowed = (
            self.RESISTANT_STRATEGIES if resistant else self.COOPERATIVE_STRATEGIES
        )
        strategy_catalog = {
            "Interpretation": "Explain the meaning of the patient's words or behavior from a useful perspective.",
            "Confrontation": "Point out an inconsistency or unhelpful belief and invite the patient to face it.",
            "Invite to Take New Perspectives": "Guide the patient toward another way of viewing the problem.",
            "Invite to Explore New Actions": "Encourage a constructive behavior or practical next step.",
            "Restatement": "Restate the patient's meaning to confirm understanding and communicate attention.",
            "Reflection of Feelings": "Name and reflect the patient's feelings to support recognition and acceptance.",
            "Self-disclosure": "Use a brief relevant self-disclosure to establish resonance and trust.",
            "Inquiring Subjective Information": "Ask about the patient's thoughts, feelings, or expectations.",
            "Inquiring Objective Information": "Ask for concrete facts needed to understand the situation.",
            "Affirmation and Reassurance": "Affirm or reassure the patient in a grounded way.",
            "Minimal Encouragement": "Briefly encourage the patient to continue expressing themselves.",
            "Answer": "Directly answer the patient's question with useful information or guidance.",
        }
        options = "\n".join(
            f"- {name}: {strategy_catalog[name]}" for name in allowed
        )
        previous = ", ".join(self.used_strategies) or "None"
        attitude = "resistant" if resistant else "cooperative"
        prompt = f"""Choose one response strategy for the counselor and give concise guidance of no more than 30 words.
Select only from the strategies allowed for the inferred patient attitude. Prefer variety when several choices are clinically reasonable.

Patient's current words: {patient_input}
Primary emotion: {emotion['primary_emotion']}
Emotional intensity: {emotion['emotional_intensity']:.1f}
Patient attitude: {attitude}
Strategies already used this session: {previous}

Allowed strategies:
{options}

Return JSON only:
{{"strategy": "exact strategy name", "strategy_text": "brief response guidance"}}"""
        raw = self._generate(
            calls,
            "response_strategy",
            prompt,
            temperature=0.5,
        )
        parsed = _json_from_text(raw)
        strategy = str(parsed.get("strategy", "")).strip()
        if strategy not in allowed:
            strategy = allowed[0]
        guidance = str(parsed.get("strategy_text", "")).strip()
        if not guidance:
            guidance = strategy_catalog[strategy]
        self.used_strategies.append(strategy)
        return {"strategy": strategy, "strategy_text": guidance}

    def _response(
        self,
        transcript: str,
        patient_input: str,
        memory: str,
        emotion: Dict[str, Any],
        resistant: bool,
        stage: str,
        strategy: Dict[str, str],
        should_end: bool,
        calls: List[Dict[str, Any]],
    ) -> str:
        prompt = f"""Write the counselor's next response using all of the clinical guidance below.

Patient's current words: {patient_input}
Relevant longitudinal memory: {memory}
Primary emotion: {emotion['primary_emotion']}
Emotional intensity: {emotion['emotional_intensity']:.1f}
Patient attitude: {'resistant' if resistant else 'cooperative'}
Current therapy: {self.current_therapy}
Current treatment stage: {stage}
Response strategy: {strategy['strategy']}
Strategy guidance: {strategy['strategy_text']}

Current-session dialogue:
{transcript}

Respond naturally and compassionately in no more than 60 words. Avoid merely repeating the patient's words and vary the wording from earlier counselor responses. {'Close the session in a declarative tone.' if should_end else ''}
Return only the counselor response, with no label, analysis, or explanation."""
        return self._generate(
            calls,
            "response_generation",
            prompt,
            temperature=0.9,
            stop=self.domain.stop_tokens,
        )

    def respond(self, transcript: str) -> Tuple[str, Dict[str, Any]]:
        self.generator.reset_turn_stats()
        calls: List[Dict[str, Any]] = []
        self.turn_index += 1

        if self.current_therapy is None:
            self.current_therapy = self._select_initial_therapy(calls)
            raw = (
                "Hello, I'm your psychological counselor. Nice to meet you. "
                "Today we can talk about your recent situation."
            )
            metadata = {
                "current_therapy": self.current_therapy,
                "cross_session_loop": "inactive: first-session evaluation",
                "module_calls": calls,
            }
        else:
            dialogue = _dialogue(transcript, self.domain)
            patient_turns = [
                turn["text"]
                for turn in dialogue
                if turn["speaker"] == self.domain.user_role
            ]
            patient_input = patient_turns[-1] if patient_turns else "Hello"
            should_end = self._should_end(patient_input, calls)
            resistant = self._assess_resistance(patient_input, calls)
            emotion = self._assess_emotion(patient_input, calls)
            memory = self._memory(transcript, patient_input, calls)
            stage = self._treatment_stage(transcript, calls)
            strategy = self._response_strategy(
                patient_input, emotion, resistant, calls
            )
            raw = self._response(
                transcript,
                patient_input,
                memory,
                emotion,
                resistant,
                stage,
                strategy,
                should_end,
                calls,
            )
            metadata = {
                "patient_state": {
                    **emotion,
                    "attitude": "resistant" if resistant else "cooperative",
                },
                "memory": memory,
                "current_therapy": self.current_therapy,
                "current_treatment_stage": stage,
                "response_strategy": strategy,
                "session_end_detected": should_end,
                "cross_session_loop": "inactive: first-session evaluation",
                "module_calls": calls,
            }

        call = {
            "utterance_system_prompt": self.COUNSELOR_SYSTEM_PROMPT,
            "utterance_user_prompt": calls[-1]["user_prompt"] if calls else "",
            "utterance_raw_output": raw,
            "theramind_metadata": metadata,
            "turn_stats": self.generator.turn_stats(),
        }
        if metadata.get("session_end_detected"):
            call["terminate_session"] = True
            call["termination_reason"] = "TheraMind session-end detector"
        return _clean(raw), call


class PromptingPolicy:
    def __init__(self, generator: TextGenerator, domain: DomainSpec = DOMAIN) -> None:
        self.generator = generator
        self.domain = domain
        default_prompt = (
            Path(__file__).resolve().parents[1]
            / "baselines"
            / "sweet_rl_cbt"
            / "prompts"
            / domain.policy_prompt_filename
        )
        prompt_path = Path(
            os.environ.get("THERAPIST_PROMPTING_SYSTEM_PROMPT_PATH", default_prompt)
        )
        self.system_prompt = prompt_path.read_text(encoding="utf-8").strip()

    def respond(self, transcript: str) -> Tuple[str, Dict[str, Any]]:
        full_prompt = _raw_policy_prompt(
            transcript,
            self.system_prompt,
            self.domain,
        )
        raw = self.generator.generate(system="", user=full_prompt, stop=self.domain.stop_tokens)
        call = {
            "utterance_system_prompt": self.system_prompt,
            "utterance_user_prompt": full_prompt,
            "utterance_raw_output": raw,
        }
        return _clean(raw), _add_prompt_diagnostics(call, self.generator)


class PromptActionPolicy:
    """Prompt-only macro/micro action ablation without RL or conversation states."""

    def __init__(
        self,
        generator: TextGenerator,
        components_dir: Path,
        domain: DomainSpec = DOMAIN,
    ) -> None:
        self.generator = generator
        self.domain = domain
        self.macro_actions: List[Dict[str, Any]] = json.loads(
            (components_dir / "macro_actions.json").read_text(encoding="utf-8")
        )["macro_actions"]
        self.micro_action_groups: List[Dict[str, Any]] = json.loads(
            (components_dir / "micro_actions.json").read_text(encoding="utf-8")
        )
        self.previous_macro_action: str | None = None

        domain_name = "persuasion" if domain.name == "p4g" else "CBT"
        self.macro_system_prompt = (
            f"You are a {domain_name} planner choosing the next macro action "
            "(dialogue phase). Return JSON only."
        )
        response_name = (
            "persuader response" if domain.name == "p4g" else "therapist utterance"
        )
        self.utterance_system_prompt = (
            f"{domain.system_prompt}\n\n"
            "For this turn, follow the supplied macro action and its objective. You will be "
            "given the last 3 dialogue utterances and the candidate micro actions belonging to "
            f"that macro action. Choose the best-fitting micro action and write only the next "
            f"{response_name}. Do not include action names, JSON, reasoning, labels, or role prefixes."
        )

    @staticmethod
    def _lines(transcript: str) -> List[str]:
        return [line.strip() for line in transcript.splitlines() if line.strip()]

    def _last_utterances(self, transcript: str, count: int = 3) -> str:
        lines = self._lines(transcript)
        return "\n".join(lines[-count:]) if lines else "NONE"

    def _macro_catalog(self) -> str:
        lines: List[str] = []
        for index, macro in enumerate(self.macro_actions, start=1):
            lines.append(f"{index}. {macro['name']}")
            lines.append(
                f"   objective: {(macro.get('goal') or {}).get('objective', '')}"
            )
        return "\n".join(lines)

    def _select_macro(self, transcript: str) -> Tuple[Dict[str, Any], str, str]:
        previous = self.previous_macro_action or "NONE"
        recent_context = self._last_utterances(transcript, count=3)
        user_prompt = f"""Previous macro action: {previous}

Last 3 dialogue utterances:
{recent_context}

Available macro actions:
{self._macro_catalog()}

Choose the best next macro action.
Decide whether to:
1. CONTINUE the current macro action
2. SWITCH to a different macro action

Return JSON only with this exact structure:
{{
  "continue_previous": <true or false>,
  "selected_macro_action": "<exact macro action name>",
  "rationale": "<brief reason>"
}}

If previous macro action is NONE, continue_previous must be false. JSON only."""
        raw = self.generator.generate(
            system=self.macro_system_prompt,
            user=user_prompt,
            stop=self.domain.stop_tokens,
        ).strip()
        parsed = _json_from_text(raw)
        valid_names = {str(macro["name"]) for macro in self.macro_actions}
        selected = str(
            parsed.get("selected_macro_action")
            or self.previous_macro_action
            or self.macro_actions[0]["name"]
        )
        if selected not in valid_names:
            selected = str(self.macro_actions[0]["name"])
        continue_previous = (
            bool(parsed.get("continue_previous"))
            and self.previous_macro_action is not None
        )
        if continue_previous:
            selected = str(self.previous_macro_action)
        metadata = next(macro for macro in self.macro_actions if macro["name"] == selected)
        decision = {
            "continue_previous": continue_previous,
            "selected_macro_action": selected,
            "rationale": str(parsed.get("rationale", "")),
            "macro_metadata": metadata,
        }
        self.previous_macro_action = selected
        return decision, raw, user_prompt

    def _micro_catalog(self, macro_name: str) -> str:
        group = next(
            group for group in self.micro_action_groups if group["name"] == macro_name
        )
        return "\n".join(
            f"- {micro['name']}: {micro['description']}"
            for micro in group["micro_actions"]
        )

    def _generate_utterance(
        self, transcript: str, macro_decision: Dict[str, Any]
    ) -> Tuple[str, str]:
        macro = macro_decision["macro_metadata"]
        role = "persuader response" if self.domain.name == "p4g" else "therapist utterance"
        user_prompt = f"""Last 3 utterances of dialogue:
{self._last_utterances(transcript)}

Macro action to follow for this turn:
{macro['name']}

Macro-action goal objective:
{(macro.get('goal') or {}).get('objective', '')}

Available micro actions:
{self._micro_catalog(str(macro['name']))}

Choose one best-fitting micro action and execute it in the next {role}.
Requirements:
- Write ONLY the next {role}.
- Use the macro action and its objective as the core instruction for this turn.
- Internally choose the micro action that best advances that objective.
- Make the utterance concrete and action-guiding.
- Do not output the action names, reasoning, or JSON."""
        raw = self.generator.generate(
            system=self.utterance_system_prompt,
            user=user_prompt,
            stop=self.domain.stop_tokens + ["Reasoning:"],
        ).strip()
        return raw, user_prompt

    def respond(self, transcript: str) -> Tuple[str, Dict[str, Any]]:
        self.generator.reset_turn_stats()
        macro, macro_raw, macro_user_prompt = self._select_macro(transcript)
        macro_diagnostics = dict(self.generator.last_prompt_diagnostics)
        utterance_raw, utterance_user_prompt = self._generate_utterance(transcript, macro)
        call = {
            "macro_system_prompt": self.macro_system_prompt,
            "macro_user_prompt": macro_user_prompt,
            "macro_raw_output": macro_raw,
            "macro_decision": macro,
            "macro_prompt_diagnostics": macro_diagnostics,
            "utterance_system_prompt": self.utterance_system_prompt,
            "utterance_user_prompt": utterance_user_prompt,
            "utterance_raw_output": utterance_raw,
            "utterance_prompt_diagnostics": dict(self.generator.last_prompt_diagnostics),
            "turn_stats": self.generator.turn_stats(),
        }
        return _clean(utterance_raw), call


class GeneratorPolicy:
    def __init__(self, generator: TextGenerator, domain: DomainSpec = DOMAIN) -> None:
        self.generator = generator
        self.domain = domain

    def respond(self, transcript: str) -> Tuple[str, Dict[str, Any]]:
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


class ActivationPolicy:
    def __init__(
        self,
        generator: TextGenerator,
        variant_name: str,
        components_dir: Path,
        domain: DomainSpec = DOMAIN,
    ) -> None:
        self.generator = generator
        self.variant_name = variant_name
        self.domain = domain
        if hasattr(self.generator, "max_input_length"):
            self.generator.max_input_length = None
        conv_state_only = os.environ.get("SIMULATION_POLICY_CONV_STATE_ONLY", "auto").lower()
        if conv_state_only not in {"true", "false", "auto"}:
            raise ValueError("SIMULATION_POLICY_CONV_STATE_ONLY must be true, false or auto")
        micro_policy_deterministic = os.environ.get(
            "SIMULATION_MICRO_POLICY_DETERMINISTIC", "true"
        )
        if micro_policy_deterministic not in {"true", "false"}:
            raise ValueError("SIMULATION_MICRO_POLICY_DETERMINISTIC must be true or false")
        macro_policy_deterministic = os.environ.get(
            "SIMULATION_MACRO_POLICY_DETERMINISTIC", "true"
        )
        if macro_policy_deterministic not in {"true", "false"}:
            raise ValueError("SIMULATION_MACRO_POLICY_DETERMINISTIC must be true or false")
        termination_deterministic = os.environ.get(
            "SIMULATION_TERMINATION_DETERMINISTIC", "true"
        )
        if termination_deterministic not in {"true", "false"}:
            raise ValueError("SIMULATION_TERMINATION_DETERMINISTIC must be true or false")
        filter_macros_by_termination = os.environ.get(
            "SIMULATION_FILTER_MACROS_BY_TERMINATION", "true"
        )
        if filter_macros_by_termination not in {"true", "false"}:
            raise ValueError("SIMULATION_FILTER_MACROS_BY_TERMINATION must be true or false")
        allow_terminated_macro_reselection = os.environ.get(
            "SIMULATION_ALLOW_TERMINATED_MACRO_RESELECTION", "false"
        )
        if allow_terminated_macro_reselection not in {"true", "false"}:
            raise ValueError(
                "SIMULATION_ALLOW_TERMINATED_MACRO_RESELECTION must be true or false"
            )
        macro_policy_dir = os.environ.get("SIMULATION_MACRO_POLICY_DIR")
        micro_policy_dir = os.environ.get("SIMULATION_MICRO_POLICY_DIR")
        if macro_policy_dir is not None and macro_policy_dir.strip().lower() == "none":
            macro_policy_dir = None
        if micro_policy_dir is not None and micro_policy_dir.strip().lower() == "none":
            micro_policy_dir = None
        topa = _topa_module()
        runs, conv = _activation_directories(domain)
        cfg = activation_config(domain)
        memory = None
        if cfg.use_memory or cfg.use_generation_memory:
            from topa.annotation.annotators.knowledge_graph import KnowledgeGraphAnnotator
            api_key = os.environ.get("TOPA_API_KEY") or os.environ.get("AZURE_OPENAI_API_KEY")
            if not api_key:
                raise ValueError("KG updates require TOPA_API_KEY or AZURE_OPENAI_API_KEY")
            # TOPA's Azure SDK path differs from the v1-compatible endpoint used
            # by this runner. Select the matching existing TOPA client explicitly.
            backend = _kg_backend()
            provider = backend["provider"]
            if provider not in {"openai", "azure"} or not backend["endpoint"]:
                raise ValueError("KG updates require AZURE_OPENAI_ENDPOINT and TOPA_LLM_PROVIDER=openai or azure")
            if "/openai/v1" in backend["endpoint"] and provider != "openai":
                raise ValueError("Azure v1 endpoints require TOPA_LLM_PROVIDER=openai for the KG updater")
            os.environ.setdefault("TOPA_LLM_PROVIDER", provider)
            model_var = "OPENAI_MODEL" if provider == "openai" else "AZURE_OPENAI_DEPLOYMENT"
            os.environ[model_var] = backend["model"]
            schema_dir = components_dir.parent if components_dir.is_file() else components_dir
            schema_path = Path(os.environ.get("SIMULATION_KG_SCHEMA_PATH", str(schema_dir / "knowledge_graph.json")))
            annotator = KnowledgeGraphAnnotator(
                domain=domain.name.upper(), api_key=api_key,
                domain_params={"adj": "cognitive behavioral therapy" if domain.name == "cbt" else "persuasion",
                               "system": domain.system_role.lower(), "user": domain.user_role.lower(),
                               "interaction_unit": "session"},
            )
            memory = topa.KnowledgeGraphMemory(annotator, json.loads(schema_path.read_text(encoding="utf-8")))
        self.agent = build_agent(
            variant_name=variant_name,
            generator=generator,
            action_space_path=components_dir,
            runs_dir=runs,
            conv_state_dir=conv,
            macro_policy_dir=macro_policy_dir,
            micro_policy_dir=micro_policy_dir,
            latent_cfg=cfg,
            memory=memory,
            context_turns=int(os.environ.get("SIMULATION_CONTEXT_TURNS", "5")),
            max_macro_turns=int(os.environ.get("SIMULATION_MAX_MACRO_TURNS", "0")),
            conv_state_update_interval=int(
                os.environ.get("SIMULATION_CONV_STATE_UPDATE_INTERVAL", "1")
            ),
            termination_threshold=float(
                os.environ.get("SIMULATION_TERMINATION_THRESHOLD", "0.5")
            ),
            macro_policy_deterministic=macro_policy_deterministic == "true",
            termination_deterministic=termination_deterministic == "true",
            filter_macros_by_termination=filter_macros_by_termination == "true",
            allow_terminated_macro_reselection=(
                allow_terminated_macro_reselection == "true"
            ),
            micro_policy_deterministic=micro_policy_deterministic == "true",
            policy_conv_state_only=None if conv_state_only == "auto" else conv_state_only == "true",
            system_prompt=domain.system_prompt,
            device=os.environ.get("SIMULATION_DEVICE"),
        )
        reference_path = os.environ.get("SIMULATION_ACTIVATION_REFERENCE")
        if reference_path:
            reference = _activation_reference(reference_path, Path(reference_path).stat().st_mtime_ns)
            if self.generator.tokenizer.convert_tokens_to_ids(cfg.system_tag) != reference.get("system_token_id"):
                raise ValueError("Tokenizer special-token IDs differ from the training reference")
            if next(self.generator.model.parameters()).dtype != reference["dtype"]:
                raise ValueError("HF model dtype differs from the training reference; set THERAPIST_HF_TORCH_DTYPE")
            hidden_dim = getattr(self.generator.model.config, "hidden_size", reference["hidden_dim"])
            if hidden_dim != reference["hidden_dim"]:
                raise ValueError("HF hidden dimension differs from the training activation reference")
            model_name = getattr(self.generator.model.config, "name_or_path", None)
            if model_name and reference.get("model_name") not in (None, "unknown", model_name):
                raise ValueError("HF model identity differs from the training activation reference")

    def respond(self, transcript: str) -> Tuple[str, Dict[str, Any]]:
        return self.respond_dialogue(_dialogue(transcript, self.domain, preserve_text=True), {})

    def reset_session(self, *, user_idx: int, session_idx: int = 0, profile_id: str = "") -> None:
        graph = None
        template = os.environ.get("SIMULATION_KG_INITIAL_GRAPH")
        if template and self.agent.memory is not None:
            path = Path(template.format(user_idx=user_idx, session_idx=session_idx, profile_id=profile_id))
            graph = json.loads(path.read_text(encoding="utf-8"))
        self.agent.reset_session(user_idx=user_idx, session_idx=session_idx, memory_graph=graph)

    def finish_session(self, dialogue: List[Dict[str, str]]):
        return self.agent.finish_session(dialogue)

    def respond_dialogue(
        self, dialogue: List[Dict[str, str]], session_metadata: Dict[str, Any],
    ) -> Tuple[str, Dict[str, Any]]:
        previous_usage = dict(self.agent.memory.usage) if self.agent.memory is not None else {}
        self.agent.begin_turn()
        raw = self.agent.next_system_utterance(dialogue, session_metadata=session_metadata)
        stats = self.agent.end_turn()
        if self.agent.memory is not None:
            stats["kg_update"] = {k: self.agent.memory.usage[k] - previous_usage[k] for k in previous_usage}
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
    components_dir: Path,
    domain: DomainSpec = DOMAIN,
    profile: Dict[str, Any] | None = None,
) -> (
    PromptingPolicy
    | PromptActionPolicy
    | GeneratorPolicy
    | ActivationPolicy
    | ProActPolicy
    | TheraMindPolicy
):
    if method in {"archer", "sweet_rl", "aria"}:
        return GeneratorPolicy(generator, domain)
    if method == "prompting":
        return PromptingPolicy(generator, domain)
    if method == "proact":
        return ProActPolicy(generator, domain)
    if method == "theramind":
        if profile is None:
            raise ValueError("TheraMind requires the CBT patient profile")
        return TheraMindPolicy(generator, profile, domain)
    if method == "topas_no_rl":
        return PromptActionPolicy(generator, components_dir, domain)
    topa = _topa_module()
    if method in topa.SIMULATION_VARIANTS:
        return ActivationPolicy(generator, method, components_dir, domain)
    if method == "simulation_variant":
        return ActivationPolicy(
            generator,
            os.environ["SIMULATION_VARIANT"],
            components_dir,
            domain,
        )
    raise ValueError(f"Unknown policy method: {method}")
