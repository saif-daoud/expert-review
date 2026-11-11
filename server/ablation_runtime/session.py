import json
import random
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from tqdm.auto import tqdm

from .config import DomainSpec, RunConfig, get_domain
from .diagnostics import repetition, save_debug_bundle
from .evaluation import Judge, evaluate
from .llm import TextGenerator, finish_utterance
from .simulators import (
    PatientActSimulator,
    PatientPsiBuilder,
    PatientPsiSimulator,
    StandardPromptingP4GSimulator,
    TopasProfileBuilder,
    TopasSimulator,
    TripP4GSimulator,
)
from .simulators.base import farewell_phrase, strip_speaker_prefixes
from .agents import ActivationPolicy, build_policy
from .progress import emit_progress, progress_enabled


def save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def transcript_text(turns: List[Dict[str, Any]], domain: DomainSpec) -> str:
    lines = []
    if domain.initial_user_utterance:
        lines.append(f"{domain.user_role}: {domain.initial_user_utterance}")
    for turn in turns:
        if turn["therapist"]:
            lines.append(f"{domain.system_role}: {turn['therapist']}")
        if turn["patient"]:
            lines.append(f"{domain.user_role}: {turn['patient']}")
    return "\n".join(lines) + ("\n" if lines else "")


def dialogue_turns(turns: List[Dict[str, Any]], domain: DomainSpec) -> List[Dict[str, str]]:
    """Observed utterances for TOPAS, without lossy transcript parsing."""
    dialogue = []
    if domain.initial_user_utterance:
        dialogue.append({"speaker": domain.user_role, "text": domain.initial_user_utterance})
    for turn in turns:
        if turn["therapist"]:
            dialogue.append({"speaker": domain.system_role, "text": turn["therapist"]})
        if turn["patient"]:
            dialogue.append({"speaker": domain.user_role, "text": turn["patient"]})
    return dialogue


def _clean(text: str) -> str:
    return finish_utterance(strip_speaker_prefixes(text))


def _build_patient(
    config: RunConfig,
    patient_generator: TextGenerator,
    profile: Dict[str, Any],
    patient_index: int,
    psi_builder: Optional[PatientPsiBuilder],
    topas_builder: Optional[TopasProfileBuilder],
    case_dir: Path,
) -> (
    PatientActSimulator
    | PatientPsiSimulator
    | TopasSimulator
    | TripP4GSimulator
    | StandardPromptingP4GSimulator
):
    if config.patient_simulator == "patient_act":
        return PatientActSimulator(
            generator=patient_generator,
            case=profile,
            prompt_path=Path(__file__).resolve().parent / "prompts" / "patient_act.yaml",
        )
    if config.patient_simulator == "trip_p4g":
        return TripP4GSimulator(
            patient_generator,
            profile,
            Path(__file__).resolve().parent / "prompts" / "trip_p4g.txt",
        )
    if config.patient_simulator == "standard_prompting":
        return StandardPromptingP4GSimulator(
            patient_generator,
            Path(__file__).resolve().parent / "prompts" / "standard_prompting_p4g.json",
        )
    if config.patient_simulator == "topas":
        if topas_builder is None:
            raise ValueError("TOPAS requires a TopasProfileBuilder")
        populated_profile = topas_builder.build_or_load(profile)
        if config.save_prompt_diagnostics and config.save_mode == "full":
            save_json(
                case_dir / "topas_profile_initialization_prompt.json",
                topas_builder.last_call,
            )
        simulator = TopasSimulator(
            generator=patient_generator,
            profile=populated_profile,
            domain=get_domain(config.domain),
            state_prompt_path=Path(__file__).resolve().parent
            / "prompts"
            / "topas_state_update.txt",
            utterance_prompt_path=Path(__file__).resolve().parent
            / "prompts"
            / (
                "topas.txt"
                if config.topas_use_dynamic_dimensions
                else "topas_static_only.txt"
            ),
            use_dynamic_dimensions=config.topas_use_dynamic_dimensions,
        )
        if config.save_mode == "full":
            save_json(case_dir / "topas_profile.json", simulator.profile)
            save_json(case_dir / "topas_initial_state.json", simulator.current_state)
        return simulator

    if psi_builder is None:
        raise ValueError("PatientPsi requires a PatientPsiBuilder")
    model = psi_builder.build_or_load(
        user_idx=patient_index,
        session_idx=0,
        profile=profile,
    )
    if config.save_mode == "full":
        save_json(case_dir / "patient_psi_cognitive_model.json", model.to_dict())
        save_json(case_dir / "patient_psi_generation_prompt.json", psi_builder.last_call)
    return PatientPsiSimulator(patient_generator, model)


def simulate_case(
    config: RunConfig,
    profile: Dict[str, Any],
    patient_index: int,
    patient_generator: TextGenerator,
    therapist_generator: TextGenerator,
    judges: List[Judge],
    psi_builder: Optional[PatientPsiBuilder] = None,
    topas_builder: Optional[TopasProfileBuilder] = None,
) -> Dict[str, Any]:
    domain = get_domain(config.domain)
    evaluations_only = config.save_mode == "evaluations"
    case_dir = config.output_dir / profile["profile_id"]
    case_dir.mkdir(parents=True, exist_ok=True)
    resume_path = case_dir / "session_resume.pt"
    patient = _build_patient(
        config,
        patient_generator,
        profile,
        patient_index,
        psi_builder,
        topas_builder,
        case_dir,
    )
    policy = build_policy(
        therapist_generator,
        config.therapist_method,
        config.components_dir,
        domain,
        profile,
    )
    if isinstance(policy, ActivationPolicy):
        import os
        policy.reset_session(user_idx=patient_index,
                             session_idx=int(os.environ.get("SIMULATION_SESSION_IDX", "0")),
                             profile_id=profile["profile_id"])
        if policy.agent.latent_cfg.memory_mode != "legacy":
            # Model construction consumes RNG differently when the retriever is
            # loaded. Start matched conditions after all heads are initialized.
            random.seed(config.seed)
            np.random.seed(config.seed)
            torch.manual_seed(config.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(config.seed)

    session_metadata = (
        {
            "presenting conditions": profile["presenting_conditions"],
            "summary": profile["summary"],
        }
        if config.domain == "cbt"
        else {"conversation_id": profile["conversation_id"]}
    )
    intermediate_checkpoints = (50, 100, 150, 200) if config.domain == "cbt" else (10, 15)
    checkpoint_turns = {
        turn
        for turn in (intermediate_checkpoints if evaluations_only else (*intermediate_checkpoints, config.max_turns))
        if turn <= config.max_turns
    }
    checkpoints: Dict[str, Any] = {}
    turns: List[Dict[str, Any]] = []
    termination: Dict[str, Any] = {}
    pending = None
    resume_enabled = isinstance(policy, ActivationPolicy) and not evaluations_only
    resume_key = (profile["profile_id"], patient_index, config.domain,
                  config.therapist_method, config.patient_simulator)
    if resume_enabled and resume_path.exists():
        saved = torch.load(resume_path, map_location="cpu", weights_only=False)
        if saved.get("version") != 1 or saved["session"] != resume_key:
            raise ValueError(f"Session checkpoint does not match this simulation: {resume_path}")
        policy.agent.load_session_state_dict(saved["policy"])
        if isinstance(patient, TopasSimulator):
            patient.current_state = deepcopy(saved["patient_state"])
        turns, pending = saved["turns"], saved["pending"]
        checkpoints, termination = saved["checkpoints"], saved["termination"]
        random.setstate(saved["rng"]["python"])
        np.random.set_state(saved["rng"]["numpy"])
        torch.set_rng_state(saved["rng"]["torch"])
        if torch.cuda.is_available():
            torch.cuda.set_rng_state_all(saved["rng"]["cuda"])
        print(f"[resume] {profile['profile_id']}: {len(turns)} completed turns"
              f"{' plus a saved partial turn' if pending is not None else ''}", flush=True)
    elif resume_enabled and any(case_dir.glob("turn_*.json")):
        raise ValueError(
            f"Partial session in {case_dir} has no session_resume.pt. Older turn logs "
            "cannot restore the full TOPAS/KG state; keep them and use a new output directory."
        )

    def checkpoint_session() -> None:
        if not resume_enabled:
            return
        saved = {
            "version": 1, "session": resume_key, "turns": turns, "pending": pending,
            "checkpoints": checkpoints, "termination": termination,
            "policy": policy.agent.session_state_dict(),
            "patient_state": deepcopy(patient.current_state) if isinstance(patient, TopasSimulator) else None,
            "rng": {"python": random.getstate(), "numpy": np.random.get_state(),
                    "torch": torch.get_rng_state(),
                    "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []},
        }
        temporary = resume_path.with_name(resume_path.name + ".tmp")
        torch.save(saved, temporary)
        temporary.replace(resume_path)

    def ensure_evaluation(turn_index: int) -> None:
        if evaluations_only and turn_index not in checkpoint_turns:
            return
        if str(turn_index) in checkpoints:
            return
        emit_progress(config.seed, patient_index, turn_index, config.max_turns,
                      profile["profile_id"], phase="evaluating")
        path = case_dir / f"evaluation_turn_{turn_index}.json"
        if resume_enabled and path.exists():
            checkpoint = json.loads(path.read_text(encoding="utf-8"))
        else:
            checkpoint = evaluate(judges, transcript_text(turns[:turn_index], domain),
                                  session_metadata, domain=config.domain)
            checkpoint["turn"] = turn_index
            save_json(path, checkpoint)
        checkpoints[str(turn_index)] = checkpoint
        checkpoint_session()

    emit_progress(config.seed, patient_index, len(turns), config.max_turns, profile["profile_id"])

    # A judge failure must not cause completed dialogue turns to run again.
    for completed_turn in sorted(checkpoint_turns | ({len(turns)} if termination else set())):
        if 0 < completed_turn <= len(turns):
            ensure_evaluation(completed_turn)

    for turn_index in tqdm(
        range(len(turns) + 1, config.max_turns + 1) if not termination else (),
        total=config.max_turns,
        initial=len(turns),
        desc=profile["profile_id"],
        leave=False,
        disable=progress_enabled(),
    ):
        before_turn = transcript_text(turns, domain)
        if pending is None:
            if isinstance(policy, ActivationPolicy):
                therapist_text, therapist_call = policy.respond_dialogue(dialogue_turns(turns, domain), session_metadata)
            else:
                therapist_text, therapist_call = policy.respond(before_turn)
            pending = {"therapist": _clean(therapist_text), "therapist_call": therapist_call}
            checkpoint_session()
        therapist_text, therapist_call = pending["therapist"], pending["therapist_call"]
        if config.save_prompt_diagnostics and not evaluations_only:
            saved_therapist_call = dict(therapist_call)
            saved_therapist_call.pop("tensor_payload", None)
            save_json(case_dir / f"agent_prompt_turn_{turn_index}.json", saved_therapist_call)
        forced_farewell = bool(therapist_call.get("terminate_session"))
        matched = (
            farewell_phrase(therapist_text) or "goodbye"
            if forced_farewell
            else farewell_phrase(therapist_text) if config.stop_on_farewell else None
        )

        patient_text = ""
        patient_call: Dict[str, Any] = {}
        if matched:
            termination = {
                "reason": "therapist_farewell",
                "speaker": "therapist",
                "phrase": matched,
                "utterance": therapist_text,
                "policy_reason": therapist_call.get("termination_reason", ""),
            }
        else:
            if "patient" not in pending:
                patient_input = before_turn + f"{domain.system_role}: {therapist_text}\n"
                patient_text = _clean(patient.respond(patient_input))
                pending.update(patient=patient_text, patient_call=patient.last_call)
                checkpoint_session()
            patient_text, patient_call = pending["patient"], pending["patient_call"]
            if isinstance(patient, TopasSimulator) and not evaluations_only:
                save_json(
                    case_dir / f"topas_state_turn_{turn_index}.json",
                    patient.current_state,
                )
            if config.save_prompt_diagnostics and not evaluations_only:
                save_json(case_dir / f"patient_prompt_turn_{turn_index}.json", patient_call)
            matched = farewell_phrase(patient_text) if config.stop_on_farewell else None
            if matched:
                termination = {
                    "reason": "patient_farewell",
                    "speaker": "patient",
                    "phrase": matched,
                    "utterance": patient_text,
                }

        turn = {
            "turn_index": turn_index,
            "therapist": therapist_text,
            "patient": patient_text,
        }
        if not evaluations_only:
            turn.update(therapist_call=therapist_call, patient_call=patient_call,
                        repetition_diagnostics={
                "therapist": repetition(turns, therapist_text, "therapist"),
                "patient": repetition(turns, patient_text, "patient"),
            })
        if config.debug and not evaluations_only:
            save_debug_bundle(case_dir, turn_index, turn)
        therapist_call.pop("tensor_payload", None)
        turns.append(turn)
        if not evaluations_only:
            save_json(case_dir / f"turn_{turn_index}.json", turn)
        pending = None
        checkpoint_session()

        if turn_index in checkpoint_turns or termination:
            ensure_evaluation(turn_index)
        emit_progress(config.seed, patient_index, turn_index, config.max_turns, profile["profile_id"])
        if termination:
            break

    final_turn = len(turns)
    kg_path = None
    if isinstance(policy, ActivationPolicy):
        final_graph = policy.finish_session(dialogue_turns(turns, domain))
        checkpoint_session()
        if final_graph is not None and not evaluations_only:
            kg_path = case_dir / "knowledge_graph_final.json"
            save_json(kg_path, final_graph)
    ensure_evaluation(final_turn)
    final_evaluation = checkpoints.get(str(final_turn), {"judge_results": {}, "metrics": {}})
    case = {
        "profile_id": profile["profile_id"],
        "profile_name": profile["name"],
        "domain": config.domain,
        "system_role": domain.system_role,
        "user_role": domain.user_role,
        "patient_simulator": config.patient_simulator,
        "therapist_method": config.therapist_method,
        "profile": profile,
        "transcript": transcript_text(turns, domain),
        "turns": turns,
        "num_turns": final_turn,
        "num_utterances": bool(domain.initial_user_utterance)
        + sum(bool(turn["therapist"]) + bool(turn["patient"]) for turn in turns),
        "terminated_early": bool(termination) and final_turn < config.max_turns,
        "termination_reason": termination.get("reason", "max_turns"),
        "termination": termination,
        "judge_results": final_evaluation["judge_results"],
        "metrics": final_evaluation["metrics"],
        "checkpoints": checkpoints,
        "save_mode": config.save_mode,
        "evaluation_turn": final_turn if str(final_turn) in checkpoints else None,
    }
    if kg_path is not None:
        case["knowledge_graph_path"] = str(kg_path)
        case["knowledge_graph_usage"] = dict(policy.agent.memory.usage)
    # Persist the completed dialogue once, including in evaluation-only mode.
    (case_dir / "transcript.txt").write_text(case["transcript"], encoding="utf-8")
    if evaluations_only:
        for field in ("profile", "transcript", "turns", "termination"):
            case.pop(field, None)
    save_json(case_dir / "case.json", case)
    if resume_enabled:
        resume_path.unlink(missing_ok=True)
    emit_progress(config.seed, patient_index, final_turn, config.max_turns,
                  profile["profile_id"], phase="complete")
    return case


def add_judges_to_case(
    case: Dict[str, Any],
    case_dir: Path,
    judges: List[Judge],
    rerun_judges: bool = False,
) -> Dict[str, Any]:
    if case.get("save_mode") == "evaluations":
        labels = {label for label, _ in judges}
        if rerun_judges or any(not labels.issubset(checkpoint.get("judge_results", {}))
                               for checkpoint in case["checkpoints"].values()):
            raise ValueError("Cannot rejudge compact evaluation-only cases: structured turns and profile metadata were not saved. Use a new output directory to rerun them.")
        return case
    domain_name = case.get("domain", "cbt")
    domain = get_domain(domain_name)
    profile = case["profile"]
    session_metadata = (
        {
            "presenting conditions": profile["presenting_conditions"],
            "summary": profile["summary"],
        }
        if domain_name == "cbt"
        else {"conversation_id": profile["conversation_id"]}
    )
    judge_labels = {label for label, _ in judges}
    for turn, checkpoint in case["checkpoints"].items():
        turn_index = int(turn)
        existing_results = checkpoint.get("judge_results", {})
        if rerun_judges:
            existing_results = {
                label: result
                for label, result in existing_results.items()
                if label not in judge_labels
            }
        updated = evaluate(
            judges,
            transcript_text(case["turns"][:turn_index], domain),
            session_metadata,
            existing_results=existing_results,
            domain=domain_name,
        )
        updated["turn"] = turn_index
        case["checkpoints"][turn] = updated
        save_json(case_dir / f"evaluation_turn_{turn_index}.json", updated)
        save_json(case_dir / "case.json", case)

    final = case["checkpoints"][str(case["num_turns"])]
    case["judge_results"] = final["judge_results"]
    case["metrics"] = final["metrics"]
    save_json(case_dir / "case.json", case)
    return case
