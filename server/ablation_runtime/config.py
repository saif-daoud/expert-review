from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple


ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class DomainSpec:
    name: str
    system_role: str
    user_role: str
    system_prompt: str
    dialogue_heading: str
    policy_prompt_filename: str
    default_profiles: Path
    default_max_turns: int
    initial_user_utterance: str = ""
    user_role_aliases: Tuple[str, ...] = ()

    @property
    def stop_tokens(self) -> List[str]:
        return [
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
    default_profiles=ROOT / "data" / "patient_act.json",
    default_max_turns=200,
    initial_user_utterance="Hello",
    user_role_aliases=("Client",),
)

P4G_DOMAIN = DomainSpec(
    name="p4g",
    system_role="Persuader",
    user_role="Persuadee",
    system_prompt=(
        "You are a helpful persuader speaking with another person about donating to Save the Children. "
        "Write only the next persuader utterance. Be respectful, concise, and do not invent facts or speaker labels."
    ),
    dialogue_heading="Persuasion dialogue",
    policy_prompt_filename="persuader_agent_prompt.txt",
    default_profiles=ROOT / "data" / "p4g" / "test.json",
    default_max_turns=30,
)


def get_domain(name: str) -> DomainSpec:
    return {"cbt": CBT_DOMAIN, "p4g": P4G_DOMAIN}[name]


@dataclass
class RunConfig:
    domain: str = "cbt"
    profiles: Path = ROOT / "data" / "patient_act.json"
    output_dir: Path = ROOT / "outputs" / "run"
    max_turns: int = 200
    seed: int = 7
    patient_simulator: str = "patient_act"
    therapist_method: str = "archer"
    stop_on_farewell: bool = True
    save_prompt_diagnostics: bool = True
    save_mode: str = "full" # full | evaluations (scores, compact summaries, and final transcript)
    debug: bool = False
    judge_models: List[str] = field(default_factory=lambda: ["gpt-4.1", "DeepSeek-V4-Pro"])
    components_dir: Path = ROOT / "topa_components"
    patient_psi_cache_dir: Path = ROOT / "cache" / "patient_psi"
    topas_profile_cache_dir: Path = ROOT / "cache" / "topas_profiles"
    topas_profile_schema_path: Path = ROOT / "topa_components" / "user_profile.json"
    topas_use_dynamic_dimensions: bool = True

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        for key in (
            "profiles",
            "output_dir",
            "components_dir",
            "patient_psi_cache_dir",
            "topas_profile_cache_dir",
            "topas_profile_schema_path",
        ):
            data[key] = str(data[key])
        return data


DOMAIN = CBT_DOMAIN
