import abc
import json
import os
import re
import sys
import time
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional



MAX_NEW_TOKENS = 64  # default
FALLBACK_MAX_NEW_TOKENS = int(os.environ.get("TOPA_FALLBACK_MAX_NEW_TOKENS", "256"))

logger = logging.getLogger(__name__)


REPO_ROOT = Path(__file__).resolve().parents[1]


def _env_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _resolve_torch_dtype(torch_module: Any, torch_dtype: Optional[str]) -> Any:
    if not torch_dtype:
        return None
    dtype_name = str(torch_dtype).strip().lower()
    if dtype_name == "auto":
        return "auto"
    mapping = {
        "bf16": torch_module.bfloat16,
        "bfloat16": torch_module.bfloat16,
        "fp16": torch_module.float16,
        "float16": torch_module.float16,
        "half": torch_module.float16,
        "fp32": torch_module.float32,
        "float32": torch_module.float32,
    }
    if dtype_name not in mapping:
        raise ValueError(f"Unsupported torch_dtype={torch_dtype}. Use one of: auto, bf16, bfloat16, fp16, float16, fp32, float32")
    return mapping[dtype_name]


def _apply_stop(text: str, stop: Optional[List[str]]) -> str:
    text = (text or "").strip()
    if stop:
        for s in stop:
            if s and s in text:
                text = text.split(s)[0].strip()
    return text


def finish_utterance(text: str) -> str:
    """Drop a trailing partial sentence left by a generation token limit."""
    text = (text or "").strip()
    if not text or re.search(r"[.!?][\"')\]]*$", text):
        return text

    sentence_ends = list(re.finditer(r"[.!?][\"')\]]*", text))
    if sentence_ends:
        return text[:sentence_ends[-1].end()].strip()
    return text


def _parse_transcript_turns(transcript: str) -> List[Dict[str, str]]:
    turns: List[Dict[str, str]] = []
    roles = "Therapist|Patient|Client|Persuader|Persuadee"
    pattern = re.compile(rf"({roles}):\s*(.*?)(?=(?:\n)?(?:{roles}):|\Z)", re.S)
    for match in pattern.finditer(transcript or ""):
        speaker = match.group(1)
        text = " ".join((match.group(2) or "").split()).strip()
        if text:
            turns.append({"speaker": speaker, "text": text})
    return turns


def _transcript_to_archer_observation(
    transcript: str,
    system_instruction: str = "",
    domain_hint: str = "",
    max_history_utterances: int = -1,
) -> str:
    is_p4g = "Persuader:" in transcript or "persuader" in domain_hint.lower()
    system_label = "Persuader" if is_p4g else "Therapist"
    user_labels = {"Persuadee"} if is_p4g else {"Patient", "Client"}
    user_label = "Persuadee" if is_p4g else "Client"
    lines = []
    if system_instruction:
        lines.append(system_instruction.strip())
        lines.append("")
    lines.append("Persuasion dialogue" if is_p4g else "CBT therapy dialogue")
    turns = _parse_transcript_turns(transcript)
    if max_history_utterances >= 0:
        turns = turns[-max_history_utterances:]
    for turn in turns:
        label = user_label if turn["speaker"] in user_labels else system_label
        lines.append(f"{label}: {turn['text']}")
    lines.append(f"{system_label}:")
    return "\n".join(lines)


def _transcript_to_chat_messages(transcript: str) -> List[Dict[str, str]]:
    messages: List[Dict[str, str]] = []
    for turn in _parse_transcript_turns(transcript):
        role = "assistant" if turn["speaker"] in {"Therapist", "Persuader"} else "user"
        messages.append({"role": role, "content": turn["text"]})
    return messages

def run_llm_query(
    prompt: str,
    deployment_name: str,
    api_key: str,
    max_new_tokens: int | None = None,
    party: str = "",
):
    """User-provided Azure OpenAI helper (with tiny safe additions).

    Notes
    -----
    - Uses Azure OpenAI (NOT the OpenAI public endpoint).
    - Enforces MAX_NEW_TOKENS.
    - Endpoint/deployment/version can be overridden via env:
        AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_DEPLOYMENT, AZURE_OPENAI_API_VERSION
    """

    from openai import AzureOpenAI

    from openai import OpenAI



    # api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
    # endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT", "https://qcri-sakina.cognitiveservices.azure.com/")
    # DEPLOYMENT_NAME = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4.1")

    # client = AzureOpenAI(
    #     azure_endpoint=endpoint,
    #     api_key=api_key,
    #     api_version=api_version,
    # )

    endpoint = os.environ.get(
        "AZURE_OPENAI_ENDPOINT",
        "https://qcri-sakina.services.ai.azure.com/openai/v1",
    )
    client = OpenAI(
        base_url=endpoint,
        api_key=api_key,
    )

    attempt = 0
    max_retries = int(os.environ.get(f"{party}_MAX_RETRIES", os.environ.get("LLM_MAX_RETRIES", "5")))
    while attempt < max_retries:
        try:
            response = client.chat.completions.create(
                model=deployment_name,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_completion_tokens=MAX_NEW_TOKENS if max_new_tokens is None else int(max_new_tokens),
                temperature=float(os.environ.get(f"{party}_TEMPERATURE", "0.0")),
                top_p=float(os.environ.get(f"{party}_TOP_P", "1.0")),
            )
            break
        except Exception as e:
            attempt += 1
            print(
                f"[llm retry] party={party.lower() or 'unknown'} model={deployment_name} "
                f"attempt={attempt} error={type(e).__name__}: {e}",
                file=sys.stderr,
                flush=True,
            )
            if attempt >= max_retries:
                raise RuntimeError(
                    f"LLM request failed after {max_retries} attempts: party={party or 'unknown'} "
                    f"model={deployment_name} last_error={type(e).__name__}: {e}"
                ) from e
            time.sleep(5)

    output_text = response.choices[0].message.content
    input_tokens = response.usage.prompt_tokens
    output_tokens = response.usage.completion_tokens
    return output_text, input_tokens, output_tokens


def run_llm_query_no_max_new_tokens(
    prompt: str,
    deployment_name: str,
    api_key: str,
    party: str = "",
):
    """User-provided Azure OpenAI helper (with tiny safe additions).

    Notes
    -----
    - Uses Azure OpenAI (NOT the OpenAI public endpoint).
    - Enforces MAX_NEW_TOKENS.
    - Endpoint/deployment/version can be overridden via env:
        AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_DEPLOYMENT, AZURE_OPENAI_API_VERSION
    """
    from openai import AzureOpenAI
    from openai import OpenAI
    # from openai import AzureOpenAI

    # api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
    # endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT", "https://qcri-sakina.cognitiveservices.azure.com/")
    # DEPLOYMENT_NAME = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4.1")

    # client = AzureOpenAI(
    #     azure_endpoint=endpoint,
    #     api_key=api_key,
    #     api_version=api_version,
    # )

    endpoint = os.environ.get(
        "AZURE_OPENAI_ENDPOINT",
        "https://qcri-sakina.services.ai.azure.com/openai/v1",
    )

    client = OpenAI(
        base_url=endpoint,
        api_key=api_key,
    )

    attempt = 0
    max_retries = int(os.environ.get(f"{party}_MAX_RETRIES", os.environ.get("LLM_MAX_RETRIES", "5")))
    while attempt < max_retries:
        try:
            response = client.chat.completions.create(
                model=deployment_name,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=float(os.environ.get(f"{party}_TEMPERATURE", "0.7")),
                top_p=float(os.environ.get(f"{party}_TOP_P", "0.9")),
            )
            break
        except Exception as e:
            attempt += 1
            print(
                f"[llm retry] party={party.lower() or 'unknown'} model={deployment_name} "
                f"attempt={attempt} error={type(e).__name__}: {e}",
                file=sys.stderr,
                flush=True,
            )
            if attempt >= max_retries:
                raise RuntimeError(
                    f"LLM request failed after {max_retries} attempts: party={party or 'unknown'} "
                    f"model={deployment_name} last_error={type(e).__name__}: {e}"
                ) from e
            time.sleep(5)

    output_text = response.choices[0].message.content
    input_tokens = response.usage.prompt_tokens
    output_tokens = response.usage.completion_tokens
    return output_text, input_tokens, output_tokens


@dataclass
class LLMCallReport:
    """Per-call generation report."""

    backend: str
    model: str
    latency_ms: float
    prompt_tokens: Optional[int]
    completion_tokens: Optional[int]
    total_tokens: Optional[int]
    prompt_chars: int
    completion_chars: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "backend": self.backend,
            "model": self.model,
            "latency_ms": float(self.latency_ms),
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "prompt_chars": int(self.prompt_chars),
            "completion_chars": int(self.completion_chars),
        }


def _sum_opt_int(vals: List[Optional[int]]) -> Optional[int]:
    if any(v is None for v in vals):
        return None
    return int(sum(int(v) for v in vals if v is not None))


class TextGenerator(abc.ABC):
    """Backend-agnostic interface for text generation."""

    last_report: Optional[LLMCallReport] = None

    def __init__(self, max_new_tokens: int | None = None) -> None:
        # Aggregated per "turn" (a higher-level unit like one environment step).
        self._turn_reports: List[LLMCallReport] = []
        self.max_new_tokens: int | None = max_new_tokens if max_new_tokens is not None else MAX_NEW_TOKENS
        self.save_prompt_diagnostics = _env_bool(
            os.environ.get("SAVE_PROMPT_DIAGNOSTICS"), True
        )
        self.last_prompt_diagnostics: Dict[str, Any] = {}

    @abc.abstractmethod
    def generate(self, *, system: str, user: str, stop: Optional[List[str]] = None) -> str:
        raise NotImplementedError

    def build_user_with_history(
        self,
        *,
        system: str,
        prefix: str,
        history: str,
        suffix: str,
    ) -> str:
        """Compose a prompt whose history may be shortened by bounded backends."""
        return f"{prefix}{history}{suffix}"

    def generate_messages_structured(self, messages: List[Dict[str, str]], response_format: Any) -> Any:
        """Generate a typed response without modifying the supplied prompts."""
        system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
        user = "\n".join(
            f"{m['role'].title()}: {m['content']}" for m in messages if m["role"] != "system"
        )
        raw = self.generate(system=system, user=user)
        match = re.search(r"```(?:json)?\s*(.*?)```", raw, re.I | re.S)
        payload = json.loads(match.group(1) if match else raw)
        return response_format.model_validate(payload)

    def _resolve_max_new_tokens(self) -> int | None:
        if self.max_new_tokens is None:
            return None
        try:
            n = int(self.max_new_tokens)
        except Exception:
            return MAX_NEW_TOKENS
        if n <= 0:
            return None
        return n

    # -------------------------
    # Reporting helpers
    # -------------------------
    def reset_turn_stats(self) -> None:
        self._turn_reports = []

    def add_report(self, report: LLMCallReport) -> None:
        self.last_report = report
        self._turn_reports.append(report)

    def record_prompt(
        self,
        *,
        system: str,
        user: str,
        rendered_before: str,
        rendered_after: str,
        tokens_before: Optional[int],
        tokens_after: Optional[int],
        max_input_tokens: Optional[int],
        truncation_side: Optional[str],
    ) -> None:
        if not self.save_prompt_diagnostics:
            self.last_prompt_diagnostics = {}
            return
        self.last_prompt_diagnostics = {
            "system_prompt": system,
            "user_prompt": user,
            "rendered_prompt_before_truncation": rendered_before,
            "rendered_prompt_after_truncation": rendered_after,
            "prompt_truncated": (
                tokens_before is not None
                and tokens_after is not None
                and tokens_after < tokens_before
            ),
            "prompt_tokens_before_truncation": tokens_before,
            "prompt_tokens_after_truncation": tokens_after,
            "prompt_characters_before_truncation": len(rendered_before),
            "prompt_characters_after_truncation": len(rendered_after),
            "max_input_tokens": max_input_tokens,
            "truncation_side": truncation_side,
        }

    def turn_stats(self) -> Dict[str, Any]:
        """Aggregate stats for calls since the last reset_turn_stats()."""
        calls = len(self._turn_reports)
        if calls == 0:
            return {
                "calls": 0,
                "latency_ms": 0.0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "prompt_chars": 0,
                "completion_chars": 0,
            }

        latency_ms = sum(r.latency_ms for r in self._turn_reports)
        prompt_chars = sum(r.prompt_chars for r in self._turn_reports)
        completion_chars = sum(r.completion_chars for r in self._turn_reports)

        prompt_tokens = _sum_opt_int([r.prompt_tokens for r in self._turn_reports])
        completion_tokens = _sum_opt_int([r.completion_tokens for r in self._turn_reports])
        total_tokens = _sum_opt_int([r.total_tokens for r in self._turn_reports])

        # Backward-friendly: if tokens are unknown (None), expose as None.
        return {
            "calls": int(calls),
            "latency_ms": float(latency_ms),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "prompt_chars": int(prompt_chars),
            "completion_chars": int(completion_chars),
            # Per-call breakdown can be useful for debugging.
            "calls_breakdown": [r.to_dict() for r in self._turn_reports],
        }


@dataclass
class CallableGenerator(TextGenerator):
    """Wrap a user-provided callable.

    The callable receives a dict payload {system, user, max_new_tokens}.

    Token counts are not exact here; we report *approximate* token counts
    using whitespace word counts.
    """

    fn: Callable[[Dict[str, Any]], str]
    backend_name: str = "callable"
    model_name: str = "callable"

    def __post_init__(self) -> None:
        super().__init__()

    def generate(self, *, system: str, user: str, stop: Optional[List[str]] = None) -> str:
        t0 = time.perf_counter()
        max_new = self._resolve_max_new_tokens()
        out = self.fn({
            "system": system,
            "user": user,
            "stop": stop,
            "max_new_tokens": max_new,
        })
        text = (out or "").strip()
        if stop:
            for s in stop:
                if s and s in text:
                    text = text.split(s)[0].strip()

        t1 = time.perf_counter()
        prompt_chars = len(system) + len(user)
        completion_chars = len(text)

        # Approximate tokens as word counts.
        prompt_tokens = len((system + " " + user).split())
        completion_tokens = len(text.split())
        total_tokens = prompt_tokens + completion_tokens
        rendered_prompt = f"[SYSTEM]\n{system}\n\n[USER]\n{user}"
        self.record_prompt(
            system=system,
            user=user,
            rendered_before=rendered_prompt,
            rendered_after=rendered_prompt,
            tokens_before=prompt_tokens,
            tokens_after=prompt_tokens,
            max_input_tokens=None,
            truncation_side=None,
        )

        self.add_report(
            LLMCallReport(
                backend=self.backend_name,
                model=self.model_name,
                latency_ms=(t1 - t0) * 1000.0,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                prompt_chars=prompt_chars,
                completion_chars=completion_chars,
            )
        )

        return text


class TransformersChatGenerator(TextGenerator):
    """Generate with a local HF causal LM.

    Notes
    -----
    - If the model doesn't support chat templates, we fall back to a simple
      "System:
...
User:
...
Assistant:" prompt.
    - Supports larger instruction models such as Qwen2.5 Instruct through
      optional device_map / torch_dtype / extra HF kwargs from env.
    """

    def __init__(
        self,
        model_name_or_path: str,
        device: Optional[str] = None,
        torch_dtype: Optional[str] = None,
        trust_remote_code: bool = False,
        device_map: Optional[str] = None,
        max_input_length: Optional[int] = 512,
        prompt_style: str = "chat",
        temperature: float = 0.7,
        top_p: float = 0.95,
        **model_kwargs: Any,
    ) -> None:
        super().__init__()

        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        self.model_name_or_path = model_name_or_path

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.device_map = device_map
        self.max_input_length = max_input_length
        self.prompt_style = prompt_style
        self.temperature = temperature
        self.top_p = top_p

        dtype = _resolve_torch_dtype(torch, torch_dtype)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=trust_remote_code)
        self.tokenizer.truncation_side = "left"

        load_kwargs = dict(model_kwargs)
        load_kwargs["trust_remote_code"] = trust_remote_code
        if dtype is not None:
            load_kwargs["torch_dtype"] = dtype
        if device_map is not None:
            load_kwargs["device_map"] = device_map

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            **load_kwargs,
        )
        self.model.config.use_cache = True
        if device_map is None:
            self.model.to(self.device)
        self.model.eval()

        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def _build_prompt(self, system: str, user: str) -> str:
        if self.prompt_style == "raw":
            return user
        if hasattr(self.tokenizer, "apply_chat_template"):
            try:
                messages = [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ]
                return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            except Exception:
                pass

        return f"System:\n{system}\n\nUser:\n{user}\n\nAssistant:\n"

    def build_user_with_history(
        self,
        *,
        system: str,
        prefix: str,
        history: str,
        suffix: str,
    ) -> str:
        """Fit only conversation history into the model's input budget.

        The system instruction, chat-template tokens, and fixed user prompt
        text are counted first and are never removed. The oldest history
        tokens are dropped until the complete rendered prompt fits.
        """
        full_user = f"{prefix}{history}{suffix}"
        if self.max_input_length is None:
            return full_user

        limit = int(self.max_input_length)

        def prompt_tokens(candidate_history: str) -> int:
            rendered = self._build_prompt(
                system,
                f"{prefix}{candidate_history}{suffix}",
            )
            return int(len(self.tokenizer(rendered)["input_ids"]))

        if prompt_tokens(history) <= limit:
            return full_user
        if prompt_tokens("") > limit:
            raise ValueError(
                "The fixed system instruction and task text exceed the model's "
                f"{limit}-token input limit; refusing to truncate them."
            )

        history_ids = list(
            self.tokenizer(history, add_special_tokens=False)["input_ids"]
        )

        def decode_suffix(keep: int) -> str:
            if keep <= 0:
                return ""
            return self.tokenizer.decode(
                history_ids[-keep:],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )

        low, high = 0, len(history_ids)
        while low < high:
            keep = (low + high + 1) // 2
            if prompt_tokens(decode_suffix(keep)) <= limit:
                low = keep
            else:
                high = keep - 1

        fitted = decode_suffix(low)
        if low < len(history_ids) and "\n" in fitted:
            # Avoid beginning with a partial old turn when a complete newer
            # turn is available. Keep a complete leading speaker line intact.
            if re.match(
                r"^\s*(?:patient|therapist|client|persuader|persuadee|user|assistant)\s*:",
                fitted,
                re.IGNORECASE,
            ) is None:
                fitted = fitted.split("\n", 1)[1]
        return f"{prefix}{fitted}{suffix}"

    def generate(self, *, system: str, user: str, stop: Optional[List[str]] = None) -> str:
        import torch

        prompt = self._build_prompt(system, user)
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=self.max_input_length is not None,
            max_length=self.max_input_length,
        )
        if self.save_prompt_diagnostics:
            tokens_before = len(self.tokenizer(prompt)["input_ids"])
            input_ids = inputs["input_ids"][0].tolist()
            rendered_after = self.tokenizer.decode(
                input_ids,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            self.record_prompt(
                system=self._effective_system_prompt(system),
                user=user,
                rendered_before=prompt,
                rendered_after=rendered_after,
                tokens_before=tokens_before,
                tokens_after=len(input_ids),
                max_input_tokens=self.max_input_length,
                truncation_side=self.tokenizer.truncation_side if self.max_input_length is not None else None,
            )

        # Always place the prompt tensors on the model input device.
        # This avoids the common HF/Accelerate failure where `device_map="auto"`
        # leaves the model sharded on CUDA but the input_ids stay on CPU.
        target_device = self.device
        if self.device_map is not None:
            try:
                target_device = next(self.model.parameters()).device
            except Exception:
                target_device = getattr(self.model, "device", self.device)
        inputs = {k: v.to(target_device) if hasattr(v, "to") else v for k, v in inputs.items()}

        prompt_len = int(inputs["input_ids"].shape[-1])

        t0 = time.perf_counter()
        max_new = self._resolve_max_new_tokens()
        if max_new is None:
            max_new = FALLBACK_MAX_NEW_TOKENS
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=int(max_new),
                do_sample=True,
                temperature=self.temperature,
                top_p=self.top_p,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        t1 = time.perf_counter()

        gen_ids = out[0][prompt_len:]
        completion_tokens = int(gen_ids.shape[-1])
        gen = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
        gen = gen.strip()

        if stop:
            for s in stop:
                if s and s in gen:
                    gen = gen.split(s)[0].strip()

            try:
                completion_tokens = int(len(self.tokenizer.encode(gen)))
            except Exception:
                pass

        prompt_tokens = prompt_len
        total_tokens = prompt_tokens + completion_tokens

        self.add_report(
            LLMCallReport(
                backend="hf",
                model=self.model_name_or_path,
                latency_ms=(t1 - t0) * 1000.0,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                prompt_chars=len(system) + len(user),
                completion_chars=len(gen),
            )
        )

        return gen

    def _effective_system_prompt(self, system: str) -> str:
        return system


class SweetRLPolicyGenerator(TransformersChatGenerator):
    """HF generator for SWEET-RL actor checkpoints using their training prompt."""

    def __init__(
        self,
        model_name_or_path: str,
        prompt_path: Optional[str] = None,
        assistant_label: str = "Therapist",
        user_label: str = "Client",
        **kwargs: Any,
    ) -> None:
        self.assistant_label = assistant_label
        self.user_label = user_label
        default_prompt = REPO_ROOT / "assets" / "therapist_agent_prompt.txt"
        prompt_file = Path(prompt_path) if prompt_path else default_prompt
        self.policy_system_prompt = prompt_file.read_text(encoding="utf-8").strip()
        super().__init__(model_name_or_path=model_name_or_path, **kwargs)

    def _build_prompt(self, system: str, user: str) -> str:
        messages = [{"role": "system", "content": self.policy_system_prompt}] + _transcript_to_chat_messages(user)
        if hasattr(self.tokenizer, "apply_chat_template"):
            try:
                return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            except Exception:
                pass

        lines = [f"System: {self.policy_system_prompt}"]
        for message in messages[1:]:
            label = self.assistant_label if message["role"] == "assistant" else self.user_label
            lines.append(f"{label}: {message['content']}")
        lines.append(f"{self.assistant_label}:")
        return "\n".join(lines)

    def _effective_system_prompt(self, system: str) -> str:
        return self.policy_system_prompt


class ArcherPolicyGenerator(TextGenerator):
    """OfflineArcher actor checkpoint generator for CBT simulations."""

    def __init__(
        self,
        model_name_or_path: str,
        checkpoint_path: Optional[str] = None,
        prompt_style: str = "raw",
        torch_dtype: str = "bf16",
        device: Optional[str] = None,
        max_input_length: int = 512,
        actor_use_lora: bool = False,
        actor_lora_r: int = 16,
        actor_lora_alpha: int = 32,
        actor_lora_dropout: float = 0.05,
        actor_gradient_checkpointing: bool = False,
        system_prompt_path: Optional[str] = None,
        use_sweet_rl_prompt: bool = False,
        max_new_tokens: int | None = None,
    ) -> None:
        super().__init__(max_new_tokens=max_new_tokens)

        import torch

        from .archer_model import GPT2

        self.checkpoint_path = str(Path(checkpoint_path).expanduser()) if checkpoint_path else None
        if self.checkpoint_path and not Path(self.checkpoint_path).exists():
            raise FileNotFoundError(f"Archer checkpoint does not exist: {self.checkpoint_path}")

        checkpoint = None
        checkpoint_state = None
        if self.checkpoint_path:
            try:
                checkpoint = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
            except TypeError:
                checkpoint = torch.load(self.checkpoint_path, map_location="cpu")

            checkpoint_state = checkpoint.get("state_dict", checkpoint)
            hparams = checkpoint.get("hyper_parameters", {})
            if not isinstance(hparams, dict):
                hparams = {}

            checkpoint_has_lora = any(".lora_A." in key for key in checkpoint_state)
            if checkpoint_has_lora != actor_use_lora:
                print(
                    f"[archer] Checkpoint LoRA={checkpoint_has_lora}; "
                    f"overriding configured LoRA={actor_use_lora}.",
                    flush=True,
                )
            actor_use_lora = checkpoint_has_lora
            if actor_use_lora:
                actor_lora_r = int(hparams.get("actor_lora_r", actor_lora_r))
                actor_lora_alpha = int(hparams.get("actor_lora_alpha", actor_lora_alpha))
                actor_lora_dropout = float(hparams.get("actor_lora_dropout", actor_lora_dropout))

        self.model_name_or_path = model_name_or_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_label = self.model_name_or_path
        if self.checkpoint_path:
            self.model_label = f"{self.model_name_or_path}@{Path(self.checkpoint_path).name}"
        self.system_instruction = ""
        if system_prompt_path or use_sweet_rl_prompt:
            prompt_file = Path(system_prompt_path) if system_prompt_path else REPO_ROOT / "assets" / "therapist_agent_prompt.txt"
            self.system_instruction = prompt_file.read_text(encoding="utf-8").strip()
        self.actor = GPT2(
            get_device=lambda: torch.device(self.device),
            from_checkpoint=None,
            model_name=model_name_or_path,
            prompt_style=prompt_style,
            max_input_length=int(max_input_length),
            max_new_tokens=int(max_new_tokens or MAX_NEW_TOKENS),
            torch_dtype=torch_dtype,
            use_lora=actor_use_lora,
            lora_r=int(actor_lora_r),
            lora_alpha=int(actor_lora_alpha),
            lora_dropout=float(actor_lora_dropout),
            gradient_checkpointing=actor_gradient_checkpointing,
        )

        if checkpoint_state is not None:
            weights = {}
            for key, value in checkpoint_state.items():
                for prefix in ("actor.", "agent."):
                    if key.startswith(prefix):
                        weights[key.removeprefix(prefix)] = value
                        break
            if not weights:
                raise RuntimeError("Archer checkpoint contains no actor weights.")

            missing, unexpected = self.actor.load_state_dict(weights, strict=False)
            if missing or unexpected:
                raise RuntimeError(
                    "Archer checkpoint did not load exactly: "
                    f"missing={len(missing)} {missing[:5]}, "
                    f"unexpected={len(unexpected)} {unexpected[:5]}"
                )
            print(
                f"[archer] Loaded {len(weights)} actor tensors from {self.checkpoint_path} "
                f"(model={self.actor.model_name}, lora={self.actor.use_lora}).",
                flush=True,
            )
            del checkpoint, checkpoint_state, weights

        self.model_name_or_path = self.actor.model_name
        self.model_label = self.model_name_or_path
        if self.checkpoint_path:
            self.model_label = f"{self.model_name_or_path}@{Path(self.checkpoint_path).name}"
        self.actor.to(self.device)
        self.actor.eval()

    def generate(self, *, system: str, user: str, stop: Optional[List[str]] = None) -> str:
        import torch

        observation = _transcript_to_archer_observation(
            user,
            system_instruction=self.system_instruction,
            domain_hint=system,
        )
        if self.save_prompt_diagnostics:
            prompt_text = self.actor._build_prompt_texts([observation])[0]
            add_special_tokens = not self.actor.use_chat_template
            tokens_before = self.actor.tokenizer(
                prompt_text,
                add_special_tokens=add_special_tokens,
            )["input_ids"]
            max_input_tokens = min(self.actor.max_input_length, self.actor.max_sequence_tokens)
            tokens_after = self.actor.tokenizer(
                prompt_text,
                truncation=True,
                max_length=max_input_tokens,
                add_special_tokens=add_special_tokens,
            )["input_ids"]
            rendered_after = self.actor.tokenizer.decode(
                tokens_after,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            self.record_prompt(
                system=self.system_instruction,
                user=user,
                rendered_before=prompt_text,
                rendered_after=rendered_after,
                tokens_before=len(tokens_before),
                tokens_after=len(tokens_after),
                max_input_tokens=max_input_tokens,
                truncation_side=self.actor.tokenizer.truncation_side,
            )
        max_new = self._resolve_max_new_tokens()
        if max_new is not None:
            self.actor.max_new_tokens = int(max_new)

        t0 = time.perf_counter()
        with torch.no_grad():
            outputs = self.actor.forward([observation], do_sample=True)
        t1 = time.perf_counter()

        text = _apply_stop(outputs[0] if outputs else "", stop)
        try:
            prompt_tokens = int(len(self.actor.tokenizer.encode(observation)))
            completion_tokens = int(len(self.actor.tokenizer.encode(text)))
        except Exception:
            prompt_tokens = None
            completion_tokens = None
        total_tokens = None if prompt_tokens is None or completion_tokens is None else prompt_tokens + completion_tokens

        self.add_report(
            LLMCallReport(
                backend="archer",
                model=self.model_label,
                latency_ms=(t1 - t0) * 1000.0,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                prompt_chars=len(observation),
                completion_chars=len(text),
            )
        )
        return text


class AriaPolicyGenerator(TextGenerator):
    """ARIA offline REINFORCE checkpoint generator for CBT simulations."""

    def __init__(
        self,
        model_name_or_path: str,
        checkpoint_path: str,
        torch_dtype: str = "bf16",
        device: Optional[str] = None,
        max_input_length: int = 4096,
        use_lora: bool = True,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        system_prompt_path: Optional[str] = None,
        max_new_tokens: int | None = None,
        trust_remote_code: bool = True,
    ) -> None:
        super().__init__(max_new_tokens=max_new_tokens)

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_name_or_path = model_name_or_path
        checkpoint = Path(checkpoint_path).expanduser()
        if checkpoint.is_dir():
            checkpoint = checkpoint / "trainer.pt"
        if not checkpoint.exists():
            raise FileNotFoundError(f"ARIA checkpoint does not exist: {checkpoint}")
        self.checkpoint_path = str(checkpoint)

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_input_length = int(max_input_length)
        dtype = _resolve_torch_dtype(torch, torch_dtype)

        load_kwargs: Dict[str, Any] = {"trust_remote_code": trust_remote_code}
        if dtype is not None:
            load_kwargs["torch_dtype"] = dtype

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=trust_remote_code,
        )
        self.tokenizer.truncation_side = "left"
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **load_kwargs)
        if use_lora:
            from peft import LoraConfig, TaskType, get_peft_model

            lora_config = LoraConfig(
                r=int(lora_r),
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                task_type=TaskType.CAUSAL_LM,
                lora_alpha=int(lora_alpha),
                lora_dropout=float(lora_dropout),
            )
            self.model = get_peft_model(self.model, lora_config)

        try:
            state = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
        except TypeError:
            state = torch.load(self.checkpoint_path, map_location="cpu")
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                "ARIA checkpoint did not load exactly: "
                f"missing={len(missing)} {missing[:5]}, "
                f"unexpected={len(unexpected)} {unexpected[:5]}"
            )
        print(
            f"[aria] Loaded {len(state)} tensors from {self.checkpoint_path} "
            f"(model={self.model_name_or_path}, lora={use_lora}).",
            flush=True,
        )
        del state

        self.model.to(self.device)
        self.model.eval()

        prompt_file = Path(system_prompt_path) if system_prompt_path else REPO_ROOT / "assets" / "therapist_agent_prompt.txt"
        self.system_instruction = prompt_file.read_text(encoding="utf-8").strip()
        self.model_label = f"{self.model_name_or_path}@{Path(self.checkpoint_path).name}"

    def generate(self, *, system: str, user: str, stop: Optional[List[str]] = None) -> str:
        import torch

        observation = _transcript_to_archer_observation(
            user,
            system_instruction=self.system_instruction,
            domain_hint=system,
        )
        inputs = self.tokenizer(
            observation,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_input_length,
        )
        if self.save_prompt_diagnostics:
            tokens_before = self.tokenizer(observation)["input_ids"]
            input_ids = inputs["input_ids"][0].tolist()
            rendered_after = self.tokenizer.decode(
                input_ids,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            self.record_prompt(
                system=self.system_instruction,
                user=user,
                rendered_before=observation,
                rendered_after=rendered_after,
                tokens_before=len(tokens_before),
                tokens_after=len(input_ids),
                max_input_tokens=self.max_input_length,
                truncation_side=self.tokenizer.truncation_side,
            )
        inputs = {k: v.to(self.device) if hasattr(v, "to") else v for k, v in inputs.items()}
        prompt_len = int(inputs["input_ids"].shape[-1])

        t0 = time.perf_counter()
        max_new = self._resolve_max_new_tokens()
        if max_new is None:
            max_new = FALLBACK_MAX_NEW_TOKENS
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=int(max_new),
                do_sample=True,
                temperature=0.7,
                top_p=0.95,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        t1 = time.perf_counter()

        gen_ids = out[0][prompt_len:]
        text = self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        text = _apply_stop(text, stop)

        try:
            prompt_tokens = int(prompt_len)
            completion_tokens = int(len(self.tokenizer.encode(text)))
        except Exception:
            prompt_tokens = None
            completion_tokens = None
        total_tokens = None if prompt_tokens is None or completion_tokens is None else prompt_tokens + completion_tokens

        self.add_report(
            LLMCallReport(
                backend="aria",
                model=self.model_label,
                latency_ms=(t1 - t0) * 1000.0,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                prompt_chars=len(observation),
                completion_chars=len(text),
            )
        )
        return text


class AzureOpenAIChatGenerator(TextGenerator):
    """Azure OpenAI chat backend.

    This intentionally does NOT use the OpenAI public endpoint.

    Defaults match the user's provided helper:
    - api_version: 2024-12-01-preview
    - endpoint: https://qcri-sakina.cognitiveservices.azure.com/
    - deployment: gpt-4.1

    Environment
    -----------
    - {prefix}_AZURE_API_KEY or AZURE_OPENAI_API_KEY
    - {prefix}_AZURE_ENDPOINT (optional)
    - {prefix}_AZURE_API_VERSION (optional)
    - {prefix}_AZURE_DEPLOYMENT (optional)
    """

    def __init__(
        self,
        deployment: str = "gpt-4.1",
        api_key: Optional[str] = None,
        endpoint: str = "https://qcri-sakina.cognitiveservices.azure.com/",
        api_version: str = "2024-12-01-preview",
        prefix: str = "",
    ) -> None:
        super().__init__()

        self.deployment = deployment
        self.api_key = api_key
        self.endpoint = endpoint
        self.api_version = api_version
        self.prefix = prefix

        if not self.api_key:
            raise RuntimeError("Azure API key is not set")

        # Keep env overrides consistent with the helper.
        os.environ.setdefault("AZURE_OPENAI_ENDPOINT", self.endpoint)
        os.environ.setdefault("AZURE_OPENAI_API_VERSION", self.api_version)
        os.environ.setdefault("AZURE_OPENAI_DEPLOYMENT", self.deployment)


    def generate(self, *, system: str, user: str, stop: Optional[List[str]] = None) -> str:
        # The user-supplied helper accepts a single prompt string.
        # We fuse system+user into one prompt while keeping roles explicit.
        prompt = f"[SYSTEM]\n{system.strip()}\n\n[USER]\n{user.strip()}\n\n[ASSISTANT]\n"

        t0 = time.perf_counter()
        max_new = self._resolve_max_new_tokens()
        if max_new is None:
            text, input_tokens, output_tokens = run_llm_query_no_max_new_tokens(
                prompt, self.deployment, self.api_key, party=self.prefix
            )
        else:
            text, input_tokens, output_tokens = run_llm_query(
                prompt,
                self.deployment,
                self.api_key,
                max_new_tokens=int(max_new),
                party=self.prefix,
            )
        t1 = time.perf_counter()

        text = (text or "").strip()

        if stop:
            for s in stop:
                if s and s in text:
                    text = text.split(s)[0].strip()

        prompt_tokens: Optional[int] = None
        completion_tokens: Optional[int] = None
        total_tokens: Optional[int] = None
        try:
            prompt_tokens = int(input_tokens) if input_tokens is not None else None
            completion_tokens = int(output_tokens) if output_tokens is not None else None
            if prompt_tokens is not None and completion_tokens is not None:
                total_tokens = int(prompt_tokens + completion_tokens)
        except Exception:
            pass

        self.record_prompt(
            system=system,
            user=user,
            rendered_before=prompt,
            rendered_after=prompt,
            tokens_before=prompt_tokens,
            tokens_after=prompt_tokens,
            max_input_tokens=None,
            truncation_side=None,
        )

        self.add_report(
            LLMCallReport(
                backend="azure",
                model=self.deployment,
                latency_ms=(t1 - t0) * 1000.0,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                prompt_chars=len(prompt),
                completion_chars=len(text),
            )
        )

        return text

    def generate_messages_structured(self, messages: List[Dict[str, str]], response_format: Any) -> Any:
        from openai import OpenAI

        client = OpenAI(base_url=self.endpoint, api_key=self.api_key)
        max_retries = int(
            os.environ.get(f"{self.prefix}_MAX_RETRIES", os.environ.get("LLM_MAX_RETRIES", "5"))
        )
        for attempt in range(1, max_retries + 1):
            try:
                t0 = time.perf_counter()
                response = client.beta.chat.completions.parse(
                    model=self.deployment,
                    messages=messages,
                    response_format=response_format,
                    temperature=0.0,
                )
                parsed = response.choices[0].message.parsed
                if parsed is None:
                    raise ValueError("provider returned no parsed structured response")
                usage = response.usage
                self.add_report(
                    LLMCallReport(
                        backend="azure",
                        model=self.deployment,
                        latency_ms=(time.perf_counter() - t0) * 1000.0,
                        prompt_tokens=getattr(usage, "prompt_tokens", None),
                        completion_tokens=getattr(usage, "completion_tokens", None),
                        total_tokens=getattr(usage, "total_tokens", None),
                        prompt_chars=sum(len(m["content"]) for m in messages),
                        completion_chars=len(parsed.model_dump_json()),
                    )
                )
                return parsed
            except Exception as error:
                print(
                    f"[llm retry] party={self.prefix.lower() or 'unknown'} model={self.deployment} "
                    f"attempt={attempt}/{max_retries} error={type(error).__name__}: {error}",
                    file=sys.stderr,
                    flush=True,
                )
                if attempt == max_retries:
                    raise RuntimeError(
                        f"Structured LLM request failed after {max_retries} attempts: "
                        f"party={self.prefix or 'unknown'} model={self.deployment}"
                    ) from error
                time.sleep(5)


def _parse_max_new_tokens(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"", "none", "null", "unlimited", "no_cap", "nocap", "inf", "infinite", "0", "-1"}:
            return None
        value = v
    try:
        n = int(value)
    except Exception:
        return None
    if n <= 0:
        return None
    return n


def build_generator_from_env(prefix: str, max_new_tokens: int | None | object = ... ) -> TextGenerator:
    """Helper to create a generator from environment variables.

    Examples
    --------
    - {prefix}_BACKEND=openai; {prefix}_OPENAI_MODEL=gpt-4o-mini
    - {prefix}_BACKEND=hf; {prefix}_HF_MODEL=meta-llama/Meta-Llama-3-8B-Instruct
    """
        

    backend = os.environ.get(f"{prefix}_BACKEND", "hf").lower()
    env_max_new_tokens = _parse_max_new_tokens(os.environ.get(f"{prefix}_MAX_NEW_TOKENS"))
    resolved_max_new_tokens = env_max_new_tokens if max_new_tokens is ... else _parse_max_new_tokens(max_new_tokens)

    if backend == "azure":
        api_key = (
            os.environ.get(f"{prefix}_AZURE_API_KEY")
            or os.environ.get("AZURE_OPENAI_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
        )
        if not api_key:
            raise RuntimeError(
                f"Missing Azure API key. Set {prefix}_AZURE_API_KEY or AZURE_OPENAI_API_KEY."
            )
        endpoint = os.environ.get(
            f"{prefix}_AZURE_ENDPOINT",
            os.environ.get(
                "AZURE_OPENAI_ENDPOINT",
                "https://qcri-sakina.services.ai.azure.com/openai/v1/",
            ),
        )
        api_version = os.environ.get(
            f"{prefix}_AZURE_API_VERSION",
            os.environ.get("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
        )
        deployment = os.environ.get(f"{prefix}_AZURE_DEPLOYMENT", "gpt-4.1")
        gen = AzureOpenAIChatGenerator(
            deployment=deployment,
            api_key=api_key,
            endpoint=endpoint,
            api_version=api_version,
            prefix=prefix,
        )
        gen.max_new_tokens = resolved_max_new_tokens
        print(f"[model] party={prefix.lower()} backend=azure model={deployment}", flush=True)
        return gen
    if backend == "callable":
        raise RuntimeError(
            "CALLABLE backend must be constructed in code (pass CallableGenerator(fn=...))."
        )
    if backend in {"archer", "offline_archer", "offlinearcher"}:
        model_name = (
            os.environ.get(f"{prefix}_ARCHER_MODEL")
            or os.environ.get(f"{prefix}_HF_MODEL")
            or os.environ.get("ACTOR_MODEL_NAME")
            or "Qwen/Qwen2.5-7B-Instruct"
        )
        checkpoint_path = os.environ.get(f"{prefix}_ARCHER_CHECKPOINT") or os.environ.get("ARCHER_CHECKPOINT")
        gen = ArcherPolicyGenerator(
            model_name_or_path=model_name,
            checkpoint_path=checkpoint_path,
            prompt_style=os.environ.get(f"{prefix}_ARCHER_PROMPT_STYLE", "raw"),
            torch_dtype=os.environ.get(
                f"{prefix}_ARCHER_TORCH_DTYPE",
                os.environ.get(f"{prefix}_HF_TORCH_DTYPE", "bf16"),
            ),
            device=os.environ.get(f"{prefix}_ARCHER_DEVICE") or os.environ.get(f"{prefix}_HF_DEVICE"),
            max_input_length=int(os.environ.get(f"{prefix}_ARCHER_MAX_INPUT_LENGTH", "512")),
            actor_use_lora=_env_bool(os.environ.get(f"{prefix}_ARCHER_USE_LORA"), False),
            actor_lora_r=int(os.environ.get(f"{prefix}_ARCHER_LORA_R", "16")),
            actor_lora_alpha=int(os.environ.get(f"{prefix}_ARCHER_LORA_ALPHA", "32")),
            actor_lora_dropout=float(os.environ.get(f"{prefix}_ARCHER_LORA_DROPOUT", "0.05")),
            actor_gradient_checkpointing=_env_bool(
                os.environ.get(f"{prefix}_ARCHER_GRADIENT_CHECKPOINTING"),
                False,
            ),
            system_prompt_path=os.environ.get(f"{prefix}_ARCHER_SYSTEM_PROMPT_PATH"),
            use_sweet_rl_prompt=_env_bool(os.environ.get(f"{prefix}_ARCHER_USE_SWEET_RL_PROMPT"), False),
            max_new_tokens=resolved_max_new_tokens,
        )
        gen.max_new_tokens = resolved_max_new_tokens
        print(f"[model] party={prefix.lower()} backend=archer model={gen.model_label}", flush=True)
        return gen
    if backend in {"aria", "aria_rl"}:
        model_name = (
            os.environ.get(f"{prefix}_ARIA_MODEL")
            or os.environ.get(f"{prefix}_HF_MODEL")
            or os.environ.get("BASE_MODEL")
            or "Qwen/Qwen2.5-7B-Instruct"
        )
        checkpoint_path = os.environ.get(f"{prefix}_ARIA_CHECKPOINT") or os.environ.get("ARIA_CHECKPOINT")
        if not checkpoint_path:
            raise RuntimeError(f"Missing ARIA checkpoint. Set {prefix}_ARIA_CHECKPOINT to trainer.pt or its checkpoint directory.")
        gen = AriaPolicyGenerator(
            model_name_or_path=model_name,
            checkpoint_path=checkpoint_path,
            torch_dtype=os.environ.get(
                f"{prefix}_ARIA_TORCH_DTYPE",
                os.environ.get(f"{prefix}_HF_TORCH_DTYPE", "bf16"),
            ),
            device=os.environ.get(f"{prefix}_ARIA_DEVICE") or os.environ.get(f"{prefix}_HF_DEVICE"),
            max_input_length=int(os.environ.get(f"{prefix}_ARIA_MAX_INPUT_LENGTH", "4096")),
            use_lora=_env_bool(os.environ.get(f"{prefix}_ARIA_USE_LORA"), True),
            lora_r=int(os.environ.get(f"{prefix}_ARIA_LORA_R", "16")),
            lora_alpha=int(os.environ.get(f"{prefix}_ARIA_LORA_ALPHA", "32")),
            lora_dropout=float(os.environ.get(f"{prefix}_ARIA_LORA_DROPOUT", "0.05")),
            system_prompt_path=os.environ.get(f"{prefix}_ARIA_SYSTEM_PROMPT_PATH"),
            max_new_tokens=resolved_max_new_tokens,
            trust_remote_code=_env_bool(os.environ.get(f"{prefix}_ARIA_TRUST_REMOTE_CODE"), True),
        )
        gen.max_new_tokens = resolved_max_new_tokens
        print(f"[model] party={prefix.lower()} backend=aria model={gen.model_label}", flush=True)
        return gen
    if backend in {"sweet_rl", "sweet-rl", "sweetrl"}:
        model_name = (
            os.environ.get(f"{prefix}_SWEET_RL_MODEL")
            or os.environ.get(f"{prefix}_HF_MODEL")
            or os.environ.get("ACTOR_MODEL_PATH")
        )
        if not model_name:
            raise RuntimeError(
                f"Missing SWEET-RL actor path. Set {prefix}_SWEET_RL_MODEL to the HF-format actor checkpoint directory."
            )
        hf_device = os.environ.get(f"{prefix}_SWEET_RL_DEVICE") or os.environ.get(f"{prefix}_HF_DEVICE")
        hf_device_map = os.environ.get(f"{prefix}_SWEET_RL_DEVICE_MAP") or os.environ.get(f"{prefix}_HF_DEVICE_MAP")
        hf_torch_dtype = os.environ.get(f"{prefix}_SWEET_RL_TORCH_DTYPE") or os.environ.get(f"{prefix}_HF_TORCH_DTYPE", "bf16")
        hf_trust_remote_code = _env_bool(
            os.environ.get(f"{prefix}_SWEET_RL_TRUST_REMOTE_CODE", os.environ.get(f"{prefix}_HF_TRUST_REMOTE_CODE")),
            False,
        )
        hf_attn_impl = os.environ.get(f"{prefix}_SWEET_RL_ATTN_IMPLEMENTATION") or os.environ.get(f"{prefix}_HF_ATTN_IMPLEMENTATION")

        model_kwargs: Dict[str, Any] = {}
        if hf_attn_impl:
            model_kwargs["attn_implementation"] = hf_attn_impl

        gen = SweetRLPolicyGenerator(
            model_name,
            prompt_path=os.environ.get(f"{prefix}_SWEET_RL_PROMPT_PATH") or os.environ.get("AGENT_PROMPT_PATH"),
            assistant_label=os.environ.get(f"{prefix}_SWEET_RL_ASSISTANT_LABEL", "Therapist"),
            user_label=os.environ.get(f"{prefix}_SWEET_RL_USER_LABEL", "Client"),
            device=hf_device,
            device_map=hf_device_map,
            torch_dtype=hf_torch_dtype,
            trust_remote_code=hf_trust_remote_code,
            max_input_length=int(os.environ.get(f"{prefix}_SWEET_RL_MAX_INPUT_LENGTH", "16384")),
            temperature=float(os.environ.get(f"{prefix}_SWEET_RL_TEMPERATURE", "0.7")),
            top_p=float(os.environ.get(f"{prefix}_SWEET_RL_TOP_P", "0.8")),
            **model_kwargs,
        )
        gen.max_new_tokens = resolved_max_new_tokens
        print(f"[model] party={prefix.lower()} backend=sweet_rl model={model_name}", flush=True)
        return gen

    model_name = os.environ.get(f"{prefix}_HF_MODEL")
    if not model_name:
        raise RuntimeError(
            f"Missing env var {prefix}_HF_MODEL. Set {prefix}_BACKEND=azure to use Azure, or set {prefix}_HF_MODEL for a local Hugging Face model."
        )

    hf_device = os.environ.get(f"{prefix}_HF_DEVICE")
    hf_device_map = os.environ.get(f"{prefix}_HF_DEVICE_MAP")
    hf_torch_dtype = os.environ.get(f"{prefix}_HF_TORCH_DTYPE", "bf16")
    hf_trust_remote_code = os.environ.get(f"{prefix}_HF_TRUST_REMOTE_CODE", "false").strip().lower() in {"1", "true", "yes", "y", "on"}
    hf_attn_impl = os.environ.get(f"{prefix}_HF_ATTN_IMPLEMENTATION")

    model_kwargs: Dict[str, Any] = {}
    if hf_attn_impl:
        model_kwargs["attn_implementation"] = hf_attn_impl

    gen = TransformersChatGenerator(
        model_name,
        device=hf_device,
        device_map=hf_device_map,
        torch_dtype=hf_torch_dtype,
        trust_remote_code=hf_trust_remote_code,
        max_input_length=int(os.environ.get(f"{prefix}_HF_MAX_INPUT_LENGTH", "512")),
        prompt_style=os.environ.get(f"{prefix}_HF_PROMPT_STYLE", "chat"),
        **model_kwargs,
    )
    gen.max_new_tokens = resolved_max_new_tokens
    print(f"[model] party={prefix.lower()} backend=hf model={model_name}", flush=True)
    return gen
