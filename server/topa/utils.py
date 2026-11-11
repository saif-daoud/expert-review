import time
import os
import json
import logging
import numpy as np
from tqdm import tqdm
from typing import List, Optional
from openai import OpenAI, AzureOpenAI
import torch
from transformers import AutoModel

def build_logger():
    # disable noisy libs globally
    for name in [
        "httpx",
        "httpcore",
        "httpcore.connection",
        "httpcore.http11",
        "httpcore.connection_pool",
        "openai",
        "pydot"
    ]:
        l = logging.getLogger(name)
        l.propagate = False
        l.handlers.clear()
        l.addHandler(logging.NullHandler())

    return logging.getLogger("topa")

logger = build_logger()

def parse_json(output: str):
    cleaned_output = output.strip()
    start_tag = "```json"
    end_tag = "```"

    start_index = cleaned_output.find(start_tag)
    if start_index == -1:
        start_index = 0
    else:
        start_index += len(start_tag)

    cleaned_output = cleaned_output[start_index:]

    end_index = cleaned_output.rfind(end_tag)
    if end_index != -1:
        cleaned_output = cleaned_output[:end_index]
    
    return json.loads(cleaned_output.strip())

def format_transcript(
    dialogue,
    *,
    system_label: str = "SYSTEM",
    user_label: str = "USER",
) -> str:
    """Format a dialogue into a standardized transcript.

    Output format (per line):
      "<role> (<role>_<idx>): <text>"

    This accepts heterogeneous turn dicts with keys like:
      - speaker / role / from / author
      - text / utterance / content
      - utterance_id (optional; used verbatim when present)
    """
    if dialogue is None:
        return ""

    counts = {}

    def _next_id(label: str) -> str:
        idx = counts.get(label, 0)
        counts[label] = idx + 1
        return f"{label}_{idx}"

    sys_norm = str(system_label or "").strip().lower()
    usr_norm = str(user_label or "").strip().lower()

    lines = []
    for turn in dialogue:
        if not isinstance(turn, dict):
            label = system_label or "SYSTEM"
            utt_id = _next_id(label)
            lines.append(f"{label} ({utt_id}): {str(turn)}")
            continue

        speaker_raw = (
            turn.get("speaker")
            or turn.get("role")
            or turn.get("from")
            or turn.get("author")
        )
        speaker = "" if speaker_raw is None else str(speaker_raw)
        sp_norm = speaker.strip().lower()

        if sp_norm in {"system", "assistant"} or (sys_norm and sp_norm == sys_norm):
            label = system_label
        elif sp_norm in {"user", "human"} or (usr_norm and sp_norm == usr_norm):
            label = user_label
        else:
            label = speaker if speaker else (user_label or "USER")

        text = turn.get("text")
        if text is None:
            text = turn.get("utterance")
        if text is None:
            text = turn.get("content")
        text = "" if text is None else str(text)

        utt_id = None
        if isinstance(turn.get("utterance_id"), (str, int)):
            utt_id = str(turn.get("utterance_id"))
        elif isinstance(turn.get("utteranceId"), (str, int)):
            utt_id = str(turn.get("utteranceId"))

        if not utt_id:
            utt_id = _next_id(label)

        lines.append(f"{label} ({utt_id}): {text}")

    return "\n".join(lines)

def _llm_provider() -> str:
    """Pick provider.

    Order:
      1) TOPA_LLM_PROVIDER env if set (azure|openai|vllm)
      2) AZURE_OPENAI_ENDPOINT env -> azure
      3) fallback -> openai
    """
    prov = (os.getenv("TOPA_LLM_PROVIDER") or "").strip().lower()
    if prov in {"azure", "openai", "vllm"}:
        return prov
    if os.getenv("AZURE_OPENAI_ENDPOINT"):
        return "azure"
    return "openai"

def _azure_client(api_key: str):
    api_version = (os.getenv("AZURE_OPENAI_API_VERSION") or "").strip()
    endpoint = (os.getenv("AZURE_OPENAI_ENDPOINT") or "").strip()
    if not endpoint:
        raise ValueError("AZURE_OPENAI_ENDPOINT is not set, cannot use Azure provider")
    return AzureOpenAI(azure_endpoint=endpoint, api_key=api_key, api_version=api_version)

def _openai_client(api_key: str):
    if _llm_provider() == "vllm":
        endpoint = (os.getenv("OPENAI_BASE_URL") or "http://127.0.0.1:8000/v1").strip()
        return OpenAI(
            base_url=endpoint,
            api_key=api_key or "EMPTY",
            timeout=float(os.getenv("TOPA_LLM_TIMEOUT") or "1800"),
        )
    endpoint = (os.getenv("OPENAI_BASE_URL") or os.getenv("AZURE_OPENAI_ENDPOINT") or "").strip()
    return OpenAI(base_url=endpoint or None, api_key=api_key)


def _llm_request_kwargs() -> dict:
    """Local Qwen settings; hosted providers keep their existing request options."""
    if _llm_provider() != "vllm":
        return {}
    enable_thinking = (os.getenv("TOPA_LLM_ENABLE_THINKING") or "false").strip().lower()
    return {
        "max_tokens": int(os.getenv("TOPA_LLM_MAX_TOKENS") or "16384"),
        "temperature": float(os.getenv("TOPA_LLM_TEMPERATURE") or "0"),
        "extra_body": {
            "chat_template_kwargs": {"enable_thinking": enable_thinking in {"true", "1", "yes"}},
        },
    }

def run_llm_query(prompt, api_key):
    provider = _llm_provider()
    max_attempts = int(os.getenv("TOPA_LLM_MAX_ATTEMPTS") or "4")

    if provider == "azure":
        client = _azure_client(api_key)
        model = (os.getenv("AZURE_OPENAI_DEPLOYMENT") or "gpt-4.1").strip()
    else:
        client = _openai_client(api_key)
        default_model = "Qwen/Qwen3.5-122B-A10B" if provider == "vllm" else "gpt-4o-mini"
        model = (os.getenv("OPENAI_MODEL") or default_model).strip()

    last_err = None
    for attempt in range(1, max_attempts + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                **_llm_request_kwargs(),
            )
            output_text = response.choices[0].message.content
            # if not output_text or not output_text.strip():
            #     raise RuntimeError("Empty completion (check reasoning mode and the output token limit)")
            input_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens
            return output_text, input_tokens, output_tokens
        except Exception as e:
            last_err = e
            logger.info(f"LLM call failed (attempt {attempt}/{max_attempts}): {e}")
            time.sleep(min(5 * attempt, 10))

    raise RuntimeError(f"LLM call failed after {max_attempts} attempts: {last_err}")

def get_llm_response(prompt, api_key):
    """Wrapper that returns only the text."""
    text, _, _ = run_llm_query(prompt, api_key)
    return text

def get_llm_response_messages(prompt, api_key, max_tokens: int = 128, temperature: float = 0, n: int = 1):
    """Chat completion with multi-message input support."""
    provider = _llm_provider()
    max_attempts = int(os.getenv("TOPA_LLM_MAX_ATTEMPTS") or "8")

    if provider == "azure":
        client = _azure_client(api_key)
        model = (os.getenv("AZURE_OPENAI_DEPLOYMENT") or "gpt-4.1").strip()
    else:
        client = _openai_client(api_key)
        default_model = "Qwen/Qwen3.5-122B-A10B" if provider == "vllm" else "gpt-4o-mini"
        model = (os.getenv("OPENAI_MODEL") or default_model).strip()

    if isinstance(prompt, list):
        chat_prompt = prompt
    elif isinstance(prompt, str):
        chat_prompt = [{"role": "user", "content": prompt}]
    else:
        raise Exception(f"Invalid prompt format: {prompt}")

    request_kwargs = _llm_request_kwargs()
    request_kwargs.update(max_tokens=max_tokens, temperature=temperature, n=n)
    last_err = None
    for attempt in range(1, max_attempts + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=chat_prompt,
                **request_kwargs,
            )
            if n == 1:
                output = (response.choices[0].message.content or "").strip()
                if output == "":
                    raise RuntimeError("Empty completion")
                return output

            outputs = []
            for choice in response.choices:
                out = choice.message.content
                if out is not None:
                    out = out.strip()
                    if out:
                        outputs.append(out)
            if not outputs:
                raise RuntimeError("All completions empty")
            return outputs
        except Exception as e:
            last_err = e
            logger.info(f"LLM call failed (attempt {attempt}/{max_attempts}): {e}")
            time.sleep(min(5 * attempt, 10))

    raise RuntimeError(f"LLM call failed after {max_attempts} attempts: {last_err}")

# ========================
# Embedder helpers
# ========================
class OpenAIEmbedder:
    def __init__(self, api_key: str, model: str = "text-embedding-3-small"):
        self.model = model
        self.client = OpenAI(api_key=api_key)

    def embed(self, text: str) -> np.ndarray:
        resp = self.client.embeddings.create(model=self.model, input=text)
        vec = np.array(resp.data[0].embedding, dtype=np.float32)
        return vec

    def embed_many(self, texts: List[str]) -> np.ndarray:
        resp = self.client.embeddings.create(model=self.model, input=texts)
        vecs = [d.embedding for d in resp.data]
        return np.array(vecs, dtype=np.float32)

class JinaLocalEmbedder:
    """
    Local wrapper for jinaai/jina-embeddings-v4.
    - For RAG, use prompt_name="query" for queries, prompt_name="passage" for docs.
    """

    def __init__(
        self,
        model_id: str = "jinaai/jina-embeddings-v4",
        device: Optional[str] = None,
        torch_dtype: Optional[torch.dtype] = None,
        task: str = "retrieval",
        prompt_name: Optional[str] = "query",
        truncate_dim: Optional[int] = None,
        batch_size: int = 8,
        return_multivector: bool = False
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if torch_dtype is None:
            torch_dtype = torch.float16 if device.startswith("cuda") else torch.float32

        self.task = task
        self.prompt_name = prompt_name
        self.truncate_dim = truncate_dim
        self.batch_size = batch_size
        self.return_multivector = return_multivector

        self.model = AutoModel.from_pretrained(
            model_id,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
        ).to(device)

        # Disable jina-embeddings-v4 tqdm progress bar
        if hasattr(self.model, "verbosity"):
            self.model.verbosity = 0
        if hasattr(self.model, "config") and hasattr(self.model.config, "verbosity"):
            self.model.config.verbosity = 0
        for obj in (self.model, getattr(self.model, "model", None)):
            if obj is not None and hasattr(obj, "verbosity"):
                obj.verbosity = 0

        self.model.eval()

    def _encode_texts(self, texts: List[str]):
        return self.model.encode_text(
            texts=texts,
            task=self.task,
            prompt_name=self.prompt_name,
            truncate_dim=self.truncate_dim,
            batch_size=len(texts), # we batch outside
            return_multivector=self.return_multivector,
        )

    @staticmethod
    def _to_numpy(out) -> np.ndarray:
        stacked = torch.stack([t.detach().to("cpu") for t in out], dim=0)
        return stacked.float().numpy()

    def embed(self, text: str) -> np.ndarray:
        X = self.embed_many([text], show_progress=False)
        return X[0].astype(np.float32, copy=False)

    def embed_many(self, texts, show_progress: bool = True):
        n = len(texts)
        if n == 0:
            return np.empty((0, 0), dtype=np.float32)

        bs = max(1, self.batch_size)
        batches = range(0, n, bs)

        pbar = tqdm(total=n, desc="Embedding", unit="text", disable=not show_progress)

        all_vecs = []
        with torch.inference_mode():
            for i in batches:
                chunk = texts[i : i + bs]
                out = self._encode_texts(chunk)
                X = self._to_numpy(out).astype(np.float32, copy=False)

                if X.ndim == 1:
                    X = X[None, :]

                all_vecs.append(X)
                pbar.update(len(chunk))
        pbar.close()

        return np.concatenate(all_vecs, axis=0)
    
def process_dialogue(dialogue: List[dict], domain_params: dict):
    system = domain_params["system"]
    user = domain_params["user"]

    utterances = []
    sys_c = 0
    usr_c = 0
    for turn in dialogue:
        if turn["speaker"] == "system":
            utt_id = f"{system}_{sys_c}"
            sys_c += 1
            speaker = system
        elif turn["speaker"] == "user":
            utt_id = f"{user}_{usr_c}"
            usr_c += 1
            speaker = user
        else:
            raise Exception(f"Unkown speaker {turn['speaker']} - Utterance : {turn}")
        utterances.append({"utterance_id": utt_id, "speaker": speaker, "utterance": turn["text"]})
    return utterances

def build_session_index(data: list, domain_params: dict):
    """Build (user_idx, session_idx) -> list[utterances], plus id->pos map."""
    sessions = {}
    pos = {}
    for user_idx, user in enumerate(data):
        for session_idx, session in enumerate(user["sessions"]):
            utterances = process_dialogue(session["dialogue"], domain_params)
            sessions[(user_idx, session_idx)] = utterances
            pos[(user_idx, session_idx)] = {u["utterance_id"]: i for i, u in enumerate(utterances)}
    return sessions, pos

