from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from .topa_models import MLPPolicy, RecurrentSFTConvStateProbe

def norm_str(x: Any) -> str:
    if x is None:
        return ""
    s = str(x)
    return " ".join(s.strip().split())

def safe_name(s: str) -> str:
    s = norm_str(str(s))
    s = re.sub(r"[^a-z0-9._-]+", "_", s)
    return s.strip("_")


def _read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


class ActionSpace:
    """Action-space helper for this standalone ``simulations`` folder.

    Supported inputs:
      1) a directory with ``macro_actions.json`` and ``micro_actions.json``
         such as ``simulations/topa_components``;
      2) a single JSON file containing a list of macro dicts;
      3) a single JSON file containing one of: macros, macro_actions, macros_list.

    Internally each macro is normalized to contain a ``micro_actions`` list.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if self.path.is_dir():
            macros = self._load_components_dir(self.path)
        else:
            macros = self._load_action_space_file(self.path)

        self.macros: List[Dict[str, Any]] = list(macros)
        self._macro_by_name: Dict[str, Dict[str, Any]] = {str(m.get("name")): m for m in self.macros}
        self._macro_names: List[str] = [str(m.get("name")) for m in self.macros]
        self._macro2id: Dict[str, int] = {m: i for i, m in enumerate(self._macro_names)}

        # Optional semantic schema for decoding the frozen conversation-state
        # probe vector. This is present in TOPAS component directories.
        self.state_specs: List[Dict[str, Any]] = []
        if self.path.is_dir():
            state_path = self.path / "conversation_states.json"
            if state_path.exists():
                state_obj = _read_json(state_path)
                if isinstance(state_obj, list):
                    self.state_specs = list(state_obj)
                elif isinstance(state_obj, dict):
                    for key in ("conversation_states", "states", "state_specs"):
                        if isinstance(state_obj.get(key), list):
                            self.state_specs = list(state_obj[key])
                            break

    @staticmethod
    def _extract_list(obj: Any, *keys: str) -> List[Dict[str, Any]]:
        if isinstance(obj, list):
            return list(obj)
        if isinstance(obj, dict):
            for key in keys:
                val = obj.get(key)
                if isinstance(val, list):
                    return list(val)
        raise TypeError(f"Unsupported action-space JSON structure; expected list or one of keys={keys}")

    def _load_components_dir(self, components_dir: Path) -> List[Dict[str, Any]]:
        macro_obj = _read_json(components_dir / "macro_actions.json")
        micro_obj = _read_json(components_dir / "micro_actions.json")
        macros = self._extract_list(macro_obj, "macro_actions", "macros", "macros_list")
        micro_groups = self._extract_list(micro_obj, "macro_actions", "macros", "macros_list")
        micro_by_name = {str(m.get("name", "")): m for m in micro_groups}

        normalized: List[Dict[str, Any]] = []
        for i, macro in enumerate(macros):
            item = dict(macro)
            name = str(item.get("name", ""))
            group = micro_by_name.get(name)
            if group is None and i < len(micro_groups):
                group = micro_groups[i]
            item["micro_actions"] = list((group or {}).get("micro_actions", []) or [])
            normalized.append(item)
        return normalized

    def _load_action_space_file(self, path: Path) -> List[Dict[str, Any]]:
        obj = _read_json(path)
        macros = self._extract_list(obj, "macros", "macro_actions", "macros_list")
        return list(macros)

    def macro_names(self) -> List[str]:
        return list(self._macro_names)

    def macro_id(self, macro: str) -> int:
        return int(self._macro2id[str(macro)])

    def macro_goal(self, macro: str) -> str:
        m = self._macro_by_name.get(str(macro), {})
        goal = m.get("goal") or m.get("objective") or ""
        if isinstance(goal, dict):
            return str(goal.get("objective") or goal.get("goal") or "").strip()
        return str(goal).strip()

    def macro_description(self, macro: str) -> str:
        m = self._macro_by_name.get(str(macro), {})
        return str(m.get("description") or m.get("desc") or m.get("definition") or "").strip()

    def micro_actions_for(self, macro: str) -> List[Dict[str, Any]]:
        m = self._macro_by_name.get(str(macro), {})
        return list(m.get("micro_actions") or m.get("actions") or [])

    def micro_names_for(self, macro: str) -> List[str]:
        return [str(a.get("name")) for a in self.micro_actions_for(macro)]

    def micro_description(self, macro: str, micro: str) -> str:
        for a in self.micro_actions_for(macro):
            if str(a.get("name")) == str(micro):
                return str(a.get("description") or a.get("desc") or "").strip()
        return ""

    def macro_for_micro(self, micro: str) -> Optional[str]:
        for macro in self.macro_names():
            if str(micro) in self.micro_names_for(macro):
                return macro
        return None

    def instruction_text(self, macro: str, micro: str) -> str:
        goal = self.macro_goal(macro)
        macro_desc = self.macro_description(macro)
        micro_desc = self.micro_description(macro, micro)
        lines = [f"Option: {macro}"]
        if goal:
            lines.append(f"Option Goal: {goal}")
        if macro_desc:
            lines.append(f"Option Description: {macro_desc}")
        lines.append(f"Primitive Action: {micro}")
        if micro_desc:
            lines.append(f"Primitive Action Description: {micro_desc}")
        return "\n".join(lines)

    def micro_list_text(self, macro: str) -> str:
        lines = []
        for item in self.micro_actions_for(macro):
            name = str(item.get("name", "")).strip()
            desc = str(item.get("description") or item.get("desc") or "").strip()
            if not name:
                continue
            lines.append(f"- {name}: {desc}" if desc else f"- {name}")
        return "\n".join(lines) if lines else "- none"

@dataclass
class LatentConfig:
    """How to extract the activation vector used by the policies."""

    layer: int = 22
    pooling: str = "last"  # last | mean
    system_tag: str = "<|therapist|>"
    user_tag: str = "<|patient|>"
    prefix: str = "Act as the therapist agent and continue the conversation: "
    user_template: str = "{user}"
    append_system_tag: bool = True


def dialogue_to_tagged_text(
    dialogue: List[Dict[str, Any]],
    *,
    system_tag: str,
    user_tag: str,
    append_system_tag: bool = True,
) -> str:
    utterances: List[str] = []
    for turn in dialogue:
        speaker = str(turn.get("speaker", "") or "").strip().lower()
        text = str(turn.get("text", "") or "").strip()
        if not text:
            continue
        is_system = speaker in {"system", "assistant", "therapist", "persuader"}
        tag = system_tag if is_system else user_tag
        utterances.append(f"{tag} {text}\n")
    if append_system_tag:
        utterances.append(system_tag)
    return "".join(utterances)


def dialogue_to_readable_text(dialogue: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    for turn in dialogue:
        speaker = str(turn.get("speaker", "user") or "user").strip()
        text = str(turn.get("text", "") or "").strip()
        if text:
            lines.append(f"{speaker}: {text}")
    return "\n".join(lines)


def _find_subsequence_positions(batch_ids: torch.Tensor, pattern: Sequence[int]) -> List[List[int]]:
    if batch_ids.ndim != 2:
        raise ValueError("batch_ids must be 2D [B, T]")
    pat = [int(x) for x in pattern]
    if not pat:
        return [[] for _ in range(batch_ids.size(0))]
    m = len(pat)
    out: List[List[int]] = []
    for b in range(batch_ids.size(0)):
        seq = [int(x) for x in batch_ids[b].tolist()]
        positions = []
        for i in range(0, len(seq) - m + 1):
            if seq[i : i + m] == pat:
                positions.append(i)
        out.append(positions)
    return out


class LatentExtractor:
    """Extracts the activation vector from the local HF generator's model.

    The generator is expected to expose ``model`` and ``tokenizer`` attributes,
    like TOPA's ``TransformersChatGenerator``.
    """

    def __init__(self, generator: Any, cfg: LatentConfig) -> None:
        if not hasattr(generator, "model") or not hasattr(generator, "tokenizer"):
            raise TypeError(
                "Activation policies need a local HF generator exposing .model and .tokenizer. "
                "Use the same local model used for activation extraction."
            )
        self.generator = generator
        self.cfg = cfg
        added = self.generator.tokenizer.add_special_tokens(
            {"additional_special_tokens": [str(cfg.system_tag), str(cfg.user_tag)]}
        )
        if added:
            self.generator.model.resize_token_embeddings(len(self.generator.tokenizer))

    def build_routing_text(self, dialogue: List[Dict[str, Any]], *, current_macro: Optional[str] = None) -> str:
        transcript = dialogue_to_tagged_text(
            dialogue,
            system_tag=str(self.cfg.system_tag),
            user_tag=str(self.cfg.user_tag),
            append_system_tag=bool(self.cfg.append_system_tag),
        )
        prefix = str(self.cfg.prefix).strip()
        if prefix:
            prefix += " "
        return prefix + transcript

    @torch.no_grad()
    def extract(self, routing_text: str) -> torch.Tensor:
        model = self.generator.model
        tokenizer = self.generator.tokenizer
        model.eval()
        device = next(model.parameters()).device

        prefix = str(self.cfg.user_template).format(user=routing_text)
        enc = tokenizer(
            prefix,
            add_special_tokens=True,
            return_tensors="pt",
        ).to(device)

        layer = int(self.cfg.layer)
        captured = {}

        layers = model.model.layers
        if layer < 0 or layer >= len(layers):
            raise ValueError(
                f"layer={layer} is out of range for {len(layers)} transformer blocks"
            )

        target_layer = layers[layer]

        def pre_hook_fn(module, inputs):
            if not inputs:
                raise RuntimeError(
                    f"Transformer block {layer} received no positional hidden-state input"
                )
            captured["h"] = inputs[0].detach()

        handle = target_layer.register_forward_pre_hook(pre_hook_fn)

        try:
            _ = model(
                input_ids=enc["input_ids"],
                attention_mask=enc.get("attention_mask", None),
                use_cache=False,
                output_hidden_states=False,   # IMPORTANT: no all-layer hidden states
                return_dict=True,
            )
        finally:
            handle.remove()

        h = captured["h"]  # [B, T, H]

        tag_ids = tokenizer(str(self.cfg.system_tag), add_special_tokens=False).input_ids

        if tag_ids:
            pos_by_batch = _find_subsequence_positions(enc["input_ids"], tag_ids)
            reps = []

            for b, positions in enumerate(pos_by_batch):
                if not positions:
                    reps.append(h[b, -1, :])
                elif str(self.cfg.pooling).lower() == "mean":
                    reps.append(h[b, positions, :].mean(dim=0))
                else:
                    reps.append(h[b, positions[-1], :])

            rep = torch.stack(reps, dim=0)
        else:
            if str(self.cfg.pooling).lower() == "mean":
                rep = h.mean(dim=1)
            else:
                rep = h[:, -1, :]

        rep = rep.detach()

        del h, enc, captured
        torch.cuda.empty_cache()

        return rep

@dataclass
class RouterOutput:
    label: str
    label_id: int
    prob: float
    probs: List[float]
    # Raw network outputs. For SFT these are classification logits; for an
    # RL actor they are policy scores/logits, not necessarily critic Q-values.
    scores: List[float]
    entropy: float
    top2_margin: Optional[float]


@dataclass
class TerminationOutput:
    terminate: bool
    prob_terminate: float
    current_macro_id: int
    current_macro: str
    current_logit: float
    logits: List[float]
    probs_all: List[float]
    threshold: float


def _read_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _meta_from_metrics(path: Path) -> dict:
    obj = _read_json(path)
    if isinstance(obj, dict) and isinstance(obj.get("meta"), dict):
        return dict(obj["meta"])
    return dict(obj)


def _apply_policy_conv_state_only(
    base_x: torch.Tensor,
    conv_state: Optional[torch.Tensor],
    enabled: bool,
    progress_state: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Drop only activations, retaining micro progress and conversation state."""
    if not enabled:
        return base_x
    if conv_state is None:
        raise RuntimeError("Conversation-state-only policy requires a conversation state.")
    parts = []
    if progress_state is not None:
        parts.append(progress_state.to(base_x.device).float())
    parts.append(conv_state.to(base_x.device).float())
    return torch.cat(parts, dim=-1)


def _infer_mlp_kwargs(state_dict: dict, meta: Optional[dict] = None) -> dict:
    """Infer enough architecture parameters to rebuild the saved TOPA MLPPolicy.

    Supports both the legacy architecture and the optional macro-action history
    branch used by the updated offline-RL policy-over-options checkpoints.
    """
    meta = dict(meta or {})
    conv_state_dim = int(meta.get("conv_state_dim", 0) or 0)
    latent_projection_dim = int(
        meta.get("latent_projection_dim", meta.get("projection_dim", 0)) or 0
    )
    conv_weight = state_dict.get("conv_state_proj.weight")
    if conv_weight is None:
        conv_weight = state_dict.get("conv_state_proj.0.weight")
    conv_state_projection_dim = int(
        meta.get(
            "conv_state_projection_dim",
            meta.get(
                "conv_state_embedding_dim",
                conv_weight.shape[0] if conv_weight is not None else 0,
            ),
        )
        or 0
    )
    progress_weight = state_dict.get("micro_progress_state_proj.0.weight")
    progress_dim = int(meta.get("micro_progress_state_dim", progress_weight.shape[1] if progress_weight is not None else 0) or 0)
    progress_projection_dim = int(
        meta.get(
            "micro_progress_state_projection_dim",
            meta.get(
                "progress_projection_dim",
                progress_weight.shape[0] if progress_weight is not None else 0,
            ),
        )
        or 0
    )

    # Output dimension.
    if "net.3.weight" in state_dict:
        num_actions = int(state_dict["net.3.weight"].shape[0])
        hidden_dim = int(state_dict["net.0.weight"].shape[0])
        net_input_dim = int(state_dict["net.0.weight"].shape[1])
    elif "net.weight" in state_dict:
        num_actions = int(state_dict["net.weight"].shape[0])
        hidden_dim = 0
        net_input_dim = int(state_dict["net.weight"].shape[1])
    else:
        raise ValueError("Unexpected MLPPolicy state_dict: missing net.3.weight or net.weight")

    # The training model has one generic categorical-history branch. Policy-over-
    # options feeds macro IDs through it, while intra-option policies feed micro
    # IDs through the same branch.
    hist_emb_w = state_dict.get("action_history_embedding.weight")
    hist_proj_w = state_dict.get("action_history_proj.0.weight")
    if hist_emb_w is None:
        hist_emb_w = state_dict.get("macro_history_embedding.weight")
    if hist_proj_w is None:
        hist_proj_w = state_dict.get("macro_history_proj.0.weight")
    state_has_history = hist_emb_w is not None and hist_proj_w is not None

    inferred_history_num_actions = 0
    inferred_history_embedding_dim = 0
    inferred_history_projection_dim = 0
    inferred_history_len = 0
    if state_has_history:
        inferred_history_num_actions = int(hist_emb_w.shape[0]) - 1
        inferred_history_embedding_dim = int(hist_emb_w.shape[1])
        inferred_history_projection_dim = int(hist_proj_w.shape[0])
        hist_flat_dim = int(hist_proj_w.shape[1])
        if inferred_history_embedding_dim <= 0 or hist_flat_dim % inferred_history_embedding_dim != 0:
            raise ValueError("Could not infer macro history length from checkpoint shapes")
        inferred_history_len = hist_flat_dim // inferred_history_embedding_dim

    history_kind = str(meta.get("history_kind", "")).strip().lower()
    if history_kind not in {"macro_action", "micro_action"}:
        if bool(meta.get("use_micro_action_history", False)):
            history_kind = "micro_action"
        elif bool(meta.get("use_macro_action_history", False)):
            history_kind = "macro_action"
        elif state_has_history:
            history_kind = "micro_action" if str(meta.get("task", "")) == "intra_option_all" else "macro_action"
        else:
            history_kind = "none"

    prefix = "micro_action" if history_kind == "micro_action" else "macro_action"
    macro_history_len = int(
        meta.get(
            "history_length",
            meta.get(f"{prefix}_history_length", meta.get("history_branch_length", inferred_history_len)),
        )
        or inferred_history_len
    )
    macro_history_num_actions = int(
        meta.get(
            "history_num_actions",
            meta.get(
                f"{prefix}_history_num_actions",
                meta.get("history_branch_num_actions", inferred_history_num_actions),
            ),
        )
        or inferred_history_num_actions
    )
    macro_history_embedding_dim = int(
        meta.get(
            "history_embedding_dim",
            meta.get(
                f"{prefix}_history_embedding_dim",
                meta.get("history_branch_embedding_dim", inferred_history_embedding_dim),
            ),
        )
        or inferred_history_embedding_dim
    )
    macro_history_projection_dim = int(
        meta.get(
            "history_projection_dim",
            meta.get(
                f"{prefix}_history_projection_dim",
                meta.get("history_branch_projection_dim", inferred_history_projection_dim),
            ),
        )
        or inferred_history_projection_dim
    )
    uses_history = bool(macro_history_len > 0 or state_has_history)

    if not uses_history:
        history_kind = "none"
        macro_history_len = 0
        macro_history_num_actions = 0
        macro_history_embedding_dim = 0
        macro_history_projection_dim = 0

    # Raw input dimension before the model-internal embedding/projection.
    # Prefer checkpoint metadata because input_dim includes the raw L history IDs.
    if "input_dim" in meta and int(meta.get("input_dim", 0) or 0) > 0:
        in_dim = int(meta["input_dim"])
    else:
        if "latent_proj.0.weight" in state_dict:
            latent_dim_inferred = int(state_dict["latent_proj.0.weight"].shape[1])
        else:
            progress_out_dim = progress_projection_dim or progress_dim
            conv_out_dim = conv_state_projection_dim or conv_state_dim
            latent_dim_inferred = net_input_dim - (macro_history_projection_dim if uses_history else 0) - progress_out_dim - conv_out_dim
        base_in_dim = latent_dim_inferred + progress_dim + conv_state_dim
        in_dim = base_in_dim + macro_history_len

    base_in_dim = int(in_dim) - int(macro_history_len)
    latent_dim = int(meta.get("latent_dim", max(0, base_in_dim - progress_dim - conv_state_dim)) or 0)
    policy_conv_state_only = meta.get("policy_conv_state_only")
    if policy_conv_state_only is None:
        policy_conv_state_only = (
            str(meta.get("task", "")) == "policy_over_options"
            and latent_dim == 0
            and conv_state_dim > 0
        )

    return {
        "in_dim": int(in_dim),
        "latent_dim": int(latent_dim),
        "num_actions": int(meta.get("num_actions", num_actions) or num_actions),
        "hidden_dim": hidden_dim,
        "conv_state_dim": conv_state_dim,
        "latent_projection_dim": latent_projection_dim,
        "conv_state_projection_dim": conv_state_projection_dim,
        "micro_progress_state_dim": progress_dim,
        "micro_progress_state_projection_dim": progress_projection_dim,
        "use_macro_action_history": bool(uses_history),
        "history_kind": history_kind,
        "history_semantics": str(
            meta.get("history_semantics", meta.get(f"{prefix}_history_semantics", ""))
        ),
        "macro_history_len": int(macro_history_len),
        "macro_history_num_actions": int(macro_history_num_actions),
        "macro_history_embedding_dim": int(macro_history_embedding_dim),
        "macro_history_projection_dim": int(macro_history_projection_dim),
        "policy_conv_state_only": bool(policy_conv_state_only),
    }


class RecurrentSFTConvStateFusion:
    """Carries the predicted conversation state from one therapist turn to the next."""

    def __init__(
        self,
        model_dir: str | Path,
        encoding: str = "label",
        update_interval: int = 1,
        device: Optional[str] = None,
    ) -> None:
        self.model_dir = Path(model_dir)
        self.encoding = str(encoding).strip().lower()
        self.update_interval = int(update_interval)
        if self.update_interval < 1:
            raise ValueError("Conversation-state update interval must be at least 1")
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = RecurrentSFTConvStateProbe(self.model_dir).to(self.device).eval()
        self.current_state_ids = self.model.initial_state_ids.to(self.device).unsqueeze(0)
        self.previous_state_ids = self.current_state_ids.clone()
        self.last_logits: List[torch.Tensor] = []
        self.turn_index = 0
        self.updated_this_turn = False
        self.output_dim = len(self.model.dim_names) if self.encoding == "label" else self.model.one_hot_dim

    def _encode(self, state_ids: torch.Tensor) -> torch.Tensor:
        if self.encoding == "label":
            return state_ids.float()
        return self.model._state_ids_to_onehot(state_ids).float()

    @torch.no_grad()
    def append_with_details(self, latent: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        latent = latent.to(self.device).float()
        self.previous_state_ids = self.current_state_ids.clone()
        self.updated_this_turn = self.turn_index % self.update_interval == 0
        if self.updated_this_turn:
            self.last_logits = self.model.predict_next_logits(
                latent,
                self.previous_state_ids,
            )
            self.current_state_ids = torch.stack(
                [torch.argmax(logits, dim=-1) for logits in self.last_logits],
                dim=1,
            )
        self.turn_index += 1
        state = self._encode(self.current_state_ids)
        return torch.cat([latent, state], dim=-1), state

    def previous_state_vector(self) -> torch.Tensor:
        return self._encode(self.previous_state_ids)

    def decode(self, state: torch.Tensor) -> Dict[str, Any]:
        current_ids = self.current_state_ids[0].detach().cpu().tolist()
        previous_ids = self.previous_state_ids[0].detach().cpu().tolist()
        heads = []
        for head_id, name in enumerate(self.model.dim_names):
            id_to_label = {
                int(value): str(label)
                for label, value in self.model.dim_to_label2id[name].items()
            }
            current_id = int(current_ids[head_id])
            previous_id = int(previous_ids[head_id])
            logits = self.last_logits[head_id][0].detach().float().cpu()
            probabilities = F.softmax(logits, dim=-1)
            heads.append({
                "head_id": head_id,
                "state_name": name,
                "previous_class_id": previous_id,
                "previous_value": id_to_label[previous_id],
                "categories": [id_to_label[index] for index in range(len(id_to_label))],
                "logits": [float(value) for value in logits.tolist()],
                "probabilities": [float(value) for value in probabilities.tolist()],
                "predicted_class_id": current_id,
                "predicted_value": id_to_label[current_id],
                "confidence": float(probabilities[current_id]),
            })
        return {
            "enabled": True,
            "source": "sft_recurrent",
            "encoding": self.encoding,
            "updated_this_turn": self.updated_this_turn,
            "update_interval": self.update_interval,
            "dimension": int(state.shape[-1]),
            "state_head_count": len(heads),
            "heads": heads,
        }


class ActivationMLPRouter:
    """Loads one activation-space MLP policy head and predicts labels."""

    def __init__(
        self,
        *,
        actor_path: str | Path,
        labels: List[str],
        meta_path: Optional[str | Path] = None,
        device: Optional[str] = None,
        use_macro_action_history: Optional[bool] = None,
    ) -> None:
        self.actor_path = Path(actor_path)
        self.labels = list(labels)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        meta = _meta_from_metrics(Path(meta_path)) if meta_path else {}
        sd = torch.load(self.actor_path, map_location="cpu")
        kwargs = _infer_mlp_kwargs(sd, meta)
        if "macro_history_embedding.weight" in sd:
            sd = dict(sd)
            sd["action_history_embedding.weight"] = sd.pop("macro_history_embedding.weight")
            sd["action_history_proj.0.weight"] = sd.pop("macro_history_proj.0.weight")
            sd["action_history_proj.0.bias"] = sd.pop("macro_history_proj.0.bias")
        if "conv_state_proj.0.weight" in sd:
            sd = dict(sd)
            sd["conv_state_proj.weight"] = sd.pop("conv_state_proj.0.weight")
            sd["conv_state_proj.bias"] = sd.pop("conv_state_proj.0.bias")
        if int(kwargs["num_actions"]) != len(self.labels):
            raise ValueError(
                f"Label count mismatch for {self.actor_path}: model has {kwargs['num_actions']} outputs, "
                f"but labels has {len(self.labels)} entries."
            )
        checkpoint_uses_history = bool(kwargs.get("use_macro_action_history", False))
        if use_macro_action_history is not None and bool(use_macro_action_history) != checkpoint_uses_history:
            raise ValueError(
                f"Macro-history flag mismatch for {self.actor_path}: runtime requested "
                f"use_macro_action_history={bool(use_macro_action_history)}, but checkpoint "
                f"expects {checkpoint_uses_history}. Use a matching checkpoint or set the "
                "runtime flag to None for auto-detection."
            )

        self.input_dim = int(kwargs["in_dim"])
        self.latent_dim = int(kwargs["latent_dim"])
        self.conv_state_dim = int(kwargs["conv_state_dim"])
        self.micro_progress_state_dim = int(kwargs["micro_progress_state_dim"])
        self.uses_macro_history = checkpoint_uses_history
        self.history_kind = str(kwargs.get("history_kind", "none"))
        self.history_semantics = str(kwargs.get("history_semantics", ""))
        self.macro_history_len = int(kwargs.get("macro_history_len", 0))
        self.macro_history_num_actions = int(kwargs.get("macro_history_num_actions", 0))
        self.macro_history_embedding_dim = int(kwargs.get("macro_history_embedding_dim", 0))
        self.macro_history_projection_dim = int(kwargs.get("macro_history_projection_dim", 0))
        self.policy_conv_state_only = bool(kwargs.get("policy_conv_state_only", False))
        self.base_input_dim = self.input_dim - self.macro_history_len

        self.model = MLPPolicy(
            in_dim=int(kwargs["in_dim"]),
            num_actions=int(kwargs["num_actions"]),
            hidden_dim=int(kwargs["hidden_dim"]),
            dropout=0.0,
            conv_state_dim=int(kwargs["conv_state_dim"]),
            latent_projection_dim=int(kwargs["latent_projection_dim"]),
            conv_state_projection_dim=int(kwargs["conv_state_projection_dim"]),
            micro_progress_state_dim=int(kwargs["micro_progress_state_dim"]),
            micro_progress_state_projection_dim=int(kwargs["micro_progress_state_projection_dim"]),
            action_history_len=int(kwargs.get("macro_history_len", 0)),
            action_history_num_actions=int(kwargs.get("macro_history_num_actions", 0)),
            action_history_embedding_dim=int(kwargs.get("macro_history_embedding_dim", 0)),
            action_history_projection_dim=int(kwargs.get("macro_history_projection_dim", 0)),
        )
        self.model.load_state_dict(sd, strict=True)
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def predict(self, x: torch.Tensor, *, deterministic: bool = True) -> RouterOutput:
        x = x.to(self.device).float()
        if x.ndim == 1:
            x = x.unsqueeze(0)
        if x.shape[-1] != self.input_dim:
            raise RuntimeError(f"Input dim mismatch for {self.actor_path}: got {x.shape[-1]}, expected {self.input_dim}")
        logits = self.model(x)[0]
        probs_t = F.softmax(logits, dim=-1)
        if deterministic:
            idx = int(torch.argmax(probs_t).item())
        else:
            idx = int(torch.multinomial(probs_t, num_samples=1).item())
        probs = [float(v) for v in probs_t.detach().cpu().tolist()]
        scores = [float(v) for v in logits.detach().cpu().tolist()]
        entropy = float((-(probs_t * torch.log(probs_t.clamp_min(1e-12))).sum()).detach().cpu().item())
        sorted_probs = sorted(probs, reverse=True)
        top2_margin = float(sorted_probs[0] - sorted_probs[1]) if len(sorted_probs) >= 2 else None
        return RouterOutput(
            label=self.labels[idx],
            label_id=idx,
            prob=float(probs[idx]),
            probs=probs,
            scores=scores,
            entropy=entropy,
            top2_margin=top2_margin,
        )


class ActivationTerminationRouter:
    """Option-conditioned termination model.

    The saved termination head outputs one binary logit per macro action. We select
    the logit for the currently active macro and apply sigmoid.
    """

    def __init__(
        self,
        *,
        actor_path: str | Path,
        macro_names: List[str],
        meta_path: Optional[str | Path] = None,
        device: Optional[str] = None,
        threshold: float = 0.5,
        use_macro_action_history: Optional[bool] = None,
    ) -> None:
        self.router = ActivationMLPRouter(
            actor_path=actor_path,
            labels=list(macro_names),
            meta_path=meta_path,
            device=device,
            use_macro_action_history=use_macro_action_history,
        )
        self.macro2id: Dict[str, int] = {m: i for i, m in enumerate(macro_names)}
        self.threshold = float(threshold)

    @property
    def input_dim(self) -> int:
        return int(self.router.input_dim)

    @property
    def latent_dim(self) -> int:
        return int(self.router.latent_dim)

    @property
    def conv_state_dim(self) -> int:
        return int(self.router.conv_state_dim)

    @property
    def uses_macro_history(self) -> bool:
        return bool(self.router.uses_macro_history)

    @property
    def macro_history_len(self) -> int:
        return int(self.router.macro_history_len)

    @torch.no_grad()
    def predict(
        self,
        x: torch.Tensor,
        current_macro: str,
        *,
        deterministic: bool = True,
    ) -> TerminationOutput:
        x = x.to(self.router.device).float()
        if x.ndim == 1:
            x = x.unsqueeze(0)
        if x.shape[-1] != self.router.input_dim:
            raise RuntimeError(
                f"Input dim mismatch for termination {self.router.actor_path}: got {x.shape[-1]}, expected {self.router.input_dim}"
            )
        macro_id = int(self.macro2id[str(current_macro)])
        logits_t = self.router.model(x)[0]
        probs_t = torch.sigmoid(logits_t)
        logits = [float(v) for v in logits_t.detach().cpu().tolist()]
        probs_all = [float(v) for v in probs_t.detach().cpu().tolist()]
        p_term = float(probs_all[macro_id])
        terminate = (
            bool(p_term >= self.threshold)
            if deterministic
            else bool(torch.bernoulli(probs_t[macro_id]).item())
        )
        return TerminationOutput(
            terminate=terminate,
            prob_terminate=p_term,
            current_macro_id=macro_id,
            current_macro=str(current_macro),
            current_logit=float(logits[macro_id]),
            logits=logits,
            probs_all=probs_all,
            threshold=float(self.threshold),
        )


class IntraOptionPolicyBank:
    """Lazy loader for per-macro intra-option policies."""

    def __init__(
        self,
        *,
        root_dir: str | Path,
        action_space: ActionSpace,
        checkpoint_kind: str,
        device: Optional[str] = None,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.action_space = action_space
        self.checkpoint_kind = str(checkpoint_kind)  # sft | iql | cql
        self.device = device
        self._cache: Dict[str, ActivationMLPRouter] = {}

    def _paths_for_macro(self, macro: str) -> Tuple[Path, Optional[Path]]:
        d = self.root_dir / f"micro_{safe_name(str(macro))}"
        actor = d / "actor.pt"
        if self.checkpoint_kind == "sft":
            meta = d / "label_map.json"
        else:
            meta = d / "metrics.json"
        return actor, meta

    def policy_for(self, macro: str) -> ActivationMLPRouter:
        macro = str(macro)
        if macro in self._cache:
            return self._cache[macro]
        actor, meta = self._paths_for_macro(macro)
        labels = self.action_space.micro_names_for(macro)
        pol = ActivationMLPRouter(actor_path=actor, labels=labels, meta_path=meta, device=self.device)
        if pol.micro_progress_state_dim not in (0, len(labels)):
            raise ValueError(
                f"Intra-option checkpoint {actor} expects {pol.micro_progress_state_dim} "
                f"micro-progress features, but {macro!r} has {len(labels)} micro actions."
            )
        if pol.uses_macro_history:
            if pol.history_kind != "micro_action":
                raise ValueError(
                    f"Intra-option checkpoint {actor} has history_kind={pol.history_kind!r}; "
                    "expected 'micro_action'."
                )
            if pol.macro_history_num_actions != len(labels):
                raise ValueError(
                    f"Micro-history checkpoint {actor} expects {pol.macro_history_num_actions} "
                    f"micro actions, but {macro!r} has {len(labels)}."
                )
        self._cache[macro] = pol
        return pol

SIMULATIONS_DIR = Path(__file__).resolve().parent
RUNS_DIR = SIMULATIONS_DIR / "checkpoints" / "cbt_topa"
ACTION_SPACE_PATH = SIMULATIONS_DIR / "topa_components"
CONV_STATE_DIR = RUNS_DIR / "sft_conv_state_acts_L22"

LAYER_TAG = "L22"

PolicySource = Literal["iql", "cql", "sft"]
AgentKind = Literal["policy_term_llm_micro", "policy_term_intra"]

DEFAULT_SYSTEM_PROMPT = """You are a helpful CBT therapist speaking with a client in an ongoing therapy session.

Your job is to produce only the next therapist response in the dialogue.
- Be empathic, collaborative, and clear.
- Ask clarifying questions when useful.
- Do not invent any client message.
- Do not output speaker labels.
- Continue only the immediate therapist turn."""


def _conv_tag(use_conv_state: bool) -> str:
    return "with_conv" if use_conv_state else "no_conv"


def _policy_dir(source: str, use_conv_state: bool, runs_dir: Path) -> Path:
    tag = _conv_tag(use_conv_state)
    return runs_dir / f"{source}_policy_over_options_{tag}_acts_{LAYER_TAG}"


def _termination_actor_and_meta(
    source: str,
    use_conv_state: bool,
    runs_dir: Path,
    macro_policy_dir: Optional[str | Path] = None,
) -> Tuple[Path, Path]:
    tag = _conv_tag(use_conv_state)

    if source == "sft":
        d = runs_dir / f"sft_termination_{tag}_acts_{LAYER_TAG}"
        return d / "actor.pt", d / "label_map.json"

    d = Path(macro_policy_dir) if macro_policy_dir is not None else _policy_dir(source, use_conv_state, runs_dir)
    return d / "termination.pt", d / "metrics.json"

def _macro_actor_and_meta(
    source: str,
    use_conv_state: bool,
    runs_dir: Path,
    macro_policy_dir: Optional[str | Path] = None,
) -> Tuple[Path, Path]:
    d = Path(macro_policy_dir) if macro_policy_dir is not None else _policy_dir(source, use_conv_state, runs_dir)
    if source == "sft":
        return d / "actor.pt", d / "model_meta.json"
    return d / "actor.pt", d / "metrics.json"


def _intra_dir(source: str, runs_dir: Path) -> Path:
    prefix = "sft" if source == "sft" else source
    return runs_dir / f"{prefix}_intra_option_policies_acts_{LAYER_TAG}"


def _parse_json_maybe(raw: str) -> Optional[dict]:
    if not raw:
        return None
    s = raw.strip()
    if "```" in s:
        s = s.replace("```json", "```").replace("```JSON", "```")
        parts = s.split("```")
        if len(parts) >= 2:
            s = parts[1].strip()
    try:
        return json.loads(s)
    except Exception:
        start = s.find("{")
        end = s.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(s[start : end + 1])
            except Exception:
                return None
    return None


def _truncate_at_stop(text: str, stops: List[str]) -> str:
    best = None
    for stop in stops:
        if not stop:
            continue
        idx = text.find(stop)
        if idx != -1:
            best = idx if best is None else min(best, idx)
    return text if best is None else text[:best].strip()


@dataclass
class TOPAAgent:
    """Variant definition and runtime for TOPA activation-policy simulations."""

    name: str
    source: PolicySource
    kind: AgentKind
    use_conv_state: bool
    # Used only to keep the grid at 12 variants when SFT baselines are repeated
    # under both the IQL and CQL grids.
    grid_group: str = ""
    # None means auto-detect from the saved RL checkpoint metadata.
    use_macro_action_history: Optional[bool] = None

    generator: Any = field(default=None, repr=False, compare=False)
    action_space: Optional[ActionSpace] = field(default=None, repr=False, compare=False)
    macro_policy: Optional[ActivationMLPRouter] = field(default=None, repr=False, compare=False)
    termination_policy: Optional[ActivationTerminationRouter] = field(default=None, repr=False, compare=False)
    latent_cfg: Optional[LatentConfig] = field(default=None, repr=False, compare=False)
    conv_fusion: Any = field(default=None, repr=False, compare=False)
    intra_policies: Optional[IntraOptionPolicyBank] = field(default=None, repr=False, compare=False)
    system_prompt: str = field(default=DEFAULT_SYSTEM_PROMPT, repr=False, compare=False)
    context_turns: int = field(default=5, repr=False, compare=False)
    max_macro_turns: int = field(default=5, repr=False, compare=False)
    conv_state_update_interval: int = field(default=1, repr=False, compare=False)
    macro_policy_deterministic: bool = field(default=True, repr=False, compare=False)
    termination_deterministic: bool = field(default=True, repr=False, compare=False)
    filter_macros_by_termination: bool = field(default=True, repr=False, compare=False)
    micro_policy_deterministic: bool = field(default=True, repr=False, compare=False)

    _latent_extractor: Optional[LatentExtractor] = field(default=None, init=False, repr=False, compare=False)
    _current_macro: Optional[str] = field(default=None, init=False, repr=False, compare=False)
    _macro_turns: int = field(default=0, init=False, repr=False, compare=False)
    _macro_history: List[int] = field(default_factory=list, init=False, repr=False, compare=False)
    _micro_history: List[int] = field(default_factory=list, init=False, repr=False, compare=False)
    _micro_history_macro: Optional[str] = field(default=None, init=False, repr=False, compare=False)
    _micro_visit_counts: Dict[str, Dict[int, int]] = field(default_factory=dict, init=False, repr=False, compare=False)
    _last_meta: Dict[str, Any] = field(default_factory=dict, init=False, repr=False, compare=False)

    def paths(
        self,
        *,
        runs_dir: str | Path = RUNS_DIR,
        action_space_path: str | Path = ACTION_SPACE_PATH,
        conv_state_dir: str | Path = CONV_STATE_DIR,
        macro_policy_dir: Optional[str | Path] = None,
        micro_policy_dir: Optional[str | Path] = None,
    ) -> Dict[str, str]:
        runs_dir = Path(runs_dir)
        macro_actor, macro_meta = _macro_actor_and_meta(
            self.source, self.use_conv_state, runs_dir, macro_policy_dir
        )
        term_actor, term_meta = _termination_actor_and_meta(
            self.source, self.use_conv_state, runs_dir, macro_policy_dir
        )
        paths = {
            "action_space": str(action_space_path),
            "macro_actor": str(macro_actor),
            "macro_meta": str(macro_meta),
            "termination_actor": str(term_actor),
            "termination_meta": str(term_meta),
        }
        if self.use_conv_state:
            meta = _meta_from_metrics(macro_meta)
            if str(meta.get("conv_state_source", "sft")) != "sft":
                raise ValueError("This simulation supports the recurrent SFT conversation-state model.")
            paths["conv_state_dir"] = str(conv_state_dir)
        if self.kind == "policy_term_intra":
            paths["intra_dir"] = str(
                Path(micro_policy_dir)
                if micro_policy_dir is not None
                else _intra_dir(self.source, runs_dir)
            )
        return paths

    def bind(
        self,
        *,
        generator: Any,
        action_space_path: str | Path = ACTION_SPACE_PATH,
        runs_dir: str | Path = RUNS_DIR,
        conv_state_dir: str | Path = CONV_STATE_DIR,
        macro_policy_dir: Optional[str | Path] = None,
        micro_policy_dir: Optional[str | Path] = None,
        latent_cfg: Optional[LatentConfig] = None,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        context_turns: int = 5,
        max_macro_turns: int = 5,
        conv_state_update_interval: int = 1,
        termination_threshold: float = 0.5,
        macro_policy_deterministic: bool = True,
        termination_deterministic: bool = True,
        filter_macros_by_termination: bool = True,
        micro_policy_deterministic: bool = True,
        use_macro_action_history: Optional[bool] = None,
        policy_conv_state_only: Optional[bool] = None,
        device: Optional[str] = None,
    ) -> "TOPAAgent":
        runs_dir = Path(runs_dir)
        action_space = ActionSpace(action_space_path)
        macro_actor, macro_meta = _macro_actor_and_meta(
            self.source, self.use_conv_state, runs_dir, macro_policy_dir
        )
        term_actor, term_meta = _termination_actor_and_meta(
            self.source, self.use_conv_state, runs_dir, macro_policy_dir
        )

        conv_fusion = None
        if self.use_conv_state:
            macro_config = _meta_from_metrics(macro_meta)
            if str(macro_config.get("conv_state_source", "sft")) != "sft":
                raise ValueError("This simulation supports the recurrent SFT conversation-state model.")
            conv_fusion = RecurrentSFTConvStateFusion(
                conv_state_dir,
                encoding=str(macro_config.get("conv_state_encoding", "label")),
                update_interval=conv_state_update_interval,
                device=device,
            )
        macro_policy = ActivationMLPRouter(
            actor_path=macro_actor,
            labels=action_space.macro_names(),
            meta_path=macro_meta,
            device=device,
            use_macro_action_history=use_macro_action_history,
        )
        termination_policy = ActivationTerminationRouter(
            actor_path=term_actor,
            macro_names=action_space.macro_names(),
            meta_path=term_meta,
            device=device,
            threshold=termination_threshold,
            use_macro_action_history=use_macro_action_history,
        )
        checkpoint_conv_state_only = macro_policy.policy_conv_state_only
        if checkpoint_conv_state_only != termination_policy.router.policy_conv_state_only:
            raise ValueError("Macro actor and termination checkpoint disagree about conversation-state-only mode.")
        if policy_conv_state_only is not None and checkpoint_conv_state_only != policy_conv_state_only:
            raise ValueError(
                f"SIMULATION_POLICY_CONV_STATE_ONLY={str(policy_conv_state_only).lower()} does not match "
                f"the macro checkpoint's policy_conv_state_only={str(checkpoint_conv_state_only).lower()}. "
                "Use a matching checkpoint via SIMULATION_RUNS_DIR."
            )
        if checkpoint_conv_state_only and conv_fusion is None:
            raise ValueError("Conversation-state-only checkpoint requires a conversation-state model.")
        if bool(macro_policy.uses_macro_history) != bool(termination_policy.uses_macro_history):
            raise RuntimeError(
                "Macro actor and termination checkpoint disagree about macro-action history usage."
            )
        if int(macro_policy.macro_history_len) != int(termination_policy.macro_history_len):
            raise RuntimeError(
                "Macro actor and termination checkpoint use different macro-history lengths."
            )
        planner_progress_dim = sum(len(action_space.micro_names_for(macro)) for macro in action_space.macro_names())
        if macro_policy.micro_progress_state_dim != termination_policy.router.micro_progress_state_dim:
            raise RuntimeError("Macro actor and termination checkpoint use different micro-progress dimensions.")
        if macro_policy.micro_progress_state_dim not in (0, planner_progress_dim):
            raise RuntimeError(
                f"Macro checkpoint expects {macro_policy.micro_progress_state_dim} micro-progress features, "
                f"but the action space has {planner_progress_dim} micro actions."
            )
        if macro_policy.uses_macro_history and int(macro_policy.macro_history_num_actions) != len(action_space.macro_names()):
            raise RuntimeError(
                f"Macro-history checkpoint expects {macro_policy.macro_history_num_actions} macro actions, "
                f"but the simulation action space has {len(action_space.macro_names())}."
            )
        if conv_fusion is not None and int(conv_fusion.output_dim) != int(macro_policy.conv_state_dim):
            raise RuntimeError(
                f"Conversation-state dimension mismatch: model produces {conv_fusion.output_dim}, "
                f"but macro policy expects {macro_policy.conv_state_dim}."
            )
        intra_policies = None
        if self.kind == "policy_term_intra":
            intra_policies = IntraOptionPolicyBank(
                root_dir=(
                    Path(micro_policy_dir)
                    if micro_policy_dir is not None
                    else _intra_dir(self.source, runs_dir)
                ),
                action_space=action_space,
                checkpoint_kind=self.source,
                device=device,
            )

        agent = TOPAAgent(
            name=self.name,
            source=self.source,
            kind=self.kind,
            use_conv_state=self.use_conv_state,
            grid_group=self.grid_group,
            use_macro_action_history=bool(macro_policy.uses_macro_history),
            generator=generator,
            action_space=action_space,
            macro_policy=macro_policy,
            termination_policy=termination_policy,
            latent_cfg=latent_cfg or LatentConfig(),
            conv_fusion=conv_fusion,
            intra_policies=intra_policies,
            system_prompt=system_prompt,
            context_turns=int(context_turns),
            max_macro_turns=int(max_macro_turns),
            conv_state_update_interval=int(conv_state_update_interval),
            macro_policy_deterministic=bool(macro_policy_deterministic),
            termination_deterministic=bool(termination_deterministic),
            filter_macros_by_termination=bool(filter_macros_by_termination),
            micro_policy_deterministic=bool(micro_policy_deterministic),
        )
        agent._latent_extractor = LatentExtractor(generator, agent.latent_cfg)
        return agent

    def reset_session(self) -> None:
        """Reset all state that must not leak across independent dialogues."""
        self._current_macro = None
        self._macro_turns = 0
        self._macro_history = []
        self._micro_history = []
        self._micro_history_macro = None
        self._micro_visit_counts = {}
        self._last_meta = {}
        if self.conv_fusion is not None:
            self.conv_fusion.current_state_ids = self.conv_fusion.model.initial_state_ids.to(
                self.conv_fusion.device
            ).unsqueeze(0)
            self.conv_fusion.previous_state_ids = self.conv_fusion.current_state_ids.clone()
            self.conv_fusion.last_logits = []
            self.conv_fusion.turn_index = 0
            self.conv_fusion.updated_this_turn = False

    def begin_turn(self) -> None:
        self._require_runtime()
        self._last_meta = {}
        if hasattr(self.generator, "reset_turn_stats"):
            self.generator.reset_turn_stats()

    def end_turn(self) -> Dict[str, Any]:
        self._require_runtime()
        if hasattr(self.generator, "turn_stats"):
            return dict(self.generator.turn_stats())
        return {}

    def get_last_turn_metadata(self) -> Dict[str, Any]:
        return dict(self._last_meta or {})

    def next_system_utterance(self, dialogue: List[Dict[str, Any]], session_metadata: Dict[str, Any]) -> str:
        self._require_runtime()
        if self.kind == "policy_term_llm_micro":
            return self._next_policy_term_llm_micro(dialogue)
        if self.kind == "policy_term_intra":
            return self._next_policy_term_intra(dialogue)
        raise ValueError(f"Unsupported TOPAAgent kind={self.kind!r}")

    def _require_runtime(self) -> None:
        if (
            self.generator is None
            or self.action_space is None
            or self.macro_policy is None
            or self.termination_policy is None
            or self.latent_cfg is None
            or self._latent_extractor is None
        ):
            raise RuntimeError("This TOPAAgent is a variant definition. Call build_agent() before running it.")

    def _context(self, dialogue: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if self.context_turns is None or int(self.context_turns) < 0:
            return list(dialogue)
        k = int(self.context_turns)
        return list(dialogue[-k:]) if k > 0 else []

    def _routing_context(self, dialogue: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return self._context(dialogue)

    def _generation_context(self, dialogue: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return self._context(dialogue)

    def _policy_inputs(
        self, dialogue: List[Dict[str, Any]]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, str, Optional[torch.Tensor]]:
        self._require_runtime()
        routing_text = self._latent_extractor.build_routing_text(dialogue)
        latent = self._latent_extractor.extract(routing_text)
        raw_x = latent.float()
        conv_probs: Optional[torch.Tensor] = None
        if self.conv_fusion is not None:
            base_x, conv_probs = self.conv_fusion.append_with_details(raw_x)
        else:
            base_x = raw_x

        # Planner progress covers every macro; its categorical history is appended last.
        macro_x = self._append_macro_history(base_x, conv_probs=conv_probs)
        return raw_x, base_x, macro_x, routing_text, conv_probs

    def _micro_progress_state(self, reference: torch.Tensor, macro: Optional[str] = None) -> torch.Tensor:
        if self.action_space is None:
            raise RuntimeError("Micro-progress state requires an action space")
        macros = [macro] if macro is not None else self.action_space.macro_names()
        values = [
            min(2, self._micro_visit_counts.get(name, {}).get(micro_id, 0)) / 2.0
            for name in macros
            for micro_id in range(len(self.action_space.micro_names_for(name)))
        ]
        return reference.new_tensor(values).unsqueeze(0).expand(reference.shape[0], -1)

    def _record_micro_visit(self, macro: str, action_id: int) -> None:
        counts = self._micro_visit_counts.setdefault(macro, {})
        counts[action_id] = counts.get(action_id, 0) + 1

    def _append_macro_history(
        self,
        base_x: torch.Tensor,
        *,
        conv_probs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.macro_policy is None:
            raise RuntimeError("Macro policy is not bound")

        progress = None
        if int(getattr(self.macro_policy, "micro_progress_state_dim", 0)) > 0:
            progress = self._micro_progress_state(base_x)
            raw_part = base_x[:, :-conv_probs.shape[-1]] if conv_probs is not None else base_x
            parts = [raw_part, progress]
            if conv_probs is not None:
                parts.append(conv_probs.to(base_x.device))
            base_x = torch.cat(parts, dim=-1)
        base_for_macro = _apply_policy_conv_state_only(
            base_x, conv_probs, self.macro_policy.policy_conv_state_only, progress
        )

        if not bool(self.use_macro_action_history):
            if int(base_for_macro.shape[-1]) != int(self.macro_policy.input_dim):
                raise RuntimeError(
                    f"Macro input mismatch: built {base_for_macro.shape[-1]} dims, "
                    f"checkpoint expects {self.macro_policy.input_dim}."
                )
            return base_for_macro

        history_len = int(self.macro_policy.macro_history_len)
        kept = self._macro_history[-history_len:]
        values = [-1] * history_len
        for i, action_id in enumerate(kept):
            values[i] = int(action_id)
        hist = torch.tensor(values, device=base_for_macro.device, dtype=base_for_macro.dtype)
        hist = hist.unsqueeze(0).expand(base_for_macro.shape[0], -1)
        macro_x = torch.cat([base_for_macro, hist], dim=-1)
        if int(macro_x.shape[-1]) != int(self.macro_policy.input_dim):
            raise RuntimeError(
                f"Macro-history input mismatch: built {macro_x.shape[-1]} dims, "
                f"checkpoint expects {self.macro_policy.input_dim}."
            )
        return macro_x

    def _record_current_macro_in_history(self) -> None:
        if not bool(self.use_macro_action_history):
            return
        if self.action_space is None or self.macro_policy is None or self._current_macro is None:
            return
        current_macro_id = int(self.action_space.macro_id(self._current_macro))
        if not self._macro_history or self._macro_history[-1] != current_macro_id:
            self._macro_history.append(current_macro_id)
        history_len = int(self.macro_policy.macro_history_len)
        if len(self._macro_history) > history_len:
            self._macro_history = self._macro_history[-history_len:]

    def _micro_policy_input(
        self,
        policy: ActivationMLPRouter,
        raw_x: torch.Tensor,
        augmented_x: torch.Tensor,
        *,
        new_phase: bool,
    ) -> torch.Tensor:
        current_macro = str(self._current_macro or "")
        if new_phase or self._micro_history_macro != current_macro:
            self._micro_history = []
            self._micro_history_macro = current_macro

        progress = None
        conv_x = augmented_x[:, raw_x.shape[-1]:]
        if policy.micro_progress_state_dim > 0:
            progress = self._micro_progress_state(raw_x, current_macro)
            raw_x = torch.cat([raw_x, progress], dim=-1)
            augmented_x = torch.cat([raw_x, conv_x], dim=-1)

        structured_x = None
        if policy.latent_dim == 0:
            parts = []
            if policy.micro_progress_state_dim > 0:
                parts.append(progress)
            if policy.conv_state_dim > 0:
                parts.append(conv_x)
            structured_x = torch.cat(parts, dim=-1)

        if not policy.uses_macro_history:
            if structured_x is not None:
                return self._input_for_policy(policy, structured_x, structured_x)
            return self._input_for_policy(policy, raw_x, augmented_x)
        if policy.history_kind != "micro_action":
            raise RuntimeError(
                f"Intra-option checkpoint {policy.actor_path} uses {policy.history_kind!r} history, "
                "not micro-action history."
            )

        history_len = int(policy.macro_history_len)
        base_dim = int(policy.input_dim) - history_len
        if structured_x is not None and base_dim == int(structured_x.shape[-1]):
            base_x = structured_x
        elif base_dim == int(augmented_x.shape[-1]):
            base_x = augmented_x
        elif base_dim == int(raw_x.shape[-1]):
            base_x = raw_x
        else:
            raise RuntimeError(
                f"Input dim mismatch for {policy.actor_path}: expected base={base_dim}, "
                f"but runtime has raw={raw_x.shape[-1]} and augmented={augmented_x.shape[-1]}."
            )

        kept = self._micro_history[-history_len:]
        values = [-1] * history_len
        for i, action_id in enumerate(kept):
            values[i] = int(action_id)
        history = torch.tensor(values, device=base_x.device, dtype=base_x.dtype)
        history = history.unsqueeze(0).expand(base_x.shape[0], -1)
        return torch.cat([base_x, history], dim=-1)

    def _record_micro_action(self, policy: ActivationMLPRouter, action_id: int) -> None:
        self._record_micro_visit(str(self._current_macro), int(action_id))
        if not policy.uses_macro_history:
            return
        self._micro_history.append(int(action_id))
        history_len = int(policy.macro_history_len)
        if len(self._micro_history) > history_len:
            self._micro_history = self._micro_history[-history_len:]

    @staticmethod
    def _input_for_policy(policy: Any, raw_x: torch.Tensor, augmented_x: torch.Tensor) -> torch.Tensor:
        expected_dim = int(getattr(policy, "input_dim"))
        raw_dim = int(raw_x.shape[-1])
        augmented_dim = int(augmented_x.shape[-1])
        if expected_dim == augmented_dim:
            return augmented_x
        if expected_dim == raw_dim:
            return raw_x
        actor_path = getattr(policy, "actor_path", getattr(getattr(policy, "router", None), "actor_path", "policy"))
        raise RuntimeError(
            f"Input dim mismatch for {actor_path}: expected {expected_dim}, "
            f"but runtime has raw={raw_dim} and augmented={augmented_dim}."
        )

    @staticmethod
    def _tensor_row_to_list(x: Optional[torch.Tensor]) -> Optional[List[float]]:
        if x is None:
            return None
        y = x.detach().float().cpu()
        if y.ndim > 1:
            y = y[0]
        return [float(v) for v in y.reshape(-1).tolist()]

    def _decode_conversation_state_vector(self, probs: Optional[torch.Tensor]) -> Dict[str, Any]:
        if probs is None or self.conv_fusion is None:
            return {"enabled": False, "heads": [], "dimension": 0}
        if hasattr(self.conv_fusion, "decode"):
            return self.conv_fusion.decode(probs)
        flat = self._tensor_row_to_list(probs) or []
        sizes = list(self.conv_fusion.probe.num_classes_per_dim)
        specs = list(getattr(self.action_space, "state_specs", []) or [])
        heads: List[Dict[str, Any]] = []
        offset = 0
        for head_id, size in enumerate(sizes):
            values = flat[offset : offset + int(size)]
            spec = specs[head_id] if head_id < len(specs) else {}
            name = str(spec.get("Variable Name") or spec.get("name") or f"state_head_{head_id}")
            categories = [str(v) for v in (spec.get("Categorical values") or spec.get("categories") or [])]
            if len(categories) != len(values):
                categories = [f"class_{i}" for i in range(len(values))]
            argmax_id = max(range(len(values)), key=values.__getitem__) if values else None
            heads.append({
                "head_id": head_id,
                "state_name": name,
                "offset_start": offset,
                "offset_end": offset + int(size),
                "categories": categories,
                "probabilities": values,
                "predicted_class_id": argmax_id,
                "predicted_value": categories[argmax_id] if argmax_id is not None else None,
                "confidence": values[argmax_id] if argmax_id is not None else None,
            })
            offset += int(size)
        return {
            "enabled": True,
            "dimension": len(flat),
            "expected_dimension": int(sum(sizes)),
            "schema_head_count": len(specs),
            "probe_head_count": len(sizes),
            "dimension_matches": len(flat) == int(sum(sizes)),
            "heads": heads,
        }

    @staticmethod
    def _router_debug(out: Optional[RouterOutput], labels: Sequence[str]) -> Optional[Dict[str, Any]]:
        if out is None:
            return None
        rows = []
        for idx, label in enumerate(labels):
            rows.append({
                "action_id": idx,
                "action": str(label),
                "score": float(out.scores[idx]),
                "probability": float(out.probs[idx]),
                "selected": idx == int(out.label_id),
            })
        return {
            "selected_action": out.label,
            "selected_action_id": int(out.label_id),
            "selected_probability": float(out.prob),
            "entropy": float(out.entropy),
            "top2_probability_margin": out.top2_margin,
            "actions": rows,
        }

    def _termination_debug(self, out: Optional[TerminationOutput]) -> Optional[Dict[str, Any]]:
        if out is None:
            return None
        labels = self.action_space.macro_names()
        return {
            "current_macro": out.current_macro,
            "current_macro_id": int(out.current_macro_id),
            "current_logit": float(out.current_logit),
            "current_probability": float(out.prob_terminate),
            "threshold": float(out.threshold),
            "terminate": bool(out.terminate),
            "actions": [
                {
                    "macro_id": idx,
                    "macro_action": str(label),
                    "termination_logit": float(out.logits[idx]),
                    "termination_probability": float(out.probs_all[idx]),
                    "is_current_macro": idx == int(out.current_macro_id),
                }
                for idx, label in enumerate(labels)
            ],
        }

    def _maybe_update_macro(
        self, x: torch.Tensor
    ) -> Tuple[
        bool,
        Optional[TerminationOutput],
        Optional[RouterOutput],
        RouterOutput,
        str,
        Optional[str],
        int,
    ]:
        self._require_runtime()
        previous_macro = self._current_macro
        macro_turns_before = int(self._macro_turns)
        term_out = None
        terminated = False
        termination_enabled = self.max_macro_turns != 1
        if previous_macro is not None and termination_enabled:
            term_out = self.termination_policy.predict(
                x,
                previous_macro,
                deterministic=self.termination_deterministic,
            )
            terminated = bool(term_out.terminate)

        # Evaluate the macro policy on every turn for diagnostics. We only apply
        # its argmax when the option lifecycle says to select/reselect.
        macro_eval = self.macro_policy.predict(
            x,
            deterministic=self.macro_policy_deterministic,
        )
        macro_out: Optional[RouterOutput] = None
        if previous_macro is None:
            selection_reason = "initial_selection"
        elif not termination_enabled:
            selection_reason = "termination_disabled"
        elif terminated:
            selection_reason = "termination_policy"
        elif self.max_macro_turns > 1 and self._macro_turns >= self.max_macro_turns:
            selection_reason = "max_macro_turns"
        else:
            selection_reason = "continue_current_macro"

        if selection_reason != "continue_current_macro":
            if selection_reason == "termination_policy":
                eligible_ids = [
                    action_id
                    for action_id, termination_prob in enumerate(term_out.probs_all)
                    if action_id != term_out.current_macro_id
                    and (
                        not self.filter_macros_by_termination
                        or termination_prob < term_out.threshold
                    )
                ]
                if not eligible_ids:
                    return (
                        terminated,
                        term_out,
                        None,
                        macro_eval,
                        "no_eligible_macro_farewell",
                        previous_macro,
                        macro_turns_before,
                    )
                if self.macro_policy_deterministic:
                    selected_id = max(
                        eligible_ids,
                        key=lambda action_id: macro_eval.probs[action_id],
                    )
                else:
                    eligible_probs = torch.tensor(
                        [macro_eval.probs[action_id] for action_id in eligible_ids],
                        dtype=torch.float32,
                    )
                    sampled_index = int(
                        torch.multinomial(eligible_probs, num_samples=1).item()
                    )
                    selected_id = eligible_ids[sampled_index]
                macro_out = RouterOutput(
                    label=self.action_space.macro_names()[selected_id],
                    label_id=selected_id,
                    prob=macro_eval.probs[selected_id],
                    probs=macro_eval.probs,
                    scores=macro_eval.scores,
                    entropy=macro_eval.entropy,
                    top2_margin=macro_eval.top2_margin,
                )
            else:
                macro_out = macro_eval
            self._current_macro = macro_out.label
            self._macro_turns = 0

        return (
            terminated,
            term_out,
            macro_out,
            macro_eval,
            selection_reason,
            previous_macro,
            macro_turns_before,
        )

    def _common_meta(
        self,
        *,
        raw_x: torch.Tensor,
        x: torch.Tensor,
        conv_probs: Optional[torch.Tensor],
        routing_text: str,
        terminated: bool,
        term_out: Optional[TerminationOutput],
        macro_out: Optional[RouterOutput],
        macro_eval: RouterOutput,
        selection_reason: str,
        previous_macro: Optional[str],
        macro_turns_before: int,
        micro_action: Optional[str],
        micro_out: Optional[RouterOutput],
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        meta = {
            "macro_action": self._current_macro,
            "previous_macro_action": previous_macro,
            "macro_changed": previous_macro != self._current_macro,
            "macro_selection_reason": selection_reason,
            "macro_policy_applied": macro_out is not None,
            "macro_policy_selection": (
                "argmax" if self.macro_policy_deterministic else "sample"
            ),
            "macro_turns_before_decision": int(macro_turns_before),
            "macro_turns": int(self._macro_turns),
            "max_macro_turns": int(self.max_macro_turns),
            "termination_policy_enabled": self.max_macro_turns != 1,
            "termination_policy_selection": (
                "threshold" if self.termination_deterministic else "sample"
            ),
            "filter_macros_by_termination": bool(
                self.filter_macros_by_termination
            ),
            "forced_macro_turn_limit_enabled": self.max_macro_turns > 1,
            "terminated": bool(terminated),
            "termination_prob": None if term_out is None else float(term_out.prob_terminate),
            "macro_prob": None if macro_out is None else float(macro_out.prob),
            "macro_label_id": None if macro_out is None else int(macro_out.label_id),
            "micro_action": micro_action,
            "micro_prob": None if micro_out is None else float(micro_out.prob),
            "micro_label_id": None if micro_out is None else int(micro_out.label_id),
            "conv_state_used": bool(self.conv_fusion is not None),
            "policy_conv_state_only": bool(self.macro_policy.policy_conv_state_only),
            "macro_action_history_used": bool(self.use_macro_action_history),
            "macro_action_history_length": int(getattr(self.macro_policy, "macro_history_len", 0)),
            "macro_action_history_size": int(len(self._macro_history)),
            "macro_action_history_ids": [int(v) for v in self._macro_history],
            "micro_progress_state_dim": int(getattr(self.macro_policy, "micro_progress_state_dim", 0)),
            "conversation_state_updated": bool(
                self.conv_fusion is not None and self.conv_fusion.updated_this_turn
            ),
            "conversation_state_update_interval": int(self.conv_state_update_interval),
            "policy_input_dim": int(x.shape[-1]),
            "raw_policy_input_dim": int(raw_x.shape[-1]),
            "conversation_state_dim": 0 if conv_probs is None else int(conv_probs.shape[-1]),
            "macro_policy_input_dim": int(x.shape[-1]),
            "routing_text": routing_text,
            "macro_policy": self._router_debug(macro_eval, self.action_space.macro_names()),
            "applied_macro_policy": self._router_debug(
                macro_out, self.action_space.macro_names()
            ),
            "termination_policy": self._termination_debug(term_out),
            "micro_policy": (
                self._router_debug(micro_out, self.action_space.micro_names_for(self._current_macro or ""))
                if micro_out is not None
                else None
            ),
            "conversation_state": self._decode_conversation_state_vector(conv_probs),
            # Kept separate so the simulation logger can write compact .npy files
            # and remove these large arrays from case.json.
            "tensor_payload": {
                "raw_latent": self._tensor_row_to_list(raw_x),
                "previous_conversation_state_vector": self._tensor_row_to_list(
                    self.conv_fusion.previous_state_vector()
                    if self.conv_fusion is not None and hasattr(self.conv_fusion, "previous_state_vector")
                    else None
                ),
                "conversation_state_vector": self._tensor_row_to_list(conv_probs),
                "planner_micro_progress_state": self._tensor_row_to_list(
                    self._micro_progress_state(raw_x)
                    if getattr(self.macro_policy, "micro_progress_state_dim", 0) > 0 else None
                ),
                "augmented_policy_input": self._tensor_row_to_list(x),
            },
        }
        if extra:
            meta.update(extra)
        return meta

    def _finish_with_farewell(
        self,
        *,
        raw_x: torch.Tensor,
        macro_x: torch.Tensor,
        conv_probs: Optional[torch.Tensor],
        routing_text: str,
        term_out: TerminationOutput,
        macro_eval: RouterOutput,
        previous_macro: Optional[str],
        macro_turns_before: int,
        variant_family: str,
        dialogue_context: List[Dict[str, Any]],
    ) -> str:
        farewell = "Goodbye."
        self._last_meta = self._common_meta(
            raw_x=raw_x,
            x=macro_x,
            conv_probs=conv_probs,
            routing_text=routing_text,
            terminated=True,
            term_out=term_out,
            macro_out=None,
            macro_eval=macro_eval,
            selection_reason="no_eligible_macro_farewell",
            previous_macro=previous_macro,
            macro_turns_before=macro_turns_before,
            micro_action=None,
            micro_out=None,
            extra={
                "variant_family": variant_family,
                "policy_source": self.source,
                "terminate_session": True,
                "termination_reason": "no_eligible_macro_after_filtering",
                "utterance_prompt_system": "",
                "utterance_prompt_user": "",
                "raw_output": farewell,
                "final_output": farewell,
                "dialogue_context": dialogue_context,
            },
        )
        return farewell

    def _select_micro_with_llm(self, dialogue: List[Dict[str, Any]]) -> Tuple[str, str, Optional[dict]]:
        self._require_runtime()
        convo = dialogue_to_readable_text(dialogue)
        micro_list = self.action_space.micro_list_text(self._current_macro or "")
        macro_instruction = self.action_space.instruction_text(self._current_macro or "", "")
        user = (
            "Conversation so far:\n"
            f"{convo}\n\n"
            f"Selected option:\n{macro_instruction}\n\n"
            "Available primitive actions under this option:\n"
            f"{micro_list}\n\n"
            "Pick the single best primitive action for the next therapist move that will help make more progress towards the goal of the current option.\n"
            "Return JSON only with key: primitive_action"
        )
        raw = self.generator.generate(system=self.system_prompt, user=user).strip()
        parsed = _parse_json_maybe(raw)
        chosen = None
        if isinstance(parsed, dict):
            chosen = parsed.get("primitive_action") or parsed.get("micro_action") or parsed.get("action")
        chosen = str(chosen).strip() if chosen else ""
        allowed = set(self.action_space.micro_names_for(self._current_macro or ""))
        if chosen not in allowed:
            chosen = self.action_space.micro_names_for(self._current_macro or "")[0]
        return chosen, raw, parsed

    def _next_policy_term_llm_micro(self, dialogue: List[Dict[str, Any]]) -> str:
        routing_ctx = self._routing_context(dialogue)
        generation_ctx = self._generation_context(dialogue)
        raw_x, base_x, macro_x, routing_text, conv_probs = self._policy_inputs(routing_ctx)
        (
            terminated, term_out, macro_out, macro_eval, selection_reason,
            previous_macro, macro_turns_before,
        ) = self._maybe_update_macro(macro_x)
        if selection_reason == "no_eligible_macro_farewell":
            return self._finish_with_farewell(
                raw_x=raw_x,
                macro_x=macro_x,
                conv_probs=conv_probs,
                routing_text=routing_text,
                term_out=term_out,
                macro_eval=macro_eval,
                previous_macro=previous_macro,
                macro_turns_before=macro_turns_before,
                variant_family="policy_term_then_llm_micro",
                dialogue_context=generation_ctx,
            )

        micro, micro_raw, micro_parsed = self._select_micro_with_llm(generation_ctx)
        instruction = self.action_space.instruction_text(self._current_macro or "", micro)
        convo = dialogue_to_readable_text(generation_ctx)
        user = (
            "Conversation so far:\n"
            f"{convo}\n\n"
            "Follow the instruction below to produce the next therapist message.\n"
            f"{instruction}\n\n"
            "Write the therapist's NEXT message only."
        )
        raw = self.generator.generate(system=self.system_prompt, user=user).strip()
        final = _truncate_at_stop(raw, ["\nUser:", "\nPatient:", "\nTherapist:"])

        self._macro_turns += 1
        self._last_meta = self._common_meta(
            raw_x=raw_x,
            x=macro_x,
            conv_probs=conv_probs,
            routing_text=routing_text,
            terminated=terminated,
            term_out=term_out,
            macro_out=macro_out,
            macro_eval=macro_eval,
            selection_reason=selection_reason,
            previous_macro=previous_macro,
            macro_turns_before=macro_turns_before,
            micro_action=micro,
            micro_out=None,
            extra={
                "variant_family": "policy_term_then_llm_micro",
                "policy_source": self.source,
                "llm_calls_after_routing": 2,
                "instruction": instruction,
                "micro_selection_mode": "llm",
                "micro_candidate_actions": self.action_space.micro_names_for(self._current_macro or ""),
                "micro_prompt_system": self.system_prompt,
                "micro_prompt_user": (
                    "Conversation so far:\n"
                    f"{convo}\n\n"
                    f"Selected option:\n{self.action_space.instruction_text(self._current_macro or '', '')}\n\n"
                    "Available primitive actions under this option:\n"
                    f"{self.action_space.micro_list_text(self._current_macro or '')}\n\n"
                    "Pick the single best primitive action for the next therapist move that will help make more progress towards the goal of the current option.\n"
                    "Return JSON only with key: primitive_action"
                ),
                "micro_prompt_raw_output": micro_raw,
                "micro_prompt_parsed_output": micro_parsed,
                "utterance_prompt_system": self.system_prompt,
                "utterance_prompt_user": user,
                "raw_output": raw,
                "final_output": final,
                "dialogue_context": generation_ctx,
            },
        )
        self._record_micro_visit(
            str(self._current_macro),
            self.action_space.micro_names_for(self._current_macro or "").index(micro),
        )
        self._record_current_macro_in_history()
        return final

    def _next_policy_term_intra(self, dialogue: List[Dict[str, Any]]) -> str:
        if self.intra_policies is None:
            raise RuntimeError(f"TOPAAgent variant {self.name} requires intra-option policies.")
        routing_ctx = self._routing_context(dialogue)
        generation_ctx = self._generation_context(dialogue)
        raw_x, base_x, macro_x, routing_text, conv_probs = self._policy_inputs(routing_ctx)
        (
            terminated, term_out, macro_out, macro_eval, selection_reason,
            previous_macro, macro_turns_before,
        ) = self._maybe_update_macro(macro_x)
        if selection_reason == "no_eligible_macro_farewell":
            return self._finish_with_farewell(
                raw_x=raw_x,
                macro_x=macro_x,
                conv_probs=conv_probs,
                routing_text=routing_text,
                term_out=term_out,
                macro_eval=macro_eval,
                previous_macro=previous_macro,
                macro_turns_before=macro_turns_before,
                variant_family="policy_term_intra",
                dialogue_context=generation_ctx,
            )

        micro_policy = self.intra_policies.policy_for(self._current_macro or "")
        micro_x = self._micro_policy_input(
            micro_policy,
            raw_x,
            base_x,
            new_phase=macro_out is not None,
        )
        micro_history_input_ids = list(self._micro_history)
        micro_out = micro_policy.predict(
            micro_x,
            deterministic=self.micro_policy_deterministic,
        )
        micro = micro_out.label
        instruction = self.action_space.instruction_text(self._current_macro or "", micro)
        convo = dialogue_to_readable_text(generation_ctx)
        phase_note = f"Current macro phase: {self._current_macro}."
        user = (
            "Conversation so far:\n"
            f"{convo}\n\n"
            f"{phase_note}\n"
            "Follow the combined macro + primitive-action instruction below.\n"
            f"{instruction}\n\n"
            "Write the therapist's NEXT message only."
        )
        raw = self.generator.generate(system=self.system_prompt, user=user).strip()
        final = _truncate_at_stop(raw, ["\nUser:", "\nPatient:", "\nTherapist:"])

        self._macro_turns += 1
        self._last_meta = self._common_meta(
            raw_x=raw_x,
            x=macro_x,
            conv_probs=conv_probs,
            routing_text=routing_text,
            terminated=terminated,
            term_out=term_out,
            macro_out=macro_out,
            macro_eval=macro_eval,
            selection_reason=selection_reason,
            previous_macro=previous_macro,
            macro_turns_before=macro_turns_before,
            micro_action=micro,
            micro_out=micro_out,
            extra={
                "variant_family": "policy_term_intra",
                "policy_source": self.source,
                "llm_calls_after_routing": 1,
                "instruction": instruction,
                "utterance_prompt_system": self.system_prompt,
                "utterance_prompt_user": user,
                "raw_output": raw,
                "final_output": final,
                "dialogue_context": generation_ctx,
                "augmented_policy_input_dim": int(macro_x.shape[-1]),
                "base_augmented_policy_input_dim": int(base_x.shape[-1]),
                "micro_policy_input_dim": int(micro_x.shape[-1]),
                "micro_policy_expected_dim": int(micro_policy.input_dim),
                "micro_policy_selection": (
                    "argmax" if self.micro_policy_deterministic else "sample"
                ),
                "micro_action_history_used": bool(micro_policy.uses_macro_history),
                "micro_action_history_kind": str(micro_policy.history_kind),
                "micro_action_history_length": int(micro_policy.macro_history_len),
                "micro_action_history_size": int(len(micro_history_input_ids)),
                "micro_action_history_ids": [int(v) for v in micro_history_input_ids],
                "micro_progress_state": self._tensor_row_to_list(
                    self._micro_progress_state(raw_x, self._current_macro)
                    if micro_policy.micro_progress_state_dim > 0 else None
                ),
                "micro_conv_state_used": bool(
                    self.conv_fusion is not None
                    and int(micro_policy.conv_state_dim) > 0
                ),
            },
        )
        self._record_micro_action(micro_policy, micro_out.label_id)
        self._record_current_macro_in_history()
        return final


SIMULATION_VARIANTS: Dict[str, TOPAAgent] = {
    # IQL grid: 6 variants.
    "iql_policy_term_llm_micro_no_conv": TOPAAgent("iql_policy_term_llm_micro_no_conv", "iql", "policy_term_llm_micro", False, "iql"),
    "iql_policy_term_llm_micro_with_conv": TOPAAgent("iql_policy_term_llm_micro_with_conv", "iql", "policy_term_llm_micro", True, "iql"),
    "iql_policy_term_intra_no_conv": TOPAAgent("iql_policy_term_intra_no_conv", "iql", "policy_term_intra", False, "iql"),
    "iql_policy_term_intra_with_conv": TOPAAgent("iql_policy_term_intra_with_conv", "iql", "policy_term_intra", True, "iql"),
    "sft_policy_term_intra_no_conv": TOPAAgent("sft_policy_term_intra_no_conv", "sft", "policy_term_intra", False, "iql"),
    "sft_policy_term_intra_with_conv": TOPAAgent("sft_policy_term_intra_with_conv", "sft", "policy_term_intra", True, "iql"),
    # CQL grid: same layout as IQL. The two SFT entries are aliases so the
    # comparison table can remain 6-vs-6 = 12 rows.
    "cql_policy_term_llm_micro_no_conv": TOPAAgent("cql_policy_term_llm_micro_no_conv", "cql", "policy_term_llm_micro", False, "cql"),
    "cql_policy_term_llm_micro_with_conv": TOPAAgent("cql_policy_term_llm_micro_with_conv", "cql", "policy_term_llm_micro", True, "cql"),
    "cql_policy_term_intra_no_conv": TOPAAgent("cql_policy_term_intra_no_conv", "cql", "policy_term_intra", False, "cql"),
    "cql_policy_term_intra_with_conv": TOPAAgent("cql_policy_term_intra_with_conv", "cql", "policy_term_intra", True, "cql"),
    "sft_policy_term_intra_no_conv__cql_grid": TOPAAgent("sft_policy_term_intra_no_conv__cql_grid", "sft", "policy_term_intra", False, "cql"),
    "sft_policy_term_intra_with_conv__cql_grid": TOPAAgent("sft_policy_term_intra_with_conv__cql_grid", "sft", "policy_term_intra", True, "cql"),
}


def list_variants() -> List[str]:
    return list(SIMULATION_VARIANTS.keys())


def variant_paths(
    variant_name: str,
    *,
    runs_dir: str | Path = RUNS_DIR,
    action_space_path: str | Path = ACTION_SPACE_PATH,
    conv_state_dir: str | Path = CONV_STATE_DIR,
    macro_policy_dir: Optional[str | Path] = None,
    micro_policy_dir: Optional[str | Path] = None,
) -> Dict[str, str]:
    if variant_name not in SIMULATION_VARIANTS:
        raise KeyError(f"Unknown variant={variant_name}. Available: {', '.join(list_variants())}")
    return SIMULATION_VARIANTS[variant_name].paths(
        runs_dir=runs_dir,
        action_space_path=action_space_path,
        conv_state_dir=conv_state_dir,
        macro_policy_dir=macro_policy_dir,
        micro_policy_dir=micro_policy_dir,
    )


def build_agent(
    *,
    variant_name: str,
    generator: Any,
    action_space_path: str | Path = ACTION_SPACE_PATH,
    runs_dir: str | Path = RUNS_DIR,
    conv_state_dir: str | Path = CONV_STATE_DIR,
    macro_policy_dir: Optional[str | Path] = None,
    micro_policy_dir: Optional[str | Path] = None,
    latent_cfg: Optional[LatentConfig] = None,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    context_turns: int = 5,
    max_macro_turns: int = 5,
    conv_state_update_interval: int = 1,
    termination_threshold: float = 0.5,
    macro_policy_deterministic: bool = True,
    termination_deterministic: bool = True,
    filter_macros_by_termination: bool = True,
    micro_policy_deterministic: bool = True,
    use_macro_action_history: Optional[bool] = None,
    policy_conv_state_only: Optional[bool] = None,
    device: Optional[str] = None,
) -> TOPAAgent:
    """Build one runtime-bound TOPAAgent from an explicit variant name."""
    if variant_name not in SIMULATION_VARIANTS:
        raise KeyError(f"Unknown variant={variant_name}. Available: {', '.join(list_variants())}")
    return SIMULATION_VARIANTS[variant_name].bind(
        generator=generator,
        action_space_path=action_space_path,
        runs_dir=runs_dir,
        conv_state_dir=conv_state_dir,
        macro_policy_dir=macro_policy_dir,
        micro_policy_dir=micro_policy_dir,
        latent_cfg=latent_cfg,
        system_prompt=system_prompt,
        context_turns=context_turns,
        max_macro_turns=max_macro_turns,
        conv_state_update_interval=conv_state_update_interval,
        termination_threshold=termination_threshold,
        macro_policy_deterministic=macro_policy_deterministic,
        termination_deterministic=termination_deterministic,
        filter_macros_by_termination=filter_macros_by_termination,
        micro_policy_deterministic=micro_policy_deterministic,
        use_macro_action_history=use_macro_action_history,
        policy_conv_state_only=policy_conv_state_only,
        device=device,
    )
