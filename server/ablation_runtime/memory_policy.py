"""Deploy a saved type-mask retriever without importing TOPA's training stack."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .topa_models import MLPPolicy

MODES = ("legacy", "zero", "full_kg", "retrieved")


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def schema_types(schema):
    names = [node["type"] for node in schema["nodes"] if node["type"] != "Patient"]
    if len(names) != len(set(names)):
        raise ValueError("KG schema has duplicate node types")
    return names


def available_graph_types(graph):
    """Types with usable facts in the current completed-phase KG snapshot."""
    if graph is None:
        return []
    return sorted({node["type"] for node in graph["nodes"]
                   if node["type"] != "Patient" and node.get("status") not in {"inactive", "retracted"}
                   and str(node.get("content", "")).strip()})


def filter_graph_types(graph, requested_types):
    """Induced active subgraph of predicted types, including the Patient root.

    No unrequested non-root node content is included, even through an edge.
    An empty prediction or no matching facts yields an empty graph.
    """
    requested = set(requested_types)
    active = [node for node in graph["nodes"] if node.get("status") not in {"inactive", "retracted"}]
    available = {node["type"] for node in active if node["type"] != "Patient"}
    matched = [node for node in active if node["type"] in requested]
    nodes = matched + [node for node in active if node["type"] == "Patient"] if matched else []
    ids = {node["id"] for node in nodes}
    edges = [edge for edge in graph["edges"] if edge["source"] in ids and edge["target"] in ids
             and edge.get("status") not in {"inactive", "retracted"}]
    return {**graph, "nodes": nodes, "edges": edges}, sorted(requested - available)


class MemorySelector:
    def __init__(self, mode, option_dir, retriever_dir="", device="cpu", *, deterministic=True, available_only=True, top_k=1):
        if mode not in MODES or mode == "legacy":
            raise ValueError("MemorySelector requires zero, full_kg or retrieved mode")
        if not isinstance(deterministic, bool):
            raise ValueError("Memory retrieval deterministic must be a boolean")
        if not isinstance(available_only, bool):
            raise ValueError("Memory retrieval available_only must be a boolean")
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
            raise ValueError("Memory retrieval top_k must be a positive integer")
        if top_k > 1 and not available_only:
            raise ValueError("Memory retrieval top_k > 1 requires available_only=true")
        self.mode = mode
        self.deterministic = deterministic
        self.available_only = available_only
        self.top_k = top_k
        self.option_dir = Path(option_dir)
        self.meta = json.loads((self.option_dir / "model_meta.json").read_text(encoding="utf-8"))
        self.width = int(self.meta.get("memory_state_dim", 0))
        self.types = self.meta.get("memory_usage_node_types", [])
        self.start = sum(int(self.meta.get(key, 0)) for key in
                         ("activation_state_dim", "micro_progress_state_dim", "conv_state_dim"))
        self.input_dim = int(self.meta["input_dim"])
        expected = self.start + self.width + int(self.meta.get("history_length", 0))
        if self.meta.get("task") != "policy_over_options" or not self.width or len(self.types) != self.width or expected != self.input_dim:
            raise ValueError("Macro checkpoint must have a valid explicit memory block and ordered node types")
        if len(set(self.types)) != self.width:
            raise ValueError("Macro checkpoint memory node types are not unique")
        self.device = torch.device(device)
        self.model = None
        if mode == "retrieved":
            directory = Path(retriever_dir)
            run = json.loads((directory / "run_config.json").read_text(encoding="utf-8"))
            for name, expected_hash in run["frozen_hashes"].items():
                if file_hash(self.option_dir / name) != expected_hash:
                    raise ValueError("Retriever was trained with a different option checkpoint: " + name)
            if run["option_meta"].get("memory_usage_node_types") != self.types:
                raise ValueError("Retriever memory type order differs from the option checkpoint")
            if file_hash(directory / "candidate_masks.npy") != run["candidate_masks_sha256"]:
                raise ValueError("Retriever candidate masks changed since training")
            candidates = np.load(directory / "candidate_masks.npy", allow_pickle=False)
            if candidates.ndim != 2 or candidates.shape[1] != self.width or not len(candidates) or not np.isin(candidates, [0, 1]).all():
                raise ValueError("Retriever candidate masks must be binary with the exact memory width")
            self.candidates = torch.as_tensor(candidates, dtype=torch.float32, device=self.device)
            meta = self.meta
            self.model = MLPPolicy(
                in_dim=self.input_dim - self.width, num_actions=self.width,
                hidden_dim=int(run["hidden_dim"]), dropout=float(run["dropout"]),
                conv_state_dim=int(meta.get("conv_state_dim", 0)),
                latent_projection_dim=int(meta.get("latent_projection_dim", 0)),
                conv_state_projection_dim=int(meta.get("conv_state_projection_dim", 0)),
                micro_progress_state_dim=int(meta.get("micro_progress_state_dim", 0)),
                micro_progress_state_projection_dim=int(meta.get("progress_projection_dim", 0)),
                action_history_len=int(meta.get("history_length", 0)),
                action_history_num_actions=int(meta.get("history_num_actions", 0)),
                action_history_embedding_dim=int(meta.get("history_embedding_dim", 0)),
                action_history_projection_dim=int(meta.get("history_projection_dim", 0)),
                memory_state_dim=0)
            self.model.load_state_dict(torch.load(directory / "retriever.pt", map_location="cpu"), strict=True)
            self.model.to(self.device).eval().requires_grad_(False)

    @torch.no_grad()
    def predict(self, zero_memory_state, *, available_node_types=None):
        if zero_memory_state.ndim != 2 or zero_memory_state.shape[1] != self.input_dim:
            raise ValueError("Macro state shape differs from the retriever's training state")
        if not torch.isfinite(zero_memory_state).all():
            raise ValueError("Non-finite macro state")
        if torch.count_nonzero(zero_memory_state[:, self.start:self.start + self.width]):
            raise ValueError("Retriever input must start with a zero placeholder, never a GT/carried-over mask")
        details = dict(mode=self.mode, node_types=[], candidate_index=None, deterministic=self.deterministic,
                       available_only=self.available_only, requested_top_k=self.top_k)
        if self.model is None:
            mask = zero_memory_state.new_zeros((len(zero_memory_state), self.width))
            details["selection_method"] = "zero"
        else:
            if self.available_only and available_node_types is None:
                raise ValueError("Availability-constrained retrieval requires available_node_types; use [] for an empty KG")
            available_types = set(available_node_types or [])
            available = torch.tensor([name in available_types for name in self.types], device=self.device, dtype=torch.bool)
            details["available_node_types"] = [name for name in self.types if name in available_types]
            states = torch.cat((zero_memory_state[:, :self.start],
                                zero_memory_state[:, self.start + self.width:]), dim=-1).to(self.device).float()
            logits = self.model(states)
            if not torch.isfinite(logits).all():
                raise ValueError("Non-finite retriever output")
            probabilities = logits.sigmoid()
            details["type_probabilities"] = probabilities[0].tolist()
            if self.available_only:
                mask = torch.zeros_like(probabilities)
                if available.any():
                    # Normalize sigmoid probabilities over usable types. Log space
                    # preserves their ratios even if all sigmoid values underflow.
                    log_weights = torch.nn.functional.logsigmoid(logits).masked_fill(~available, -torch.inf)
                    effective_probabilities = log_weights.softmax(dim=-1)
                    count = min(self.top_k, int(available.sum()))
                    if self.deterministic:
                        index = (effective_probabilities.argmax(dim=-1, keepdim=True) if count == 1
                                 else log_weights.topk(count, dim=-1).indices)
                        details["selection_method"] = "available_type_argmax" if count == 1 else "available_types_top_k"
                        mask.scatter_(1, index, 1.)
                    else:
                        # Recompute normalized probabilities after each draw.
                        # This preserves sigmoid-weighted sampling without
                        # replacement even when tiny probabilities underflow.
                        remaining = log_weights.clone()
                        for draw in range(count):
                            probabilities_for_draw = effective_probabilities if draw == 0 else remaining.softmax(dim=-1)
                            index = torch.multinomial(probabilities_for_draw, num_samples=1)
                            mask.scatter_(1, index, 1.)
                            remaining.scatter_(1, index, -torch.inf)
                        details["selection_method"] = ("available_type_sample" if count == 1
                                                       else "available_types_sample_without_replacement")
                else:
                    effective_probabilities = torch.zeros_like(probabilities)
                    details["selection_method"] = "empty_kg"
            elif self.deterministic:
                # Original training-validation decoding when availability is disabled.
                effective_probabilities = probabilities
                index = (logits @ self.candidates.T).argmax(dim=-1)
                mask = self.candidates[index]
                details["candidate_index"] = int(index[0])
                details["selection_method"] = "supported_mask_map"
            else:
                # Independent type draws: multi-hot, including NONE and unseen combinations.
                effective_probabilities = probabilities
                mask = torch.bernoulli(effective_probabilities)
                details["selection_method"] = "independent_bernoulli"
            details["effective_type_probabilities"] = effective_probabilities[0].tolist()
            mask = mask.to(zero_memory_state.device)
            details["node_types"] = [name for name, selected in zip(self.types, mask[0].tolist()) if selected]
        details["vector"] = mask[0].tolist()
        details["selected_count"] = int(mask[0].sum())
        return mask, details
