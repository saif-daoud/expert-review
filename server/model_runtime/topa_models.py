import json
from pathlib import Path
from typing import List

import torch
from torch import nn
import torch.nn.functional as F


class _ActionHistoryMixin:
    """Categorical action-history branch shared with the RL training code."""

    def _init_action_history(
        self,
        *,
        action_history_len: int,
        action_history_num_actions: int,
        action_history_embedding_dim: int,
        action_history_projection_dim: int,
        dropout: float,
    ) -> None:
        self.action_history_len = max(0, int(action_history_len))
        self.action_history_num_actions = max(0, int(action_history_num_actions))
        self.action_history_embedding_dim = max(0, int(action_history_embedding_dim))
        self.action_history_projection_dim = max(0, int(action_history_projection_dim))

        if self.action_history_len > 0:
            if self.action_history_num_actions <= 0:
                raise ValueError("action_history_num_actions must be > 0 when history is enabled")
            if self.action_history_embedding_dim <= 0:
                raise ValueError("action_history_embedding_dim must be > 0 when history is enabled")
            if self.action_history_projection_dim <= 0:
                raise ValueError("action_history_projection_dim must be > 0 when history is enabled")
            self.action_history_embedding = nn.Embedding(
                self.action_history_num_actions + 1,
                self.action_history_embedding_dim,
                padding_idx=0,
            )
            self.action_history_proj = nn.Sequential(
                nn.Linear(
                    self.action_history_len * self.action_history_embedding_dim,
                    self.action_history_projection_dim,
                ),
                nn.ReLU(),
                nn.Dropout(float(dropout)),
            )
        else:
            self.action_history_embedding = None
            self.action_history_proj = None

    def _split_and_encode_action_history(self, input_embeds: torch.Tensor):
        if self.action_history_len <= 0:
            return input_embeds, None
        if input_embeds.shape[-1] < self.action_history_len:
            raise ValueError(
                f"Input width {input_embeds.shape[-1]} is smaller than "
                f"action_history_len={self.action_history_len}."
            )
        base = input_embeds[:, :-self.action_history_len]
        raw_history = input_embeds[:, -self.action_history_len:]
        history_ids = torch.round(raw_history).long() + 1
        history_ids = history_ids.clamp(0, self.action_history_num_actions)
        hist = self.action_history_embedding(history_ids)
        hist = hist.reshape(hist.shape[0], -1)
        hist = self.action_history_proj(hist)
        return base, hist


class MLPPolicy(_ActionHistoryMixin, nn.Module):
    """Discrete policy/Q head identical to ``rl.simple_models.MLPPolicy``."""

    def __init__(
        self,
        in_dim: int,
        num_actions: int,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        conv_state_dim: int = 0,
        latent_projection_dim: int = 0,
        conv_state_projection_dim: int = 0,
        micro_progress_state_dim: int = 0,
        micro_progress_state_projection_dim: int = 0,
        action_history_len: int = 0,
        action_history_num_actions: int = 0,
        action_history_embedding_dim: int = 0,
        action_history_projection_dim: int = 0,
    ):
        nn.Module.__init__(self)
        self.in_dim = int(in_dim)
        self.hidden_dim = int(hidden_dim)
        self.conv_state_dim = max(0, int(conv_state_dim))
        self.micro_progress_state_dim = max(0, int(micro_progress_state_dim))

        self._init_action_history(
            action_history_len=action_history_len,
            action_history_num_actions=action_history_num_actions,
            action_history_embedding_dim=action_history_embedding_dim,
            action_history_projection_dim=action_history_projection_dim,
            dropout=dropout,
        )

        self.base_in_dim = self.in_dim - self.action_history_len
        structured_dim = self.micro_progress_state_dim + self.conv_state_dim
        if self.base_in_dim < structured_dim:
            raise ValueError(
                f"Base input width {self.base_in_dim} is smaller than structured width {structured_dim}."
            )
        self.latent_dim = self.base_in_dim - structured_dim

        self.latent_projection_dim = max(0, int(latent_projection_dim))
        self.conv_state_projection_dim = max(0, int(conv_state_projection_dim))
        self.micro_progress_state_projection_dim = max(
            0, int(micro_progress_state_projection_dim)
        )

        self.latent_proj = None
        if self.latent_dim > 0 and self.latent_projection_dim > 0:
            self.latent_proj = nn.Sequential(
                nn.Linear(self.latent_dim, self.latent_projection_dim),
                nn.ReLU(),
                nn.Dropout(float(dropout)),
            )

        self.micro_progress_state_proj = None
        if self.micro_progress_state_dim > 0 and self.micro_progress_state_projection_dim > 0:
            self.micro_progress_state_proj = nn.Sequential(
                nn.Linear(
                    self.micro_progress_state_dim,
                    self.micro_progress_state_projection_dim,
                ),
                nn.ReLU(),
                nn.Dropout(float(dropout)),
            )

        self.conv_state_proj = None
        if self.conv_state_dim > 0 and self.conv_state_projection_dim > 0:
            self.conv_state_proj = nn.Linear(
                self.conv_state_dim,
                self.conv_state_projection_dim,
            )

        latent_out = self.latent_projection_dim if self.latent_proj is not None else self.latent_dim
        progress_out = (
            self.micro_progress_state_projection_dim
            if self.micro_progress_state_proj is not None
            else self.micro_progress_state_dim
        )
        conv_out = (
            self.conv_state_projection_dim
            if self.conv_state_proj is not None
            else self.conv_state_dim
        )
        history_out = self.action_history_projection_dim if self.action_history_len > 0 else 0
        net_in_dim = int(latent_out + progress_out + conv_out + history_out)

        if int(hidden_dim) > 0:
            self.net = nn.Sequential(
                nn.Linear(net_in_dim, int(hidden_dim)),
                nn.ReLU(),
                nn.Dropout(float(dropout)),
                nn.Linear(int(hidden_dim), int(num_actions)),
            )
        else:
            self.net = nn.Linear(net_in_dim, int(num_actions))

    def forward(self, input_embeds: torch.Tensor) -> torch.Tensor:
        input_embeds, history = self._split_and_encode_action_history(input_embeds)
        latent_end = self.latent_dim
        progress_end = latent_end + self.micro_progress_state_dim
        z = input_embeds[:, :latent_end]
        p = input_embeds[:, latent_end:progress_end]
        c = input_embeds[:, progress_end:]
        parts = []
        if self.latent_dim > 0:
            parts.append(self.latent_proj(z) if self.latent_proj is not None else z)
        if self.micro_progress_state_dim > 0:
            parts.append(
                self.micro_progress_state_proj(p)
                if self.micro_progress_state_proj is not None
                else p
            )
        if self.conv_state_dim > 0:
            parts.append(self.conv_state_proj(c) if self.conv_state_proj is not None else c)
        if history is not None:
            parts.append(history)
        if not parts:
            raise ValueError("No active state branch reached the model.")
        return self.net(torch.cat(parts, dim=-1))



class RecurrentSFTConvStateProbe(nn.Module):
    """Conversation-state model used by the saved CBT checkpoint."""

    def __init__(self, model_dir: str | Path) -> None:
        super().__init__()
        self.model_dir = Path(model_dir)
        meta = json.loads((self.model_dir / "label_map.json").read_text(encoding="utf-8"))
        self.dim_names = list(meta["dim_names"])
        self.dim_to_label2id = meta["dim_to_label2id"]
        self.dim_num_labels = [
            int(value)
            for value in meta.get(
                "dim_num_labels",
                [len(self.dim_to_label2id[name]) for name in self.dim_names],
            )
        ]
        self.activation_dim = int(meta["activation_dim"])
        self.prev_state_dim = int(meta["prev_state_dim"])
        state_projection_dim = int(meta.get("state_projection_dim", 0))
        hidden_dim = int(meta.get("hidden_dim", 0))

        if state_projection_dim:
            self.state_projection = nn.Sequential(
                nn.Linear(self.prev_state_dim, state_projection_dim),
                nn.ReLU(),
            )
        else:
            self.state_projection = nn.Identity()
        fused_dim = self.activation_dim + (state_projection_dim or self.prev_state_dim)

        if hidden_dim:
            self.trunk = nn.Sequential(nn.Linear(fused_dim, hidden_dim), nn.ReLU())
            head_dim = hidden_dim
        else:
            self.trunk = nn.Identity()
            head_dim = fused_dim
        self.heads = nn.ModuleList(
            [nn.Linear(head_dim, num_labels) for num_labels in self.dim_num_labels]
        )

        state = torch.load(self.model_dir / "actor.pt", map_location="cpu", weights_only=False)
        state = state.get("model_state_dict", state)
        state = {
            key.replace("module.", ""): value.detach().float().cpu()
            for key, value in state.items()
            if isinstance(value, torch.Tensor)
        }
        self.load_state_dict(state, strict=True)
        self.initial_state_ids = torch.tensor(
            [int(self.dim_to_label2id[name]["none"]) for name in self.dim_names],
            dtype=torch.long,
        )
        self.one_hot_dim = sum(self.dim_num_labels)
        self.requires_grad_(False)
        self.eval()

    def _state_ids_to_onehot(self, state_ids: torch.Tensor) -> torch.Tensor:
        if state_ids.ndim == 1:
            state_ids = state_ids.unsqueeze(0)
        return torch.cat(
            [
                F.one_hot(state_ids[:, index].long(), num_classes=num_labels).float()
                for index, num_labels in enumerate(self.dim_num_labels)
            ],
            dim=-1,
        )

    @torch.no_grad()
    def predict_next_logits(
        self,
        activations: torch.Tensor,
        previous_state_ids: torch.Tensor,
    ) -> List[torch.Tensor]:
        if activations.ndim == 1:
            activations = activations.unsqueeze(0)
        previous_state = self._state_ids_to_onehot(previous_state_ids).to(
            device=activations.device,
            dtype=activations.dtype,
        )
        state_features = self.state_projection(previous_state)
        hidden = self.trunk(torch.cat([activations, state_features], dim=-1))
        return [head(hidden) for head in self.heads]
