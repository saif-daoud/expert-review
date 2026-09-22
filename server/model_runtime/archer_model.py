"""OfflineArcher actor model required for inference only."""

from __future__ import annotations

import torch


TWENTY_QUESTIONS_POLICY_TEMPLATE = """You are playing Twenty Questions. Ask exactly one short yes/no question that helps identify the hidden object.
Reply with only the next question and nothing else.

Conversation so far:
{obs}

Next question:"""

class GPT2(torch.nn.Module):
    def __init__(
        self,
        get_device,
        from_checkpoint=None,
        model_name: str = "gpt2",
        prompt_style: str = "auto",
        max_input_length: int = 512,
        max_new_tokens: int = 32,
        torch_dtype: str = "auto",
        use_lora: bool = False,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        gradient_checkpointing: bool = False,
    ):
        super().__init__()

        self.get_device = get_device
        self.model_name = model_name
        self.prompt_style = prompt_style
        self.max_input_length = max_input_length
        self.max_new_tokens = max_new_tokens
        self.use_lora = use_lora
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.gradient_checkpointing = gradient_checkpointing

        ### Initialize Model
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_kwargs = {"trust_remote_code": True}
        resolved_dtype = self._resolve_torch_dtype(torch_dtype)
        if resolved_dtype is not None:
            model_kwargs["torch_dtype"] = resolved_dtype

        self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.tokenizer.truncation_side = 'left'
        self.tokenizer.padding_side = 'left'
        if self.tokenizer.pad_token is None:
            if self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                self.tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
                self.model.resize_token_embeddings(len(self.tokenizer))
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        if getattr(self.model.config, "pad_token_id", None) is None:
            self.model.config.pad_token_id = self.tokenizer.pad_token_id

        self.max_sequence_tokens = self._resolve_max_sequence_tokens()

        self.use_chat_template = self._infer_use_chat_template(prompt_style=prompt_style)

        if self.gradient_checkpointing and hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable()
            if hasattr(self.model, "enable_input_require_grads"):
                self.model.enable_input_require_grads()
            if hasattr(self.model.config, "use_cache"):
                self.model.config.use_cache = False

        if self.use_lora:
            from peft import LoraConfig, TaskType, get_peft_model

            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules="all-linear",
            )
            self.model = get_peft_model(self.model, peft_config)
            try:
                self.model.print_trainable_parameters()
            except Exception:
                pass

        if from_checkpoint is not None:
            checkpoint = torch.load(from_checkpoint, map_location=torch.device('cpu'))
            valid_prefixes = ("agent.", "actor.")
            weights = {}
            for k, v in checkpoint["state_dict"].items():
                for prefix in valid_prefixes:
                    if k.startswith(prefix):
                        weights[k.removeprefix(prefix)] = v
                        break
            missing, unexpected = self.load_state_dict(weights, strict=False)
            print("I have initialized the actor from the checkpoint: ", from_checkpoint)
            if missing:
                print("Missing actor keys while loading checkpoint:", missing)
            if unexpected:
                print("Unexpected actor keys while loading checkpoint:", unexpected)

    def _resolve_torch_dtype(self, torch_dtype):
        if torch_dtype in (None, "auto"):
            return None
        if isinstance(torch_dtype, torch.dtype):
            return torch_dtype
        mapping = {
            "float32": torch.float32,
            "fp32": torch.float32,
            "float16": torch.float16,
            "fp16": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
        }
        if torch_dtype not in mapping:
            raise ValueError(f"Unsupported torch_dtype={torch_dtype}. Use one of: auto, float32, float16, bfloat16")
        return mapping[torch_dtype]

    def _infer_use_chat_template(self, prompt_style: str):
        if prompt_style == "chat":
            return hasattr(self.tokenizer, "apply_chat_template")
        if prompt_style == "raw":
            return False
        model_name = self.model_name.lower()
        return hasattr(self.tokenizer, "apply_chat_template") and any(
            keyword in model_name for keyword in ["instruct", "qwen", "llama-3.1"]
        )

    def _resolve_max_sequence_tokens(self) -> int:
        config_limit = getattr(self.model.config, "max_position_embeddings", None)
        if config_limit is None:
            config_limit = getattr(self.model.config, "n_positions", None)

        tokenizer_limit = getattr(self.tokenizer, "model_max_length", None)
        if tokenizer_limit is not None and tokenizer_limit > 100000:
            tokenizer_limit = None

        limits = [limit for limit in [config_limit, tokenizer_limit] if isinstance(limit, int) and limit > 0]
        if not limits:
            raise ValueError(f"Could not infer a valid max sequence length for model {self.model_name}.")
        return min(limits)

    def _format_observation(self, observation: str) -> str:
        if not self.use_chat_template:
            return observation
        return TWENTY_QUESTIONS_POLICY_TEMPLATE.format(obs=observation.rstrip())

    def _build_chat_messages(self, observation: str):
        return [
            {"role": "system", "content": "You are a concise game-playing assistant."},
            {"role": "user", "content": self._format_observation(observation)},
        ]

    def _build_prompt_texts(self, observations):
        if not self.use_chat_template:
            return [self._format_observation(obs) for obs in observations]
        return [
            self.tokenizer.apply_chat_template(
                self._build_chat_messages(obs),
                tokenize=False,
                add_generation_prompt=True,
            )
            for obs in observations
        ]

    def _tokenize_prompts(self, prompt_texts):
        return self.tokenizer(
            prompt_texts,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=min(self.max_input_length, self.max_sequence_tokens),
            add_special_tokens=not self.use_chat_template,
        ).to(self.model.device)

    def _clean_generated_action(self, text: str) -> str:
        text = text.strip()
        if not text:
            return text
        first_line = text.splitlines()[0].strip()
        prefixes = ["assistant:", "Assistant:", "question:", "Question:"]
        for prefix in prefixes:
            if first_line.startswith(prefix):
                first_line = first_line[len(prefix):].strip()
        return first_line

    def forward(self, observation, do_sample=True):
        prompt_texts = self._build_prompt_texts(observation)
        obs_ids = self._tokenize_prompts(prompt_texts)
        outputs = self.model.generate(
            input_ids=obs_ids["input_ids"],
            attention_mask=obs_ids['attention_mask'],
            max_new_tokens=self.max_new_tokens,
            do_sample=do_sample,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        prompt_lengths = obs_ids['attention_mask'].sum(dim=1)
        actions = []
        for output_ids, prompt_len in zip(outputs, prompt_lengths):
            generated_ids = output_ids[int(prompt_len):]
            decoded = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            actions.append(self._clean_generated_action(decoded))
        return actions

    def behavioral_cloning_loss(self, observation, action, **kwargs):
        logsum_probs = self.get_logsum_prob(observation, action)
        loss = -logsum_probs.mean()
        return loss, {"behavioral_cloning/loss": loss.detach()}

    def _get_prompt_input_ids(self, observation: str):
        prompt_text = self._format_observation(observation)
        if self.use_chat_template:
            prompt_ids = self.tokenizer.apply_chat_template(
                self._build_chat_messages(observation),
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            )
        else:
            prompt_ids = self.tokenizer(
                prompt_text,
                return_tensors="pt",
                truncation=True,
                max_length=min(self.max_input_length, self.max_sequence_tokens),
                add_special_tokens=True,
            )["input_ids"]
        if prompt_ids.ndim == 1:
            prompt_ids = prompt_ids.unsqueeze(0)
        return prompt_ids[0]

    def get_logsum_prob(self, observation, action_from_dataloader, **kwargs):
        batch_input_ids = []
        batch_attention_masks = []
        batch_target_masks = []

        for obs, action in zip(observation, action_from_dataloader):
            prompt_ids = self._get_prompt_input_ids(obs)
            max_action_tokens = self.max_sequence_tokens - 1 if self.tokenizer.eos_token_id is not None else self.max_sequence_tokens
            max_action_tokens = max(1, max_action_tokens)
            action_ids = self.tokenizer(
                action,
                return_tensors='pt',
                add_special_tokens=False,
                truncation=True,
                max_length=max_action_tokens,
            )["input_ids"][0]
            if self.tokenizer.eos_token_id is not None:
                action_ids = torch.cat(
                    [action_ids, torch.tensor([self.tokenizer.eos_token_id], dtype=action_ids.dtype)],
                    dim=0,
                )
            input_ids = torch.cat([prompt_ids, action_ids], dim=0)
            action_token_mask = torch.cat(
                [
                    torch.zeros(prompt_ids.numel(), dtype=torch.bool),
                    torch.ones(action_ids.numel(), dtype=torch.bool),
                ],
                dim=0,
            )
            if input_ids.numel() > self.max_sequence_tokens:
                input_ids = input_ids[-self.max_sequence_tokens:]
                action_token_mask = action_token_mask[-self.max_sequence_tokens:]
            attention_mask = torch.ones_like(input_ids)
            target_mask = action_token_mask[1:]

            batch_input_ids.append(input_ids)
            batch_attention_masks.append(attention_mask)
            batch_target_masks.append(target_mask)

        pad_id = self.tokenizer.pad_token_id
        max_len = max(x.numel() for x in batch_input_ids)
        padded_input_ids = torch.full((len(batch_input_ids), max_len), pad_id, dtype=batch_input_ids[0].dtype)
        padded_attention_mask = torch.zeros((len(batch_input_ids), max_len), dtype=batch_attention_masks[0].dtype)
        padded_target_mask = torch.zeros((len(batch_input_ids), max_len - 1), dtype=torch.bool)

        for i, (input_ids, attention_mask, target_mask) in enumerate(zip(batch_input_ids, batch_attention_masks, batch_target_masks)):
            padded_input_ids[i, -input_ids.numel():] = input_ids
            padded_attention_mask[i, -attention_mask.numel():] = attention_mask
            padded_target_mask[i, -target_mask.numel():] = target_mask

        padded_input_ids = padded_input_ids.to(self.model.device)
        padded_attention_mask = padded_attention_mask.to(self.model.device)
        padded_target_mask = padded_target_mask.to(self.model.device)

        outputs = self.model(input_ids=padded_input_ids, attention_mask=padded_attention_mask)
        shift_logits = outputs.logits[:, :-1, :]
        shift_labels = padded_input_ids[:, 1:]
        shift_log_probs = torch.log_softmax(shift_logits, dim=-1)
        token_log_probs = torch.gather(shift_log_probs, 2, shift_labels.unsqueeze(-1)).squeeze(-1)
        token_log_probs = torch.where(padded_target_mask, token_log_probs, torch.zeros_like(token_log_probs))
        return token_log_probs.sum(dim=1)

    def to_tokens_and_logprobs(self, input_texts):
        inputs = self.tokenizer(
            input_texts,
            padding=True,
            truncation=True,
            max_length=self.max_sequence_tokens,
            return_tensors="pt",
            add_special_tokens=not self.use_chat_template,
        )
        input_ids = inputs.input_ids.to(self.get_device())
        attention_mask = inputs.attention_mask.to(self.get_device())
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        probs = torch.softmax(outputs.logits, dim=-1)

        probs = probs[:, :-1, :]
        input_ids = input_ids[:, 1:]
        gen_probs = torch.gather(probs, 2, input_ids[:, :, None]).squeeze(-1)

        return gen_probs


