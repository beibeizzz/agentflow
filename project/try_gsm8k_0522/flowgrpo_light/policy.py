from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import torch
import torch.nn.functional as F


@dataclass
class GeneratedResponse:
    prompt: str
    response: str


def _adapter_disabled(value: Any) -> bool:
    return str(value).strip().lower() in {"", "0", "false", "none", "no", "off"}


class PlannerPolicy:
    def __init__(
        self,
        model_path: str,
        *,
        lora_rank: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.0,
        max_new_tokens: int = 128,
        temperature: float = 0.8,
        top_p: float = 0.95,
        dtype: str = "bfloat16",
        gradient_checkpointing: bool = True,
        adapter_path: str | bool | None = None,
    ) -> None:
        from peft import LoraConfig, PeftModel, TaskType, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer

        adapter_is_disabled = _adapter_disabled(adapter_path)
        tokenizer_path = model_path if adapter_is_disabled else adapter_path or model_path
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        torch_dtype = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }[dtype]
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
            device_map="auto",
        )
        if gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
            self.model.config.use_cache = False
        if adapter_path is None:
            config = LoraConfig(
                r=lora_rank,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                task_type=TaskType.CAUSAL_LM,
                target_modules="all-linear",
            )
            self.model = get_peft_model(self.model, config)
        elif not adapter_is_disabled:
            self.model = PeftModel.from_pretrained(self.model, adapter_path)
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def train(self) -> None:
        self.model.train()

    def eval(self) -> None:
        self.model.eval()

    def save_adapter(self, output_dir: str) -> None:
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)

    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        do_sample: bool | None = None,
    ) -> GeneratedResponse:
        return self.generate_many(
            [prompt],
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
        )[0]

    def generate_many(
        self,
        prompts: list[str],
        *,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        do_sample: bool | None = None,
    ) -> list[GeneratedResponse]:
        if not prompts:
            return []
        self.eval()
        inputs = self._encode_prompts(prompts)
        prompt_width = inputs["input_ids"].shape[-1]
        effective_temperature = self.temperature if temperature is None else float(temperature)
        effective_top_p = self.top_p if top_p is None else float(top_p)
        effective_max_new_tokens = self.max_new_tokens if max_new_tokens is None else int(max_new_tokens)
        effective_do_sample = effective_temperature > 0 if do_sample is None else bool(do_sample)
        generation_kwargs = {
            **inputs,
            "do_sample": effective_do_sample,
            "max_new_tokens": effective_max_new_tokens,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if effective_do_sample:
            generation_kwargs["temperature"] = effective_temperature
            generation_kwargs["top_p"] = effective_top_p
        with torch.no_grad():
            output = self.model.generate(**generation_kwargs)
        generated: list[GeneratedResponse] = []
        for prompt, output_ids in zip(prompts, output, strict=True):
            response_ids = output_ids[prompt_width:]
            response = self.tokenizer.decode(response_ids, skip_special_tokens=True).strip()
            generated.append(GeneratedResponse(prompt=prompt, response=response))
        return generated

    def render_agentflow_prompt(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        think_mode: str = "default",
    ) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        if hasattr(self.tokenizer, "apply_chat_template"):
            template_kwargs = {}
            if think_mode in {"on", "off"}:
                template_kwargs["enable_thinking"] = think_mode == "on"
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                **template_kwargs,
            )
        if system_prompt:
            return f"{system_prompt}\n\n{prompt}"
        return prompt

    def generate_for_agentflow(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        think_mode: str = "default",
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        do_sample: bool | None = None,
    ) -> GeneratedResponse:
        return self.generate_many_for_agentflow(
            [prompt],
            system_prompts=[system_prompt],
            think_mode=think_mode,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
        )[0]

    def generate_many_for_agentflow(
        self,
        prompts: list[str],
        *,
        system_prompts: list[str | None] | None = None,
        think_mode: str = "default",
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        do_sample: bool | None = None,
    ) -> list[GeneratedResponse]:
        if system_prompts is None:
            system_prompts = [None for _ in prompts]
        if len(system_prompts) != len(prompts):
            raise ValueError("system_prompts must have the same length as prompts")
        rendered_prompts = [
            self.render_agentflow_prompt(prompt, system_prompt=system_prompt, think_mode=think_mode)
            for prompt, system_prompt in zip(prompts, system_prompts, strict=True)
        ]
        return self.generate_many(
            rendered_prompts,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
        )

    def sequence_logprob(self, prompt: str, response: str, *, use_adapter: bool = True) -> torch.Tensor:
        return self.sequence_logprob_many([prompt], [response], use_adapter=use_adapter)[0]

    def sequence_logprob_many(
        self,
        prompts: list[str],
        responses: list[str],
        *,
        use_adapter: bool = True,
    ) -> torch.Tensor:
        if len(prompts) != len(responses):
            raise ValueError("prompts and responses must have the same length")
        if not prompts:
            return torch.empty(0, device=self.device)

        input_ids, attention_mask, response_mask = self._encode_logprob_batch(prompts, responses)
        if use_adapter:
            logits = self.model(input_ids=input_ids, attention_mask=attention_mask).logits
        else:
            with self.model.disable_adapter():
                logits = self.model(input_ids=input_ids, attention_mask=attention_mask).logits

        shift_logits = logits[:, :-1, :]
        shift_labels = input_ids[:, 1:]
        response_positions = response_mask.to(dtype=torch.bool)
        if not bool(response_positions.any()):
            return torch.zeros(len(prompts), dtype=shift_logits.dtype, device=self.device)

        selected_logits = shift_logits[response_positions]
        selected_labels = shift_labels[response_positions]
        selected_token_logprobs = -F.cross_entropy(selected_logits, selected_labels, reduction="none")

        batch_indices = torch.arange(len(prompts), device=self.device).unsqueeze(1).expand_as(shift_labels)
        selected_batch_indices = batch_indices[response_positions]
        sequence_logprobs = torch.zeros(len(prompts), dtype=selected_token_logprobs.dtype, device=self.device)
        sequence_logprobs.index_add_(0, selected_batch_indices, selected_token_logprobs)
        return sequence_logprobs

    def _encode_prompt(self, prompt: str) -> dict[str, torch.Tensor]:
        inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
        return {key: value.to(self.device) for key, value in inputs.items()}

    def _encode_prompts(self, prompts: list[str]) -> dict[str, torch.Tensor]:
        if hasattr(self.tokenizer, "padding_side"):
            self.tokenizer.padding_side = "left"
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            add_special_tokens=True,
            padding=True,
        )
        return {key: value.to(self.device) for key, value in inputs.items()}

    def _encode_logprob_batch(
        self,
        prompts: list[str],
        responses: list[str],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        encoded_prompts = [self._tokenize(prompt, add_special_tokens=True) for prompt in prompts]
        encoded_responses = [
            self._tokenize(response, add_special_tokens=False) or [self.tokenizer.eos_token_id]
            for response in responses
        ]
        sequences = [
            prompt_ids + response_ids
            for prompt_ids, response_ids in zip(encoded_prompts, encoded_responses, strict=True)
        ]
        max_length = max(len(sequence) for sequence in sequences)
        input_ids = torch.full(
            (len(sequences), max_length),
            int(self.tokenizer.pad_token_id),
            dtype=torch.long,
            device=self.device,
        )
        attention_mask = torch.zeros_like(input_ids)
        response_mask = torch.zeros((len(sequences), max(max_length - 1, 0)), dtype=torch.float32, device=self.device)

        for index, (prompt_ids, response_ids, sequence) in enumerate(
            zip(encoded_prompts, encoded_responses, sequences, strict=True)
        ):
            sequence_tensor = torch.tensor(sequence, dtype=torch.long, device=self.device)
            sequence_length = len(sequence)
            input_ids[index, :sequence_length] = sequence_tensor
            attention_mask[index, :sequence_length] = 1
            response_start = max(len(prompt_ids) - 1, 0)
            response_end = response_start + len(response_ids)
            response_mask[index, response_start:response_end] = 1.0
        return input_ids, attention_mask, response_mask

    def _tokenize(self, text: str, *, add_special_tokens: bool) -> list[int]:
        ids = self.tokenizer(text, add_special_tokens=add_special_tokens)["input_ids"]
        return list(ids)


def normalize_advantages(rewards: Iterable[float]) -> list[float]:
    values = torch.tensor(list(rewards), dtype=torch.float32)
    if values.numel() == 0:
        return []
    std = values.std(unbiased=False)
    if float(std) < 1e-6:
        return [0.0 for _ in values]
    advantages = (values - values.mean()) / (std + 1e-6)
    return [float(item) for item in advantages]
