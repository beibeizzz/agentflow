from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn.functional as F


@dataclass
class GeneratedResponse:
    prompt: str
    response: str


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
    ) -> None:
        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
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
        config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            task_type=TaskType.CAUSAL_LM,
            target_modules="all-linear",
        )
        self.model = get_peft_model(self.model, config)
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

    def generate(self, prompt: str) -> GeneratedResponse:
        self.eval()
        inputs = self._encode_prompt(prompt)
        input_len = inputs["input_ids"].shape[-1]
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                do_sample=True,
                temperature=self.temperature,
                top_p=self.top_p,
                max_new_tokens=self.max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        response_ids = output[0, input_len:]
        response = self.tokenizer.decode(response_ids, skip_special_tokens=True).strip()
        return GeneratedResponse(prompt=prompt, response=response)

    def sequence_logprob(self, prompt: str, response: str, *, use_adapter: bool = True) -> torch.Tensor:
        prompt_ids = self._tokenize(prompt, add_special_tokens=True)
        response_ids = self._tokenize(response, add_special_tokens=False)
        if not response_ids:
            response_ids = [self.tokenizer.eos_token_id]

        input_ids = torch.tensor([prompt_ids + response_ids], device=self.device)
        attention_mask = torch.ones_like(input_ids)

        if use_adapter:
            logits = self.model(input_ids=input_ids, attention_mask=attention_mask).logits
        else:
            with self.model.disable_adapter():
                logits = self.model(input_ids=input_ids, attention_mask=attention_mask).logits

        shift_logits = logits[:, :-1, :]
        shift_labels = input_ids[:, 1:]
        logprobs = F.log_softmax(shift_logits, dim=-1)
        token_logprobs = logprobs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)

        response_start = max(len(prompt_ids) - 1, 0)
        response_token_logprobs = token_logprobs[:, response_start:]
        return response_token_logprobs.sum()

    def _encode_prompt(self, prompt: str) -> dict[str, torch.Tensor]:
        inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
        return {key: value.to(self.device) for key, value in inputs.items()}

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
