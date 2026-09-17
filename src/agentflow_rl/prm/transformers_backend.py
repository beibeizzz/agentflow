from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from agentflow_rl.rewards.schemas import ProcessTransition
from agentflow_rl.rewards.transition_view import encode_process_transitions, render_process_transition, PROCESS_VIEW_REVISION
from agentflow_rl.rewards.rubric import PROCESS_RUBRIC_REVISION, validate_rubric_revision


def validate_checkpoint_protocol(config: Any) -> None:
    if getattr(config, "num_labels", None) != 1:
        raise ValueError("PRM checkpoint must have one scalar regression output")
    if getattr(config, "agentflow_process_view_revision", None) != PROCESS_VIEW_REVISION:
        raise ValueError("PRM checkpoint input view differs from the current protocol; retrain with current labels")
    validate_rubric_revision(getattr(config, "agentflow_process_rubric_revision", "legacy"))


class TransformersProcessRewardBackend:
    rubric_revision = PROCESS_RUBRIC_REVISION
    def __init__(
        self,
        *,
        model: Any,
        tokenizer: Any,
        revision: str,
        device: str,
        max_length: int = 8192,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.revision = revision
        self.device = device
        self.max_length = max_length
        self._lock = asyncio.Lock()

    @classmethod
    def from_pretrained(
        cls,
        path: str | Path,
        *,
        revision: str,
        device: str = "cuda:0",
        max_length: int = 8192,
    ) -> "TransformersProcessRewardBackend":
        import torch
        from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

        config = AutoConfig.from_pretrained(str(path))
        validate_checkpoint_protocol(config)
        tokenizer = AutoTokenizer.from_pretrained(str(path), use_fast=True)
        model = AutoModelForSequenceClassification.from_pretrained(
            str(path),
            config=config,
            torch_dtype=torch.bfloat16 if device.startswith("cuda") else torch.float32,
        ).to(device)
        validate_checkpoint_protocol(model.config)
        model.eval()
        return cls(
            model=model,
            tokenizer=tokenizer,
            revision=revision,
            device=device,
            max_length=max_length,
        )

    async def predict(self, transition: ProcessTransition) -> float:
        async with self._lock:
            return await asyncio.to_thread(self._predict_blocking, transition)

    async def predict_many(
        self, transitions: tuple[ProcessTransition, ...]
    ) -> tuple[float, ...]:
        if not transitions:
            return ()
        async with self._lock:
            return await asyncio.to_thread(self._predict_many_blocking, transitions)

    def _predict_blocking(self, transition: ProcessTransition) -> float:
        return self._predict_many_blocking((transition,))[0]

    def _predict_many_blocking(
        self, transitions: tuple[ProcessTransition, ...]
    ) -> tuple[float, ...]:
        import torch

        encoded = encode_process_transitions(
            transitions, self.tokenizer, max_length=self.max_length,
            return_tensors="pt",
            padding=True,
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        with torch.inference_mode():
            logits = self.model(**encoded).logits.float().view(-1)
        return tuple(float(value) for value in torch.sigmoid(logits).tolist())


__all__ = ["TransformersProcessRewardBackend", "render_process_transition"]
