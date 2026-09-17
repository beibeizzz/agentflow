from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class QwenProcessRewardModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    model_name_or_path: str = "Qwen/Qwen3-0.6B"
    max_length: int = Field(default=8192, ge=256)
    learning_rate: float = Field(default=1e-5, gt=0)
    num_train_epochs: float = Field(default=1.0, gt=0)
    regression_head_dropout: float = Field(default=0.0, ge=0.0, lt=1.0)


def mse_regression_loss(predictions, targets):
    import torch.nn.functional as functional

    return functional.mse_loss(predictions.float().view(-1), targets.float().view(-1))


__all__ = ["QwenProcessRewardModelConfig", "mse_regression_loss"]
