from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


def clipped_grpo_loss(
    current_logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    *,
    clip_range_low: float,
    clip_range_high: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    log_ratio = current_logprobs.float() - old_logprobs.float()
    ratio = torch.exp(log_ratio)
    clipped_ratio = torch.clamp(
        ratio,
        1.0 - clip_range_low,
        1.0 + clip_range_high,
    )
    advantages = advantages.float()
    surrogate = torch.minimum(ratio * advantages, clipped_ratio * advantages)
    loss = -surrogate.mean()
    clip_fraction = (
        (ratio < 1.0 - clip_range_low) | (ratio > 1.0 + clip_range_high)
    ).to(torch.float32).mean()
    approx_kl = (old_logprobs.float() - current_logprobs.float()).mean()
    return loss, {
        "ratio_mean": float(ratio.detach().mean().cpu()),
        "ratio_min": float(ratio.detach().min().cpu()),
        "ratio_max": float(ratio.detach().max().cpu()),
        "clip_fraction": float(clip_fraction.detach().cpu()),
        "approx_kl": float(approx_kl.detach().cpu()),
    }


@dataclass
class LossItem:
    prompt: str
    response: str
    advantage: float
    response_len: int
    prompt_token_ids: list[int] | None
    response_token_ids: list[int] | None


def build_loss_items(policy: Any, rollouts: list[Any], advantages: list[float]) -> list[LossItem]:
    loss_items: list[LossItem] = []
    for rollout, advantage in zip(rollouts, advantages, strict=True):
        if abs(float(advantage)) < 1e-8:
            continue
        for sample in rollout.samples:
            prompt_token_ids = getattr(sample, "prompt_token_ids", None)
            response_token_ids = getattr(sample, "response_token_ids", None)
            if response_token_ids is not None:
                response_len = max(1, len(response_token_ids))
            else:
                response_len = max(1, len(policy._tokenize(sample.response, add_special_tokens=False)))
            loss_items.append(
                LossItem(
                    prompt=sample.prompt,
                    response=sample.response,
                    advantage=float(advantage),
                    response_len=response_len,
                    prompt_token_ids=prompt_token_ids,
                    response_token_ids=response_token_ids,
                )
            )
    return loss_items


def _sequence_logprobs(policy: Any, items: list[LossItem]) -> torch.Tensor:
    values: list[torch.Tensor | None] = [None] * len(items)
    exact_indices = [
        index
        for index, item in enumerate(items)
        if item.prompt_token_ids is not None and item.response_token_ids is not None
    ]
    if exact_indices:
        exact_values = policy.sequence_logprob_token_ids_many(
            [items[index].prompt_token_ids for index in exact_indices],
            [items[index].response_token_ids for index in exact_indices],
            use_adapter=True,
        )
        for index, value in zip(exact_indices, exact_values, strict=True):
            values[index] = value

    legacy_indices = [index for index, value in enumerate(values) if value is None]
    if legacy_indices:
        legacy_values = policy.sequence_logprob_many(
            [items[index].prompt for index in legacy_indices],
            [items[index].response for index in legacy_indices],
            use_adapter=True,
        )
        for index, value in zip(legacy_indices, legacy_values, strict=True):
            values[index] = value

    return torch.stack([value for value in values if value is not None])


def _compute_old_normalized_logprobs(
    policy: Any,
    loss_items: list[LossItem],
    *,
    micro_batch_size: int,
) -> torch.Tensor:
    old_parts: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, len(loss_items), micro_batch_size):
            batch = loss_items[start : start + micro_batch_size]
            response_lengths = torch.tensor(
                [item.response_len for item in batch],
                dtype=torch.float32,
                device=policy.device,
            )
            old_logprobs = _sequence_logprobs(policy, batch)
            old_parts.append((old_logprobs.float() / response_lengths).detach().cpu())
    return torch.cat(old_parts, dim=0)


def train_step_grpo(
    *,
    policy: Any,
    optimizer: torch.optim.Optimizer,
    rollouts: list[Any],
    advantages: list[float],
    clip_range_low: float,
    clip_range_high: float,
    max_grad_norm: float,
    logprob_micro_batch_size: int = 1,
    policy_epochs: int = 1,
) -> dict[str, Any] | None:
    policy.train()
    loss_items = build_loss_items(policy, rollouts, advantages)
    if not loss_items:
        return None

    micro_batch_size = max(1, int(logprob_micro_batch_size))
    policy_epochs = max(1, int(policy_epochs))
    old_normalized_logprobs = _compute_old_normalized_logprobs(
        policy,
        loss_items,
        micro_batch_size=micro_batch_size,
    )

    total_items = len(loss_items)
    epoch_losses: list[float] = []
    epoch_ratio_means: list[float] = []
    epoch_ratio_mins: list[float] = []
    epoch_ratio_maxs: list[float] = []
    epoch_clip_fractions: list[float] = []
    epoch_approx_kls: list[float] = []

    for _ in range(policy_epochs):
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        weighted_stats = {
            "ratio_mean": 0.0,
            "ratio_min": float("inf"),
            "ratio_max": float("-inf"),
            "clip_fraction": 0.0,
            "approx_kl": 0.0,
        }
        for start in range(0, total_items, micro_batch_size):
            batch = loss_items[start : start + micro_batch_size]
            response_lengths = torch.tensor(
                [item.response_len for item in batch],
                dtype=torch.float32,
                device=policy.device,
            )
            advantages_tensor = torch.tensor(
                [item.advantage for item in batch],
                dtype=torch.float32,
                device=policy.device,
            )
            current_logprobs = _sequence_logprobs(policy, batch)
            current_normalized_logprobs = current_logprobs.float() / response_lengths
            old_batch = old_normalized_logprobs[start : start + len(batch)].to(
                device=policy.device,
                dtype=current_normalized_logprobs.dtype,
            )
            micro_loss, micro_stats = clipped_grpo_loss(
                current_normalized_logprobs,
                old_batch,
                advantages_tensor.to(dtype=current_normalized_logprobs.dtype),
                clip_range_low=clip_range_low,
                clip_range_high=clip_range_high,
            )
            (micro_loss * (len(batch) / total_items)).backward()
            total_loss += float(micro_loss.detach().cpu()) * len(batch)
            weighted_stats["ratio_mean"] += micro_stats["ratio_mean"] * len(batch)
            weighted_stats["clip_fraction"] += micro_stats["clip_fraction"] * len(batch)
            weighted_stats["approx_kl"] += micro_stats["approx_kl"] * len(batch)
            weighted_stats["ratio_min"] = min(weighted_stats["ratio_min"], micro_stats["ratio_min"])
            weighted_stats["ratio_max"] = max(weighted_stats["ratio_max"], micro_stats["ratio_max"])

        torch.nn.utils.clip_grad_norm_(policy.model.parameters(), max_grad_norm)
        optimizer.step()
        epoch_losses.append(total_loss / total_items)
        epoch_ratio_means.append(weighted_stats["ratio_mean"] / total_items)
        epoch_ratio_mins.append(weighted_stats["ratio_min"])
        epoch_ratio_maxs.append(weighted_stats["ratio_max"])
        epoch_clip_fractions.append(weighted_stats["clip_fraction"] / total_items)
        epoch_approx_kls.append(weighted_stats["approx_kl"] / total_items)

    return {
        "loss": epoch_losses[-1],
        "policy_loss": epoch_losses[-1],
        "ratio_mean": epoch_ratio_means[-1],
        "ratio_min": epoch_ratio_mins[-1],
        "ratio_max": epoch_ratio_maxs[-1],
        "clip_fraction": epoch_clip_fractions[-1],
        "approx_kl": epoch_approx_kls[-1],
        "effective_sample_count": total_items,
        "policy_epochs": policy_epochs,
        "policy_update_count": policy_epochs,
        "epoch_losses": epoch_losses,
        "epoch_ratio_means": epoch_ratio_means,
        "epoch_ratio_mins": epoch_ratio_mins,
        "epoch_ratio_maxs": epoch_ratio_maxs,
        "epoch_clip_fractions": epoch_clip_fractions,
        "epoch_approx_kls": epoch_approx_kls,
    }
