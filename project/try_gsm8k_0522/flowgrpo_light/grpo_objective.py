from __future__ import annotations

from typing import Any

import torch


def clipped_grpo_loss(
    current_logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    *,
    clip_range: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    log_ratio = current_logprobs - old_logprobs
    ratio = torch.exp(log_ratio)
    clipped_ratio = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range)
    surrogate = torch.minimum(ratio * advantages, clipped_ratio * advantages)
    loss = -surrogate.mean()
    clip_fraction = ((ratio < 1.0 - clip_range) | (ratio > 1.0 + clip_range)).to(torch.float32).mean()
    approx_kl = (old_logprobs - current_logprobs).mean()
    return loss, {
        "ratio_mean": float(ratio.detach().mean().cpu()),
        "ratio_min": float(ratio.detach().min().cpu()),
        "ratio_max": float(ratio.detach().max().cpu()),
        "clip_fraction": float(clip_fraction.detach().cpu()),
        "approx_kl": float(approx_kl.detach().cpu()),
    }


def build_loss_items(policy: Any, rollouts: list[Any], advantages: list[float]) -> list[tuple[str, str, float, int]]:
    loss_items: list[tuple[str, str, float, int]] = []
    for rollout, advantage in zip(rollouts, advantages, strict=True):
        if abs(float(advantage)) < 1e-8:
            continue
        for sample in rollout.samples:
            response_len = max(1, len(policy._tokenize(sample.response, add_special_tokens=False)))
            loss_items.append((sample.prompt, sample.response, float(advantage), response_len))
    return loss_items


def _compute_old_normalized_logprobs(
    policy: Any,
    loss_items: list[tuple[str, str, float, int]],
    *,
    micro_batch_size: int,
) -> torch.Tensor:
    old_parts: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, len(loss_items), micro_batch_size):
            batch = loss_items[start : start + micro_batch_size]
            prompts = [item[0] for item in batch]
            responses = [item[1] for item in batch]
            response_lengths = torch.tensor(
                [item[3] for item in batch],
                dtype=torch.float32,
                device=policy.device,
            )
            old_logprobs = policy.sequence_logprob_many(prompts, responses, use_adapter=True)
            old_parts.append((old_logprobs / response_lengths).detach().cpu())
    return torch.cat(old_parts, dim=0)


def train_step_grpo(
    *,
    policy: Any,
    optimizer: torch.optim.Optimizer,
    rollouts: list[Any],
    advantages: list[float],
    clip_range: float,
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
            prompts = [item[0] for item in batch]
            responses = [item[1] for item in batch]
            response_lengths = torch.tensor(
                [item[3] for item in batch],
                dtype=torch.float32,
                device=policy.device,
            )
            advantages_tensor = torch.tensor(
                [item[2] for item in batch],
                dtype=torch.float32,
                device=policy.device,
            )
            current_logprobs = policy.sequence_logprob_many(prompts, responses, use_adapter=True)
            current_normalized_logprobs = current_logprobs / response_lengths
            old_batch = old_normalized_logprobs[start : start + len(batch)].to(
                device=policy.device,
                dtype=current_normalized_logprobs.dtype,
            )
            micro_loss, micro_stats = clipped_grpo_loss(
                current_normalized_logprobs,
                old_batch,
                advantages_tensor.to(dtype=current_normalized_logprobs.dtype),
                clip_range=clip_range,
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
