from __future__ import annotations


def test_verl_gspo_has_finite_nonzero_gradient_for_mixed_advantages() -> None:
    import torch

    from verl.trainer.ppo.core_algos import compute_policy_loss_gspo
    from verl.workers.config.actor import ActorConfig

    config = ActorConfig(
        strategy="fsdp",
        rollout_n=2,
        ppo_micro_batch_size_per_gpu=1,
        clip_ratio_low=0.001,
        clip_ratio_high=0.003,
        global_batch_info={
            "dp_size": 1,
            "batch_num_tokens": None,
            "global_batch_size": 2,
            "loss_scale_factor": None,
        },
    )
    old_log_prob = torch.zeros((2, 3))
    current_log_prob = torch.tensor(
        [[0.0, 0.0, 0.0], [0.0005, 0.0005, 0.0005]],
        requires_grad=True,
    )
    advantages = torch.tensor([[-1.0, -1.0, 0.0], [1.0, 1.0, 1.0]])
    response_mask = torch.tensor([[1, 1, 0], [1, 1, 1]], dtype=torch.bool)

    loss, metrics = compute_policy_loss_gspo(
        old_log_prob=old_log_prob,
        log_prob=current_log_prob,
        advantages=advantages,
        response_mask=response_mask,
        config=config,
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert current_log_prob.grad is not None
    assert torch.isfinite(current_log_prob.grad).all()
    assert current_log_prob.grad.abs().sum() > 0
    assert set(metrics) == {
        "actor/pg_clipfrac",
        "actor/ppo_kl",
        "actor/pg_clipfrac_lower",
    }
