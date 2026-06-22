# FlowGRPO General 2x40G

This variant keeps the same rollout path as `flowgrpo_light_2x40g`, but the planner update uses old-logprob importance sampling with PPO/GRPO-style ratio clipping.

The KL coefficient is intentionally omitted and treated as `0`.

## Objective

For each trainable planner response:

```text
old_logprob = log pi_old(response | prompt)
current_logprob = log pi_theta(response | prompt)
ratio = exp(current_logprob - old_logprob)
loss = -min(
    ratio * advantage,
    clip(ratio, 1-eps_low, 1+eps_high) * advantage,
)
```

The implementation uses the exact sampled response token IDs, normalizes
sequence logprob by response token length, and applies asymmetric clipping.

The default sequence-ratio clipping bounds are:

```text
clip_range_low=0.001
clip_range_high=0.003
```

## Start vLLM

```bash
cd /home/north/vllm_test/new_agentflow0527/agentflow/try_agentflow/AgentFlow
CUDA_VISIBLE_DEVICES=1 bash try_gsm8k_0522/flowgrpo_general_2x40g/run_vllm_gpu1.sh
```

If you already started vLLM manually from `local_model/run_vllm.txt`, skip this step.

## Train

```bash
cd /home/north/vllm_test/new_agentflow0527/agentflow/try_agentflow/AgentFlow
CUDA_VISIBLE_DEVICES=0 bash try_gsm8k_0522/flowgrpo_general_2x40g/run_train_general_2x40g.sh
```

Useful overrides:

```bash
CLIP_RANGE_LOW=0.001 CLIP_RANGE_HIGH=0.003 \
  bash try_gsm8k_0522/flowgrpo_general_2x40g/run_train_general_2x40g.sh
```

```bash
GROUP_SIZE=4 ROLLOUT_CONCURRENCY=16 PLANNER_BATCH_SIZE=16 \
  bash try_gsm8k_0522/flowgrpo_general_2x40g/run_train_general_2x40g.sh
```

## Outputs

Default output directory:

```text
try_gsm8k_0522/flowgrpo_general_2x40g/outputs/train_general_2x40g
```

Important files:

```text
metrics.jsonl
summary_metrics.jsonl
train_summary.json
rollout_logs/step_000001.log
checkpoint_step_<n>/
final_adapter/
```

The terminal prints one high-level line per training step, including reward mean/std, loss, ratio mean, clip fraction, valid rollout count, GPU memory, and timing.
