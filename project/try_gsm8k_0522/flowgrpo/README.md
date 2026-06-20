# GSM8K AgentFlow Flow-GRPO

This directory contains the GSM8K Flow-GRPO experiment that keeps the baseline AgentFlow pipeline but trains only `Planner.generate_next_step`.

## Training Boundary

Trainable:

- `planner.generate_next_step`: the planner action JSON containing `Sub_goal` and `Calculation`.

Frozen:

- `planner.analyze_query`
- `executor.generate_tool_command`
- `verifier.verificate_context`
- `planner.generate_final_output`
- `planner.generate_direct_output`

The rollout worker wraps only `planner.generate_next_step` in an AgentOps agent span named `planner_next_step`; the AgentFlow trainer filters triplets to that span through `trained_agents="planner_next_step"`.

## Runtime Model Layout

Use one model checkpoint path on disk and two runtime instances:

- The trainable actor/rollout model is started by verl from `actor_rollout_ref.model.path`.
- The frozen model is served separately through OpenAI-compatible vLLM at `FROZEN_BASE_URL`.

Start the frozen model first:

```bash
vllm serve /home/north/vllm_test/models/Qwen/Qwen3-0.6B \
  --host 0.0.0.0 \
  --port 8000 \
  --served-model-name Qwen3-0.6B-Frozen \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.25 \
  --max-model-len 4096
```

The smoke config enables LoRA with `actor_rollout_ref.model.lora_rank=8`, `lora_alpha=16`, `target_modules=all-linear`, and `rollout.load_format=safetensors`.

## Commands

From `try_agentflow/AgentFlow/try_gsm8k_0522/flowgrpo`:

```bash
bash prepare_data.sh
```

Then use two terminals from the same directory:

```bash
bash run_rollout_worker.sh
```

```bash
bash run_train_smoke.sh
```

The trainer owns the trainable vLLM instance. The worker polls the AgentFlow control server and calls the frozen vLLM endpoint for non-trainable subagents.

## Outputs

- `data/train/gsm8k_smoke_train.parquet`
- `data/val/gsm8k_smoke_val.parquet`
- `rollout_logs/<experiment>/train/*.json`
- `rollout_logs/<experiment>/val/*.json`

Rewards are rule-based GSM8K numeric correctness: `1.0` for a numerically equivalent final answer, otherwise `0.0`.
