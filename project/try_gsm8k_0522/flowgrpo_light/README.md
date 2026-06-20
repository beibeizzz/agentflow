# Lightweight Flow-GRPO

This is a minimal planner-only GRPO path for GSM8K. It avoids verl, Ray, and trainable vLLM rollout. The trainable planner is a local `transformers` + `peft` LoRA model, while the frozen model stays behind an OpenAI-compatible vLLM HTTP server.

Only planner next-step JSON responses are trained:

```json
{"Sub_goal": "...", "Calculation": "6*5"}
```

Frozen vLLM is used for query analysis and final numeric answer generation.

## 12G Smoke Run

Start the frozen server first:

```bash
vllm serve model/Qwen/Qwen3-0.6B \
  --host 0.0.0.0 \
  --port 8000 \
  --served-model-name Qwen3-0.6B \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.60 \
  --max-model-len 4096
```

Then run:

```bash
bash try_gsm8k_0522/flowgrpo_light/run_smoke_12g.sh
```

The 12G config uses:

- `question_batch_size: 1`
- `group_size: 2`
- `planner_batch_size: 2`
- `logprob_micro_batch_size: 4`
- `max_train_items: 4`
- `lora_rank: 8`
- `planner_max_new_tokens: 128`
- local Qwen3-0.6B LoRA training

Outputs are written to:

```text
try_gsm8k_0522/flowgrpo_light/outputs/smoke_12g/
```

The final LoRA adapter is saved at:

```text
try_gsm8k_0522/flowgrpo_light/outputs/smoke_12g/final_adapter/
```

## Why This Exists

The official AgentFlow + verl path currently couples Ray, async vLLM, LoRA serving, AgentOps token extraction, and vLLM V1 internals. This lightweight path isolates the experiment question: whether GRPO improves `planner.generate_next_step` for the GSM8K calculator pipeline.
