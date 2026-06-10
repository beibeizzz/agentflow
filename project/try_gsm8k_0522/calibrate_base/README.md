# Base GSM8K Calibration Pipeline

This folder calibrates `gsm8k_train.json` with the base Qwen3-0.6B model served by vLLM. It reuses `try_gsm8k_0522/run_gsm8k_agentflow.py` solver construction so AgentFlow prompts and control flow match the baseline path.

## vLLM

Start vLLM before running calibration:

```bash
vllm serve /home/north/vllm_test/models/Qwen/Qwen3-0.6B \
  --host 0.0.0.0 \
  --port 8000 \
  --served-model-name Qwen3-0.6B \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.80 \
  --max-model-len 4096 \
  --attention-backend FLASH_ATTN
```

## Rollout

Default rollout settings:

- 4 questions per batch.
- 8 rollouts per question.
- 32 worker threads per question batch.
- `planner_next_step` temperature is `1.2`.
- query analysis, final answer, verifier, and executor temperatures are `0.0`.
- all subagent `max_tokens` are `512`.
- think modes: planner/off through global `think_mode=off`, query analysis/on, final/on, verifier/on.

Run a small smoke calibration first:

```bash
LIMIT=8 OVERWRITE=1 bash try_gsm8k_0522/calibrate_base/run_calibration.sh
bash try_gsm8k_0522/calibrate_base/run_aggregate.sh
```

Run full calibration:

```bash
OVERWRITE=1 bash try_gsm8k_0522/calibrate_base/run_calibration.sh
bash try_gsm8k_0522/calibrate_base/run_aggregate.sh
```

If vLLM is overloaded, reduce concurrency:

```bash
MAX_WORKERS=16 bash try_gsm8k_0522/calibrate_base/run_calibration.sh
```

## Outputs

Default output directory:

```text
try_gsm8k_0522/calibrate_base/outputs/base_calibration
```

Important files:

```text
raw/repeat_00/output_<pid>.json
raw/repeat_01/output_<pid>.json
...
calibration_manifest.jsonl
run_summary.json
calibration_records.jsonl
calibration_summary.json
buckets/gsm8k_train_easy.json
buckets/gsm8k_train_learnable.json
buckets/gsm8k_train_hard.json
buckets/gsm8k_train_bad.json
```

Bucket rule:

- `bad`: fewer than half of expected rollouts are scored.
- `learnable`: scored rewards have non-zero standard deviation.
- `easy`: scored rewards are all correct.
- `hard`: scored rewards are all wrong.
