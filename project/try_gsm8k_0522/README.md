# GSM8K AgentFlow Qwen3-0.6B Evaluation

This folder contains a self-contained GSM8K evaluation for the local vLLM model served as `Qwen3-0.6B-Instruct`.

## Prerequisites

Start vLLM before running the scripts:

```bash
vllm serve /home/north/vllm_test/model \
  --host 0.0.0.0 \
  --port 8000 \
  --served-model-name Qwen3-0.6B-Instruct \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.85 \
  --max-model-len 8192
```

The Python environment needs AgentFlow dependencies and `pyarrow`:

```bash
uv pip install --python /home/north/vllm_test/.venv/bin/python -e ..
uv pip install --python /home/north/vllm_test/.venv/bin/python pyarrow tenacity
```

## Commands

Run the 20-example smoke experiments:

```bash
bash try_gsm8k_0522/run_smoke.sh
```

This runs three variants with `max_tokens=2048`:

- `smoke_calculator_steps1`: `output_types=direct`, `max_steps=1`.
- `smoke_calculator_steps2`: `output_types=direct`, `max_steps=2`.
- `smoke_calculator_steps3`: `output_types=direct`, `max_steps=3`.

All variants use the custom `Calculator_Tool` as the only available AgentFlow tool and prompt the model for a `Planner` / `Executor` / `Verifier` / `Generator` solution format.

The role-specific prompt design is documented in `PROMPT_DESIGN.md`.

Run the full 1319-example test:

```bash
bash try_gsm8k_0522/run_full.sh
```

From this directory, the equivalent commands are:

```bash
bash run_smoke.sh
bash run_full.sh
```

## Outputs

- `data/gsm8k_test.json`: converted full test split.
- `data/gsm8k_smoke_20.json`: first 20 examples for smoke testing.
- `results/smoke_calculator_steps1/output_*.json`: calculator-only AgentFlow outputs with `max_steps=1`.
- `results/smoke_calculator_steps2/output_*.json`: calculator-only AgentFlow outputs with `max_steps=2`.
- `results/smoke_calculator_steps3/output_*.json`: calculator-only AgentFlow outputs with `max_steps=3`.
- `results/full/output_*.json`: per-example full outputs, when full evaluation is run.
- `summary/smoke_calculator_steps1_summary.json`: `max_steps=1` smoke accuracy and details.
- `summary/smoke_calculator_steps2_summary.json`: `max_steps=2` smoke accuracy and details.
- `summary/smoke_calculator_steps3_summary.json`: `max_steps=3` smoke accuracy and details.
- `summary/full_summary.json`: full accuracy and details, when full evaluation is run.
- `logs/smoke/*.log` and `logs/full/*.log`: command logs.
- `logs/*/problems/`: noisy AgentFlow per-problem logs.

Scoring extracts the GSM8K gold answer from `#### ANSWER`, prioritizes the model's `Generator:` line, falls back to `#### ...`, boxed answers, and then the final number, and compares values numerically. Missing or errored result files count as incorrect in aggregate accuracy.

`run_full.sh` uses `WORKERS=4` by default. Override it when needed:

```bash
WORKERS=2 bash run_full.sh
```

## 2026-05-22 Smoke Results

The smoke run used vLLM model `Qwen3-0.6B-Instruct` at `http://localhost:8000/v1`, `temperature=0.0`, and `max_tokens=2048`.

| Variant | Output type | Max steps | Correct / Total | Accuracy |
| --- | --- | ---: | ---: | ---: |
| `smoke_calculator_steps1` | `direct` | 1 | 1 / 20 | 5.0% |
| `smoke_calculator_steps2` | `direct` | 2 | 3 / 20 | 15.0% |
| `smoke_calculator_steps3` | `direct` | 3 | 0 / 20 | 0.0% |

Configured and final available tool: `Calculator_Tool`.
