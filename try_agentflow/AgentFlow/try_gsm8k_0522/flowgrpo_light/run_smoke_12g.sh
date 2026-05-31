#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export NO_PROXY="${NO_PROXY:-},127.0.0.1,localhost"
export no_proxy="${no_proxy:-},127.0.0.1,localhost"

if [[ -z "${MODEL_PATH:-}" ]]; then
  for candidate in \
    "model/Qwen/Qwen3-0.6B" \
    "models/Qwen/Qwen3-0.6B" \
    "../../../../models/Qwen/Qwen3-0.6B"
  do
    if [[ -d "$candidate" ]]; then
      MODEL_PATH="$candidate"
      break
    fi
  done
fi

MODEL_PATH="${MODEL_PATH:-/home/north/vllm_test/models/Qwen/Qwen3-0.6B}"
echo "Using MODEL_PATH=${MODEL_PATH}"

FROZEN_BASE_URL="${FROZEN_BASE_URL:-http://127.0.0.1:8000/v1}"
FROZEN_MODEL="${FROZEN_MODEL:-Qwen3-0.6B}"
ROLLOUT_BACKEND="${ROLLOUT_BACKEND:-agentflow}"
THINK_MODE="${THINK_MODE:-default}"
QUERY_ANALYSIS_THINK_MODE="${QUERY_ANALYSIS_THINK_MODE:-$THINK_MODE}"
FINAL_OUTPUT_THINK_MODE="${FINAL_OUTPUT_THINK_MODE:-$THINK_MODE}"
VERIFIER_THINK_MODE="${VERIFIER_THINK_MODE:-$THINK_MODE}"
echo "Using FROZEN_BASE_URL=${FROZEN_BASE_URL}"
echo "Using FROZEN_MODEL=${FROZEN_MODEL}"
echo "Using ROLLOUT_BACKEND=${ROLLOUT_BACKEND}"
echo "Using THINK_MODE=${THINK_MODE}"
echo "Using QUERY_ANALYSIS_THINK_MODE=${QUERY_ANALYSIS_THINK_MODE}"
echo "Using FINAL_OUTPUT_THINK_MODE=${FINAL_OUTPUT_THINK_MODE}"
echo "Using VERIFIER_THINK_MODE=${VERIFIER_THINK_MODE}"

PYTHON_BIN="${PYTHON:-}"
if [[ -z "$PYTHON_BIN" ]]; then
  if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
  else
    PYTHON_BIN="python"
  fi
fi

"$PYTHON_BIN" try_gsm8k_0522/flowgrpo_light/train_light_grpo.py \
  --config try_gsm8k_0522/flowgrpo_light/config_smoke_12g.yaml \
  --model-path "$MODEL_PATH" \
  --frozen-base-url "$FROZEN_BASE_URL" \
  --frozen-model "$FROZEN_MODEL" \
  --rollout-backend "$ROLLOUT_BACKEND" \
  --think-mode "$THINK_MODE" \
  --query-analysis-think-mode "$QUERY_ANALYSIS_THINK_MODE" \
  --final-output-think-mode "$FINAL_OUTPUT_THINK_MODE" \
  --verifier-think-mode "$VERIFIER_THINK_MODE"
