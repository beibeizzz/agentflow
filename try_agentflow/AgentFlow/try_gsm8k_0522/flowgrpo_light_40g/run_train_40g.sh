#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export NO_PROXY="${NO_PROXY:-},127.0.0.1,localhost"
export no_proxy="${no_proxy:-},127.0.0.1,localhost"

MODEL_PATH="${MODEL_PATH:-/home/north/vllm_test/models/Qwen/Qwen3-0.6B}"
FROZEN_BASE_URL="${FROZEN_BASE_URL:-http://127.0.0.1:8000/v1}"
FROZEN_MODEL="${FROZEN_MODEL:-Qwen3-0.6B}"
TRAIN_FILE="${TRAIN_FILE:-try_gsm8k_0522/data/gsm8k_test_train_1000.json}"
OUTPUT_DIR="${OUTPUT_DIR:-try_gsm8k_0522/flowgrpo_light_40g/outputs/train_40g}"
ROLLOUT_BACKEND="${ROLLOUT_BACKEND:-agentflow}"
THINK_MODE="${THINK_MODE:-off}"
QUERY_ANALYSIS_THINK_MODE="${QUERY_ANALYSIS_THINK_MODE:-on}"
FINAL_OUTPUT_THINK_MODE="${FINAL_OUTPUT_THINK_MODE:-on}"
VERIFIER_THINK_MODE="${VERIFIER_THINK_MODE:-on}"
PYTHON_BIN="${PYTHON:-/home/north/vllm_test/.venv/bin/python}"

echo "Using MODEL_PATH=${MODEL_PATH}"
echo "Using FROZEN_BASE_URL=${FROZEN_BASE_URL}"
echo "Using FROZEN_MODEL=${FROZEN_MODEL}"
echo "Using TRAIN_FILE=${TRAIN_FILE}"
echo "Using OUTPUT_DIR=${OUTPUT_DIR}"
echo "Using ROLLOUT_BACKEND=${ROLLOUT_BACKEND}"
echo "Using THINK_MODE=${THINK_MODE}"
echo "Using QUERY_ANALYSIS_THINK_MODE=${QUERY_ANALYSIS_THINK_MODE}"
echo "Using FINAL_OUTPUT_THINK_MODE=${FINAL_OUTPUT_THINK_MODE}"
echo "Using VERIFIER_THINK_MODE=${VERIFIER_THINK_MODE}"

if [[ ! -f "$TRAIN_FILE" ]]; then
  bash try_gsm8k_0522/flowgrpo_light_40g/prepare_split_1000.sh
fi

"$PYTHON_BIN" try_gsm8k_0522/flowgrpo_light/train_light_grpo.py \
  --config try_gsm8k_0522/flowgrpo_light_40g/config_train_40g.yaml \
  --model-path "$MODEL_PATH" \
  --train-file "$TRAIN_FILE" \
  --output-dir "$OUTPUT_DIR" \
  --frozen-base-url "$FROZEN_BASE_URL" \
  --frozen-model "$FROZEN_MODEL" \
  --rollout-backend "$ROLLOUT_BACKEND" \
  --think-mode "$THINK_MODE" \
  --query-analysis-think-mode "$QUERY_ANALYSIS_THINK_MODE" \
  --final-output-think-mode "$FINAL_OUTPUT_THINK_MODE" \
  --verifier-think-mode "$VERIFIER_THINK_MODE"
