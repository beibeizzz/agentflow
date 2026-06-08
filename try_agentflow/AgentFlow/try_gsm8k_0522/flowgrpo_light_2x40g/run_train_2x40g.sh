#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export NO_PROXY="${NO_PROXY:-},127.0.0.1,localhost"
export no_proxy="${no_proxy:-},127.0.0.1,localhost"

PYTHON_BIN="${PYTHON:-/inspire/hdd/project/fdu-aidake-cfff/public/zhangjiangbei/agentflow/AgentFlow/.venv/bin/python}"
CONFIG_FILE="${CONFIG_FILE:-try_gsm8k_0522/flowgrpo_light_2x40g/config_train_2x40g.yaml}"
MODEL_PATH="${MODEL_PATH:-/inspire/hdd/project/fdu-aidake-cfff/public/zhangjiangbei/agentflow/AgentFlow/agentflow0525/agentflow/model/Qwen/Qwen3-0.6B}"
FROZEN_BASE_URL="${FROZEN_BASE_URL:-http://127.0.0.1:8000/v1}"
FROZEN_MODEL="${FROZEN_MODEL:-Qwen3-0.6B}"
TRAIN_FILE="${TRAIN_FILE:-try_gsm8k_0522/data/gsm8k_train_learnable.json}"
OUTPUT_DIR="${OUTPUT_DIR:-try_gsm8k_0522/flowgrpo_light_2x40g/outputs/train_2x40g}"
ROLLOUT_BACKEND="${ROLLOUT_BACKEND:-agentflow}"
QUESTION_BATCH_SIZE="${QUESTION_BATCH_SIZE:-8}"
GROUP_SIZE="${GROUP_SIZE:-6}"
ROLLOUT_CONCURRENCY="${ROLLOUT_CONCURRENCY:-24}"
PLANNER_BATCH_SIZE="${PLANNER_BATCH_SIZE:-24}"
PLANNER_BATCH_TIMEOUT_S="${PLANNER_BATCH_TIMEOUT_S:-0.10}"
LOGPROB_MICRO_BATCH_SIZE="${LOGPROB_MICRO_BATCH_SIZE:-12}"
MAX_TRAIN_ITEMS="${MAX_TRAIN_ITEMS:-2000}"
EPOCHS="${EPOCHS:-1}"
THINK_MODE="${THINK_MODE:-off}"
QUERY_ANALYSIS_THINK_MODE="${QUERY_ANALYSIS_THINK_MODE:-on}"
FINAL_OUTPUT_THINK_MODE="${FINAL_OUTPUT_THINK_MODE:-off}"
VERIFIER_THINK_MODE="${VERIFIER_THINK_MODE:-on}"
ROLLOUT_LOG_DIR="${ROLLOUT_LOG_DIR:-${OUTPUT_DIR}/rollout_logs}"

echo "Using CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "Using CONFIG_FILE=${CONFIG_FILE}"
echo "Using MODEL_PATH=${MODEL_PATH}"
echo "Using FROZEN_BASE_URL=${FROZEN_BASE_URL}"
echo "Using FROZEN_MODEL=${FROZEN_MODEL}"
echo "Using TRAIN_FILE=${TRAIN_FILE}"
echo "Using OUTPUT_DIR=${OUTPUT_DIR}"
echo "Using ROLLOUT_BACKEND=${ROLLOUT_BACKEND}"
echo "Using QUESTION_BATCH_SIZE=${QUESTION_BATCH_SIZE}"
echo "Using GROUP_SIZE=${GROUP_SIZE}"
echo "Using ROLLOUT_CONCURRENCY=${ROLLOUT_CONCURRENCY}"
echo "Using PLANNER_BATCH_SIZE=${PLANNER_BATCH_SIZE}"
echo "Using PLANNER_BATCH_TIMEOUT_S=${PLANNER_BATCH_TIMEOUT_S}"
echo "Using LOGPROB_MICRO_BATCH_SIZE=${LOGPROB_MICRO_BATCH_SIZE}"
echo "Using MAX_TRAIN_ITEMS=${MAX_TRAIN_ITEMS}"
echo "Using EPOCHS=${EPOCHS}"
echo "Using ROLLOUT_LOG_DIR=${ROLLOUT_LOG_DIR}"

if [[ ! -f "$TRAIN_FILE" ]]; then
  echo "Training file does not exist: ${TRAIN_FILE}" >&2
  exit 1
fi

"$PYTHON_BIN" try_gsm8k_0522/flowgrpo_light/train_light_grpo_quiet.py \
  --config "$CONFIG_FILE" \
  --model-path "$MODEL_PATH" \
  --train-file "$TRAIN_FILE" \
  --output-dir "$OUTPUT_DIR" \
  --frozen-base-url "$FROZEN_BASE_URL" \
  --frozen-model "$FROZEN_MODEL" \
  --rollout-backend "$ROLLOUT_BACKEND" \
  --question-batch-size "$QUESTION_BATCH_SIZE" \
  --group-size "$GROUP_SIZE" \
  --rollout-concurrency "$ROLLOUT_CONCURRENCY" \
  --planner-batch-size "$PLANNER_BATCH_SIZE" \
  --planner-batch-timeout-s "$PLANNER_BATCH_TIMEOUT_S" \
  --logprob-micro-batch-size "$LOGPROB_MICRO_BATCH_SIZE" \
  --max-train-items "$MAX_TRAIN_ITEMS" \
  --epochs "$EPOCHS" \
  --think-mode "$THINK_MODE" \
  --query-analysis-think-mode "$QUERY_ANALYSIS_THINK_MODE" \
  --final-output-think-mode "$FINAL_OUTPUT_THINK_MODE" \
  --verifier-think-mode "$VERIFIER_THINK_MODE" \
  --quiet-rollout \
  --rollout-log-dir "$ROLLOUT_LOG_DIR"
