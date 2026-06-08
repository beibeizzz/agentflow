#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

export NO_PROXY="${NO_PROXY:-},127.0.0.1,localhost"
export no_proxy="${no_proxy:-},127.0.0.1,localhost"

PYTHON_BIN="${PYTHON:-/inspire/hdd/project/fdu-aidake-cfff/public/zhangjiangbei/agentflow/AgentFlow/.venv/bin/python}"
DATA_FILE="${DATA_FILE:-try_gsm8k_0522/data/gsm8k_train.json}"
OUTPUT_DIR="${OUTPUT_DIR:-try_gsm8k_0522/calibrate_base/outputs/base_calibration}"
BASE_URL="${BASE_URL:-http://127.0.0.1:8000/v1}"
LLM_ENGINE_NAME="${LLM_ENGINE_NAME:-vllm-Qwen3-0.6B}"
SUBAGENT_CONFIG="${SUBAGENT_CONFIG:-try_gsm8k_0522/calibrate_base/calibration_subagent_config.json}"

QUESTION_BATCH_SIZE="${QUESTION_BATCH_SIZE:-4}"
ROLLOUTS_PER_QUESTION="${ROLLOUTS_PER_QUESTION:-8}"
MAX_WORKERS="${MAX_WORKERS:-32}"
MAX_STEPS="${MAX_STEPS:-3}"
MAX_TIME="${MAX_TIME:-120}"
MAX_TOKENS="${MAX_TOKENS:-512}"
START="${START:-0}"
LIMIT_ARGS=()
if [[ -n "${LIMIT:-}" ]]; then
  LIMIT_ARGS=(--limit "$LIMIT")
fi
OVERWRITE_ARGS=()
if [[ "${OVERWRITE:-0}" == "1" ]]; then
  OVERWRITE_ARGS=(--overwrite)
fi

echo "Using DATA_FILE=${DATA_FILE}"
echo "Using OUTPUT_DIR=${OUTPUT_DIR}"
echo "Using BASE_URL=${BASE_URL}"
echo "Using LLM_ENGINE_NAME=${LLM_ENGINE_NAME}"
echo "Using QUESTION_BATCH_SIZE=${QUESTION_BATCH_SIZE}"
echo "Using ROLLOUTS_PER_QUESTION=${ROLLOUTS_PER_QUESTION}"
echo "Using MAX_WORKERS=${MAX_WORKERS}"

"$PYTHON_BIN" try_gsm8k_0522/calibrate_base/calibration_runner.py \
  --data-file "$DATA_FILE" \
  --output-dir "$OUTPUT_DIR" \
  --base-url "$BASE_URL" \
  --llm-engine-name "$LLM_ENGINE_NAME" \
  --subagent-config "$SUBAGENT_CONFIG" \
  --question-batch-size "$QUESTION_BATCH_SIZE" \
  --rollouts-per-question "$ROLLOUTS_PER_QUESTION" \
  --max-workers "$MAX_WORKERS" \
  --max-steps "$MAX_STEPS" \
  --max-time "$MAX_TIME" \
  --max-tokens "$MAX_TOKENS" \
  --start "$START" \
  --think-mode off \
  --query-analysis-think-mode on \
  --final-output-think-mode off \
  --verifier-think-mode on \
  "${LIMIT_ARGS[@]}" \
  "${OVERWRITE_ARGS[@]}"
