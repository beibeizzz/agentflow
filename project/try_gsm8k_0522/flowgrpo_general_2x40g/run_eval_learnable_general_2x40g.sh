#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export NO_PROXY="${NO_PROXY:-},127.0.0.1,localhost"
export no_proxy="${no_proxy:-},127.0.0.1,localhost"

PYTHON_BIN="${PYTHON:-/inspire/hdd/project/fdu-aidake-cfff/public/zhangjiangbei/agentflow/AgentFlow/.venv/bin/python}"
CONFIG_FILE="${CONFIG_FILE:-try_gsm8k_0522/flowgrpo_general_2x40g/config_eval_learnable_general_2x40g.yaml}"
MODEL_PATH="${MODEL_PATH:-/inspire/hdd/project/fdu-aidake-cfff/public/zhangjiangbei/agentflow/AgentFlow/agentflow0525/agentflow/model/Qwen/Qwen3-0.6B}"
ADAPTER_PATH="${ADAPTER_PATH:-try_gsm8k_0522/flowgrpo_general_2x40g/outputs/train_general_2x40g/final_adapter}"
#ADAPTER_PATH="${ADAPTER_PATH:-false}"
FROZEN_BASE_URL="${FROZEN_BASE_URL:-http://127.0.0.1:8000/v1}"
FROZEN_MODEL="${FROZEN_MODEL:-Qwen3-0.6B}"
EVAL_FILE="${EVAL_FILE:-try_gsm8k_0522/data/gsm8k_train_learnable.json}"
EVAL_MODE="${EVAL_MODE:-adapter}"
case "${ADAPTER_PATH,,}" in
  false|0|none|no|off|"")
    EVAL_MODE="baseline"
    ;;
esac
OUTPUT_DIR="${OUTPUT_DIR:-try_gsm8k_0522/flowgrpo_general_2x40g/outputs/eval_learnable_${EVAL_MODE}}"
ROLLOUT_BACKEND="${ROLLOUT_BACKEND:-agentflow}"
MAX_EVAL_ITEMS="${MAX_EVAL_ITEMS:-1000000}"
MAX_STEPS="${MAX_STEPS:-3}"
THINK_MODE="${THINK_MODE:-off}"
QUERY_ANALYSIS_THINK_MODE="${QUERY_ANALYSIS_THINK_MODE:-on}"
FINAL_OUTPUT_THINK_MODE="${FINAL_OUTPUT_THINK_MODE:-off}"
VERIFIER_THINK_MODE="${VERIFIER_THINK_MODE:-on}"

echo "Using CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "Using EVAL_MODE=${EVAL_MODE}"
echo "Using CONFIG_FILE=${CONFIG_FILE}"
echo "Using MODEL_PATH=${MODEL_PATH}"
echo "Using ADAPTER_PATH=${ADAPTER_PATH}"
echo "Using FROZEN_BASE_URL=${FROZEN_BASE_URL}"
echo "Using FROZEN_MODEL=${FROZEN_MODEL}"
echo "Using EVAL_FILE=${EVAL_FILE}"
echo "Using OUTPUT_DIR=${OUTPUT_DIR}"
echo "Using MAX_EVAL_ITEMS=${MAX_EVAL_ITEMS}"

if [[ ! -f "$EVAL_FILE" ]]; then
  echo "Eval file does not exist: ${EVAL_FILE}" >&2
  exit 1
fi

if [[ "$EVAL_MODE" == "adapter" && ! -d "$ADAPTER_PATH" ]]; then
  echo "Adapter path does not exist: ${ADAPTER_PATH}" >&2
  exit 1
fi

"$PYTHON_BIN" try_gsm8k_0522/flowgrpo_general_2x40g/eval_general_learnable.py \
  --config "$CONFIG_FILE" \
  --model-path "$MODEL_PATH" \
  --adapter-path "$ADAPTER_PATH" \
  --eval-mode "$EVAL_MODE" \
  --eval-file "$EVAL_FILE" \
  --output-dir "$OUTPUT_DIR" \
  --frozen-base-url "$FROZEN_BASE_URL" \
  --frozen-model "$FROZEN_MODEL" \
  --rollout-backend "$ROLLOUT_BACKEND" \
  --max-eval-items "$MAX_EVAL_ITEMS" \
  --max-steps "$MAX_STEPS" \
  --think-mode "$THINK_MODE" \
  --query-analysis-think-mode "$QUERY_ANALYSIS_THINK_MODE" \
  --final-output-think-mode "$FINAL_OUTPUT_THINK_MODE" \
  --verifier-think-mode "$VERIFIER_THINK_MODE"
