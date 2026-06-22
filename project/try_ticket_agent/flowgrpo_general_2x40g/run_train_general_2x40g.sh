#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export NO_PROXY="${NO_PROXY:-},127.0.0.1,localhost"
export no_proxy="${no_proxy:-},127.0.0.1,localhost"

PYTHON_BIN="${PYTHON:-python}"
CONFIG_FILE="${CONFIG_FILE:-try_ticket_agent/flowgrpo_general_2x40g/config_train_general_2x40g.yaml}"
MODEL_PATH="${MODEL_PATH:-/inspire/hdd/project/fdu-aidake-cfff/public/zhangjiangbei/agentflow/AgentFlow/agentflow0525/agentflow/model/Qwen/Qwen3-0.6B}"
TRAIN_FILE="${TRAIN_FILE:-try_ticket_agent/data/generated/train.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-try_ticket_agent/flowgrpo_general_2x40g/outputs/train_general_2x40g}"
FROZEN_BASE_URL="${FROZEN_BASE_URL:-http://127.0.0.1:8000/v1}"
FROZEN_MODEL="${FROZEN_MODEL:-Qwen3-0.6B}"
QUESTION_BATCH_SIZE="${QUESTION_BATCH_SIZE:-4}"
GROUP_SIZE="${GROUP_SIZE:-8}"
ROLLOUT_CONCURRENCY="${ROLLOUT_CONCURRENCY:-32}"
PLANNER_BATCH_SIZE="${PLANNER_BATCH_SIZE:-32}"
MAX_STEPS="${MAX_STEPS:-3}"
CLIP_RANGE_LOW="${CLIP_RANGE_LOW:-0.001}"
CLIP_RANGE_HIGH="${CLIP_RANGE_HIGH:-0.003}"
POLICY_EPOCHS="${POLICY_EPOCHS:-2}"
MAX_TRAIN_ITEMS="${MAX_TRAIN_ITEMS:-2500}"
EPOCHS="${EPOCHS:-1}"

[[ -f "$TRAIN_FILE" ]] || { echo "Training file does not exist: $TRAIN_FILE" >&2; exit 1; }
"$PYTHON_BIN" try_ticket_agent/flowgrpo_general_2x40g/train_ticket_gspo.py \
  --config "$CONFIG_FILE" --model-path "$MODEL_PATH" --train-file "$TRAIN_FILE" \
  --output-dir "$OUTPUT_DIR" --frozen-base-url "$FROZEN_BASE_URL" --frozen-model "$FROZEN_MODEL" \
  --question-batch-size "$QUESTION_BATCH_SIZE" --group-size "$GROUP_SIZE" \
  --rollout-concurrency "$ROLLOUT_CONCURRENCY" --planner-batch-size "$PLANNER_BATCH_SIZE" \
  --max-steps "$MAX_STEPS" --clip-range-low "$CLIP_RANGE_LOW" --clip-range-high "$CLIP_RANGE_HIGH" \
  --policy-epochs "$POLICY_EPOCHS" --max-train-items "$MAX_TRAIN_ITEMS" --epochs "$EPOCHS"
