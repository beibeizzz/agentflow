#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
PYTHON_BIN="${PYTHON:-python}"
CONFIG_FILE="${CONFIG_FILE:-try_ticket_agent/flowgrpo_general_2x40g/config_eval_learnable_general_2x40g.yaml}"
EVAL_MODE="${EVAL_MODE:-adapter}"
ADAPTER_PATH="${ADAPTER_PATH:-try_ticket_agent/flowgrpo_general_2x40g/outputs/train_general_2x40g/final_adapter}"

if [[ "$EVAL_MODE" == "adapter" && ! -d "$ADAPTER_PATH" ]]; then
  echo "Adapter path does not exist: $ADAPTER_PATH" >&2
  exit 1
fi
"$PYTHON_BIN" try_ticket_agent/flowgrpo_general_2x40g/eval_ticket_agent.py \
  --config "$CONFIG_FILE" --eval-mode "$EVAL_MODE" --adapter-path "$ADAPTER_PATH"
