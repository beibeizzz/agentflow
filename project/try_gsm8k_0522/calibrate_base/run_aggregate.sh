#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

PYTHON_BIN="${PYTHON:-/inspire/hdd/project/fdu-aidake-cfff/public/zhangjiangbei/agentflow/AgentFlow/.venv/bin/python}"
DATA_FILE="${DATA_FILE:-try_gsm8k_0522/data/gsm8k_train.json}"
CALIBRATION_DIR="${CALIBRATION_DIR:-try_gsm8k_0522/calibrate_base/outputs/base_calibration}"
OUTPUT_DIR="${OUTPUT_DIR:-$CALIBRATION_DIR}"
ROLLOUTS_PER_QUESTION="${ROLLOUTS_PER_QUESTION:-8}"
START="${START:-0}"
LIMIT_ARGS=()
if [[ -n "${LIMIT:-}" ]]; then
  LIMIT_ARGS=(--limit "$LIMIT")
fi

"$PYTHON_BIN" try_gsm8k_0522/calibrate_base/aggregate_calibration.py \
  --data-file "$DATA_FILE" \
  --calibration-dir "$CALIBRATION_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --rollouts-per-question "$ROLLOUTS_PER_QUESTION" \
  --start "$START" \
  "${LIMIT_ARGS[@]}"
