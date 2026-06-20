#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GSM8K_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

if [[ -z "${PYTHON:-}" ]]; then
  if [[ -x "/home/north/vllm_test/.venv/bin/python" ]]; then
    PYTHON="/home/north/vllm_test/.venv/bin/python"
  else
    PYTHON="python"
  fi
fi

cd "$GSM8K_DIR"
"$PYTHON" prepare_gsm8k_json.py --smoke-size "${SMOKE_SIZE:-50}"
"$PYTHON" -m flowgrpo.data \
  --input "data/gsm8k_smoke_${SMOKE_SIZE:-50}.json" \
  --train-output "data/train/gsm8k_smoke_train.parquet" \
  --val-output "data/val/gsm8k_smoke_val.parquet" \
  --val-size "${VAL_SIZE:-8}"

