#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

if [[ -z "${PYTHON:-}" ]]; then
  if [[ -x "/home/north/vllm_test/.venv/bin/python" ]]; then
    PYTHON="/home/north/vllm_test/.venv/bin/python"
  else
    PYTHON="python"
  fi
fi

cd "$PROJECT_ROOT"
"$PYTHON" try_gsm8k_0522/flowgrpo/train_flowgrpo.py \
  --config try_gsm8k_0522/flowgrpo/config_smoke.yaml \
  "$@"

