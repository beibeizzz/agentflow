#!/usr/bin/env bash
set -euo pipefail
python scripts/remote/check_frozen_server.py
bash scripts/build_code_sandbox.sh
python scripts/remote/check_code_sandbox.py
ADAPTER_PATH="${ADAPTER_PATH:-$(python scripts/find_latest_adapter.py outputs/coding/train)}"
mkdir -p outputs/coding/eval
export VERL_FILE_LOGGER_PATH="${VERL_FILE_LOGGER_PATH:-outputs/coding/eval/metrics.jsonl}"
export NO_PROXY="${NO_PROXY:-127.0.0.1,localhost}"
export no_proxy="${no_proxy:-127.0.0.1,localhost}"
CUDA_VISIBLE_DEVICES=1 python -m agentflow_rl.verl.main --config configs/coding/eval.yaml --adapter-path "${ADAPTER_PATH}" "$@"
