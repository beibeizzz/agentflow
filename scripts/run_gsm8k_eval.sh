#!/usr/bin/env bash
set -euo pipefail
python scripts/remote/check_frozen_server.py --model Qwen3-0.6B
mkdir -p outputs/gsm8k/eval
export VERL_FILE_LOGGER_PATH="${VERL_FILE_LOGGER_PATH:-outputs/gsm8k/eval/metrics.jsonl}"
export NO_PROXY="${NO_PROXY:-127.0.0.1,localhost}"
export no_proxy="${no_proxy:-127.0.0.1,localhost}"
CUDA_VISIBLE_DEVICES=1 python -m agentflow_rl.verl.main --config configs/gsm8k/eval.yaml "$@"
