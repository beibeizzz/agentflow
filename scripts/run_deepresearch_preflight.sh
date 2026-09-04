#!/usr/bin/env bash
set -euo pipefail
python scripts/remote/check_frozen_server.py
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
OUTPUT_ROOT="${PREFLIGHT_OUTPUT_ROOT:-outputs/deepresearch/preflight/${RUN_ID}}"
mkdir -p "${OUTPUT_ROOT}"
export VERL_FILE_LOGGER_PATH="${VERL_FILE_LOGGER_PATH:-${OUTPUT_ROOT}/metrics.jsonl}"
export NO_PROXY="${NO_PROXY:-127.0.0.1,localhost}"
export no_proxy="${no_proxy:-127.0.0.1,localhost}"
CUDA_VISIBLE_DEVICES=1 python -m agentflow_rl.verl.main \
  --config configs/deepresearch/preflight.yaml \
  --override "trainer.rollout_data_dir=${OUTPUT_ROOT}/rollouts" \
  --override "trainer.default_local_dir=${OUTPUT_ROOT}" \
  "$@"
python scripts/remote/check_preflight_outputs.py --root "${OUTPUT_ROOT}"
