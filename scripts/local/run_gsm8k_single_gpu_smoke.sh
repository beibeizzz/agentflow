#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/mnt/c/Users/north/Desktop/agentflow/project_alpha}"
PYTHON="${PYTHON:-/home/north/agentflow_local/.venv/bin/python}"
export AGENTFLOW_LOCAL_MODEL_PATH="${AGENTFLOW_LOCAL_MODEL_PATH:-/home/north/vllm_test/models/Qwen/Qwen3-0.6B}"
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export RAY_memory_usage_threshold="${RAY_memory_usage_threshold:-0.99}"
export NO_PROXY="${NO_PROXY:-127.0.0.1,localhost}"
export no_proxy="${no_proxy:-127.0.0.1,localhost}"

cd "${PROJECT_ROOT}"
test -x "${PYTHON}"
test -f "${AGENTFLOW_LOCAL_MODEL_PATH}/config.json"
run_id="${SMOKE_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
output_root="outputs/gsm8k/local_single_gpu_smoke/${run_id}"
mkdir -p "${output_root}"
echo "smoke_output=${output_root}"

"${PYTHON}" scripts/prepare_verl_data.py --task gsm8k

server_pid=""
cleanup() {
  if [[ -n "${server_pid}" ]]; then
    kill "${server_pid}" 2>/dev/null || true
    wait "${server_pid}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

if ! "${PYTHON}" scripts/remote/check_frozen_server.py \
  --url http://127.0.0.1:8001/v1 --model Qwen3-0.6B --timeout 1 >/dev/null 2>&1; then
  bash scripts/local/serve_gsm8k_frozen_single_gpu.sh \
    > "${output_root}/frozen_server.log" 2>&1 &
  server_pid="$!"
fi

"${PYTHON}" scripts/remote/check_frozen_server.py \
  --url http://127.0.0.1:8001/v1 --model Qwen3-0.6B --timeout 300
"${PYTHON}" scripts/local/probe_frozen_server.py

export VERL_FILE_LOGGER_PATH="${output_root}/metrics.jsonl"
CUDA_VISIBLE_DEVICES=0 "${PYTHON}" -m agentflow_rl.verl.main \
  --config configs/gsm8k/local_single_gpu_smoke.yaml \
  --override "trainer.default_local_dir=${output_root}" \
  --override "trainer.rollout_data_dir=${output_root}/rollouts" \
  --override "trainer.experiment_name=gsm8k_local_single_gpu_smoke_${run_id}" \
  "$@"
