#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-/home/north/agentflow_local/.venv/bin/python}"
MODEL_PATH="${AGENTFLOW_LOCAL_MODEL_PATH:-/home/north/vllm_test/models/Qwen/Qwen3-0.6B}"

exec env CUDA_VISIBLE_DEVICES=0 VLLM_WORKER_MULTIPROC_METHOD=spawn \
  "${PYTHON}" -m vllm.entrypoints.openai.api_server \
  --model "${MODEL_PATH}" \
  --served-model-name Qwen3-0.6B \
  --dtype bfloat16 \
  --max-model-len 2048 \
  --max-num-seqs 4 \
  --gpu-memory-utilization "${FROZEN_GPU_MEMORY_UTILIZATION:-0.20}" \
  --enforce-eager \
  --port 8001
