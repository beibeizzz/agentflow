#!/usr/bin/env bash
set -euo pipefail
MODEL_PATH="${MODEL_PATH:-model/Qwen/Qwen3-8B}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Qwen3-8B}"
CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
  --model "${MODEL_PATH}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --dtype bfloat16 \
  --max-model-len "${MAX_MODEL_LEN:-8192}" \
  --max-num-seqs "${MAX_NUM_SEQS:-32}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.88}" \
  --enable-prefix-caching \
  --port "${PORT:-8000}"
