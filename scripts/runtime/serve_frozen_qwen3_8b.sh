#!/usr/bin/env bash
set -euo pipefail

: "${PYTHON:?Set PYTHON to the runtime Python executable}"
: "${FROZEN_MODEL_PATH:?Set FROZEN_MODEL_PATH to the pinned Qwen3-8B directory}"

exec env \
  CUDA_VISIBLE_DEVICES="${FROZEN_CUDA_VISIBLE_DEVICES:-1}" \
  VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}" \
  VLLM_WORKER_MULTIPROC_METHOD=spawn \
  TOKENIZERS_PARALLELISM=false \
  "${PYTHON}" -m vllm.entrypoints.openai.api_server \
  --model "${FROZEN_MODEL_PATH}" \
  --served-model-name Qwen3-8B \
  --dtype bfloat16 \
  --max-model-len "${FROZEN_MAX_MODEL_LEN:-10240}" \
  --max-num-seqs "${FROZEN_MAX_NUM_SEQS:-64}" \
  --gpu-memory-utilization "${FROZEN_GPU_MEMORY_UTILIZATION:-0.65}" \
  --enable-prefix-caching \
  --port "${FROZEN_PORT:-8001}"
