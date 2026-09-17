#!/usr/bin/env bash
set -euo pipefail

: "${PYTHON:?Set PYTHON to the runtime Python executable}"
: "${PRM_MODEL_PATH:?Set PRM_MODEL_PATH to the pinned Qwen3-0.6B PRM checkpoint}"

exec env \
  CUDA_VISIBLE_DEVICES="${PRM_CUDA_VISIBLE_DEVICES:-1}" \
  VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}" \
  VLLM_WORKER_MULTIPROC_METHOD=spawn \
  TOKENIZERS_PARALLELISM=false \
  "${PYTHON}" -m vllm.entrypoints.openai.api_server \
  --model "${PRM_MODEL_PATH}" \
  --served-model-name "${PRM_VLLM_MODEL:-AgentFlow-PRM}" \
  --runner pooling \
  --convert classify \
  --dtype bfloat16 \
  --max-model-len "${PRM_MAX_LENGTH:-8192}" \
  --max-num-seqs "${PRM_MAX_NUM_SEQS:-32}" \
  --max-num-batched-tokens "${PRM_MAX_BATCHED_TOKENS:-65536}" \
  --gpu-memory-utilization "${PRM_GPU_MEMORY_UTILIZATION:-0.10}" \
  --port "${PRM_VLLM_PORT:-8004}"
