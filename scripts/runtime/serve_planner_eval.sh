#!/usr/bin/env bash
set -euo pipefail

: "${PYTHON:?Set PYTHON to the runtime Python executable}"
: "${EVAL_PLANNER_MODEL_PATH:?Set EVAL_PLANNER_MODEL_PATH to the evaluated checkpoint}"

exec env \
  CUDA_VISIBLE_DEVICES="${EVAL_PLANNER_CUDA_VISIBLE_DEVICES:-0}" \
  VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}" \
  VLLM_WORKER_MULTIPROC_METHOD=spawn \
  TOKENIZERS_PARALLELISM=false \
  "${PYTHON}" -m vllm.entrypoints.openai.api_server \
  --model "${EVAL_PLANNER_MODEL_PATH}" \
  --served-model-name Qwen3-4B-Planner \
  --dtype bfloat16 \
  --max-model-len "${EVAL_PLANNER_MAX_MODEL_LEN:-18432}" \
  --max-num-seqs "${EVAL_PLANNER_MAX_NUM_SEQS:-8}" \
  --gpu-memory-utilization "${EVAL_PLANNER_GPU_MEMORY_UTILIZATION:-0.80}" \
  --enable-prefix-caching \
  --port "${EVAL_PLANNER_PORT:-8010}"
