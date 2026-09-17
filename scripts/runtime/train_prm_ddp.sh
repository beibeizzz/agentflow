#!/usr/bin/env bash
set -euo pipefail

# Offline PRM training owns both A800s. Stop the Planner, frozen-role, and PRM
# inference services before launching this job.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

python_bin="${PYTHON:-python}"
labels="${PRM_LABELS:-outputs/prm/formal/labels.jsonl}"
model="${PRM_BASE_MODEL:-/data/models/Qwen3-0.6B}"
output="${PRM_OUTPUT:-outputs/prm/formal/training-ddp}"
per_device_batch="${PRM_PER_DEVICE_BATCH:-16}"
gradient_accumulation="${PRM_GRADIENT_ACCUMULATION:-1}"
max_steps="${PRM_MAX_STEPS:--1}"

"${python_bin}" -m torch.distributed.run \
  --standalone \
  --nproc_per_node=2 \
  scripts/prm/train_prm.py \
  --labels "${labels}" \
  --model "${model}" \
  --output "${output}" \
  --max-length 8192 \
  --learning-rate 1e-5 \
  --epochs 1 \
  --max-steps "${max_steps}" \
  --batch-size "${per_device_batch}" \
  --gradient-accumulation "${gradient_accumulation}" \
  --gradient-checkpointing \
  --group-by-length \
  --expected-world-size 2 \
  --seed 42
