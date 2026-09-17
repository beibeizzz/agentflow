#!/usr/bin/env bash
set -euo pipefail

: "${PYTHON:?Set PYTHON to the runtime Python executable}"
project_root="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${project_root}"
export PYTHONPATH="${project_root}/src${PYTHONPATH:+:${PYTHONPATH}}"
export TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}"

bash scripts/runtime/preflight_unified.sh

: "${SANDBOX_IMAGE_MANIFEST:?Set SANDBOX_IMAGE_MANIFEST}"
: "${WIKIPEDIA_INDEX_REVISION:?Set WIKIPEDIA_INDEX_REVISION}"
: "${WIKIPEDIA_BENCHMARK_REPORT:?Set WIKIPEDIA_BENCHMARK_REPORT}"
reward_mode="${REWARD_MODE:-prm}"
process_revision="terminal-only"
case "${reward_mode}" in
  prm) process_revision="${PRM_REVISION:?Set PRM_REVISION}" ;;
  judge) process_revision="${JUDGE_REVISION:?Set JUDGE_REVISION}" ;;
esac
preflight_report="${PREFLIGHT_REPORT:-outputs/preflight/${reward_mode}/report.json}"
"${PYTHON}" scripts/runtime/verify_preflight_report.py \
  --report "${preflight_report}" \
  --reward-mode "${reward_mode}" \
  --config configs/train/unified_terminal_prm.yaml \
  --train-data data/verl/train_mixed.parquet \
  --private-data data/private/all.jsonl \
  --image-manifest "${SANDBOX_IMAGE_MANIFEST}" \
  --wikipedia-benchmark "${WIKIPEDIA_BENCHMARK_REPORT}" \
  --sandbox-image "${SANDBOX_IMAGE}" \
  --bigcodebench-image "${BIGCODEBENCH_IMAGE}" \
  --planner-revision "${PLANNER_REVISION}" \
  --frozen-revision "${FROZEN_MODEL_REVISION}" \
  --process-revision "${process_revision}" \
  --wikipedia-revision "${WIKIPEDIA_INDEX_REVISION}"

overrides=()
overrides+=(--override "agentflow.tools.sandbox_image=${SANDBOX_IMAGE}")
overrides+=(--override "agentflow.tools.bigcodebench_image=${BIGCODEBENCH_IMAGE}")
overrides+=(--override "agentflow.tools.bigcodebench_revision=${BIGCODEBENCH_REVISION}")
case "${REWARD_MODE:-prm}" in
  terminal)
    overrides+=(--override agentflow.lambda_process=0.0)
    overrides+=(--override agentflow.process_reward.mode=none)
    overrides+=(--override agentflow.process_reward.revision=terminal-only)
    ;;
  prm) ;;
  judge)
    overrides+=(--override agentflow.process_reward.mode=online_judge)
    overrides+=(--override agentflow.process_reward.base_url="${JUDGE_BASE_URL}")
    overrides+=(--override agentflow.process_reward.model="${JUDGE_MODEL}")
    overrides+=(--override agentflow.process_reward.revision="${JUDGE_REVISION}")
    overrides+=(--override agentflow.process_reward.cache_path="${JUDGE_CACHE_PATH:-outputs/judge/cache.jsonl}")
    ;;
esac

run_name="${RUN_NAME:-rtg_loo_${REWARD_MODE:-prm}}"
overrides+=(--override trainer.experiment_name="${run_name}")
overrides+=(--override trainer.rollout_data_dir="outputs/train/${run_name}/rollouts")
overrides+=(--override trainer.default_local_dir="outputs/train/${run_name}")

exec env CUDA_VISIBLE_DEVICES="${TRAIN_CUDA_VISIBLE_DEVICES:-0}" \
  "${PYTHON}" -m agentflow_rl.integrations.verl_main \
  --config configs/train/unified_terminal_prm.yaml \
  "${overrides[@]}"
