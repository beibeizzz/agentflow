#!/usr/bin/env bash
set -euo pipefail

: "${PYTHON:?Set PYTHON to the runtime Python executable}"
: "${PLANNER_REVISION:?Set PLANNER_REVISION}"
: "${FROZEN_MODEL_REVISION:?Set FROZEN_MODEL_REVISION}"
: "${WIKIPEDIA_INDEX_REVISION:?Set WIKIPEDIA_INDEX_REVISION}"
: "${SANDBOX_IMAGE:?Set SANDBOX_IMAGE to the built public image}"
: "${BIGCODEBENCH_IMAGE:?Set BIGCODEBENCH_IMAGE to the built private image}"
: "${BIGCODEBENCH_REVISION:?Set BIGCODEBENCH_REVISION}"
: "${SANDBOX_IMAGE_MANIFEST:?Set SANDBOX_IMAGE_MANIFEST}"
: "${WIKIPEDIA_BENCHMARK_REPORT:?Set WIKIPEDIA_BENCHMARK_REPORT}"

project_root="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${project_root}"
export PYTHONPATH="${project_root}/src${PYTHONPATH:+:${PYTHONPATH}}"
export TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}"

reward_mode="${REWARD_MODE:-prm}"
output_root="${PREFLIGHT_ROOT:-outputs/preflight/${reward_mode}}"
test ! -e "${output_root}" || {
  echo "preflight output already exists: ${output_root}" >&2
  exit 1
}
mkdir -p "${output_root}"

bash scripts/runtime/preflight_unified.sh
"${PYTHON}" scripts/runtime/prepare_preflight_data.py \
  --input data/verl/train_mixed.parquet \
  --output "${output_root}/preflight_mixed.parquet" \
  --manifest "${output_root}/data_manifest.json" \
  --per-task 32

common_overrides=(
  --override "agentflow.tools.sandbox_image=${SANDBOX_IMAGE}"
  --override "agentflow.tools.bigcodebench_image=${BIGCODEBENCH_IMAGE}"
  --override "agentflow.tools.bigcodebench_revision=${BIGCODEBENCH_REVISION}"
)
reward_overrides=()
process_revision="terminal-only"
case "${reward_mode}" in
  terminal)
    reward_overrides+=(--override agentflow.lambda_process=0.0)
    reward_overrides+=(--override agentflow.process_reward.mode=none)
    reward_overrides+=(--override agentflow.process_reward.revision=terminal-only)
    ;;
  prm)
    process_revision="${PRM_REVISION:?Set PRM_REVISION}"
    ;;
  judge)
    : "${JUDGE_BASE_URL:?Set JUDGE_BASE_URL}"
    : "${JUDGE_MODEL:?Set JUDGE_MODEL}"
    process_revision="${JUDGE_REVISION:?Set JUDGE_REVISION}"
    reward_overrides+=(--override agentflow.process_reward.mode=online_judge)
    reward_overrides+=(--override "agentflow.process_reward.base_url=${JUDGE_BASE_URL}")
    reward_overrides+=(--override "agentflow.process_reward.model=${JUDGE_MODEL}")
    reward_overrides+=(--override "agentflow.process_reward.revision=${process_revision}")
    reward_overrides+=(--override "agentflow.process_reward.cache_path=${output_root}/judge_cache.jsonl")
    ;;
  *) echo "unsupported REWARD_MODE: ${reward_mode}" >&2; exit 1 ;;
esac

"${PYTHON}" scripts/runtime/probe_real_tools.py \
  --config configs/train/unified_terminal_prm.yaml \
  --output "${output_root}/tool_probe.json" \
  "${common_overrides[@]}" "${reward_overrides[@]}"

"${PYTHON}" scripts/runtime/snapshot_host_resources.py \
  --data-root "${project_root}" --output "${output_root}/host_initial.json"

train_dir="${output_root}/train"
initial_trajectories="${output_root}/trajectories_initial"
reload_trajectories="${output_root}/trajectories_reload"
initial_overrides=(
  "${common_overrides[@]}" "${reward_overrides[@]}"
  --override "data.train_files=[${output_root}/preflight_mixed.parquet]"
  --override "data.gen_batch_size=${PREFLIGHT_GEN_BATCH_SIZE:-8}"
  --override "actor_rollout_ref.rollout.n=${PREFLIGHT_GROUP_SIZE:-5}"
  --override trainer.total_training_steps=1
  --override agentflow.diagnostic_successful_update_limit=1
  --override trainer.save_freq=1
  --override trainer.test_freq=-1
  --override trainer.val_before_train=false
  --override "trainer.default_local_dir=${train_dir}"
  --override "trainer.rollout_data_dir=${output_root}/verl_rollouts_initial"
  --override "agentflow.trajectory_artifact_dir=${initial_trajectories}"
)
"${PYTHON}" scripts/runtime/monitor_gpu.py \
  --data-root "${project_root}" --output "${output_root}/gpu_initial.json" -- \
  env CUDA_VISIBLE_DEVICES="${TRAIN_CUDA_VISIBLE_DEVICES:-0}" \
  "${PYTHON}" -m agentflow_rl.integrations.verl_main \
  --config configs/train/unified_terminal_prm.yaml "${initial_overrides[@]}"

reload_overrides=(
  "${common_overrides[@]}" "${reward_overrides[@]}"
  --override "data.train_files=[${output_root}/preflight_mixed.parquet]"
  --override data.gen_batch_size=1
  --override data.train_batch_size=1
  --override actor_rollout_ref.actor.ppo_mini_batch_size=1
  --override actor_rollout_ref.rollout.n=2
  --override agentflow.dynamic_sampling.enabled=false
  --override trainer.total_training_steps=2
  --override agentflow.diagnostic_successful_update_limit=2
  --override trainer.save_freq=1
  --override trainer.test_freq=-1
  --override trainer.val_before_train=false
  --override trainer.resume_mode=auto
  --override "trainer.default_local_dir=${train_dir}"
  --override "trainer.rollout_data_dir=${output_root}/verl_rollouts_reload"
  --override "agentflow.trajectory_artifact_dir=${reload_trajectories}"
)
"${PYTHON}" scripts/runtime/monitor_gpu.py \
  --data-root "${project_root}" --output "${output_root}/gpu_reload.json" -- \
  env CUDA_VISIBLE_DEVICES="${TRAIN_CUDA_VISIBLE_DEVICES:-0}" \
  "${PYTHON}" -m agentflow_rl.integrations.verl_main \
  --config configs/train/unified_terminal_prm.yaml "${reload_overrides[@]}"

"${PYTHON}" scripts/runtime/snapshot_host_resources.py \
  --data-root "${project_root}" --output "${output_root}/host_final.json"

"${PYTHON}" scripts/runtime/collect_preflight_report.py \
  --reward-mode "${reward_mode}" \
  --config configs/train/unified_terminal_prm.yaml \
  --private-data data/private/all.jsonl \
  --data-manifest "${output_root}/data_manifest.json" \
  --image-manifest "${SANDBOX_IMAGE_MANIFEST}" \
  --tool-probe "${output_root}/tool_probe.json" \
  --wikipedia-benchmark "${WIKIPEDIA_BENCHMARK_REPORT}" \
  --gpu-report "${output_root}/gpu_initial.json" \
  --gpu-report "${output_root}/gpu_reload.json" \
  --host-report "${output_root}/host_initial.json" \
  --host-report "${output_root}/host_final.json" \
  --train-dir "${train_dir}" \
  --trajectory-dir "${initial_trajectories}" \
  --reload-trajectory-dir "${reload_trajectories}" \
  --planner-revision "${PLANNER_REVISION}" \
  --frozen-revision "${FROZEN_MODEL_REVISION}" \
  --process-revision "${process_revision}" \
  --wikipedia-revision "${WIKIPEDIA_INDEX_REVISION}" \
  --output "${output_root}/report.json"

echo "accepted real preflight: ${output_root}/report.json"
