#!/usr/bin/env bash
set -euo pipefail
python scripts/remote/check_frozen_server.py
export JAVA_TOOL_OPTIONS="${JAVA_TOOL_OPTIONS:--Xms1g -Xmx8g}"
python scripts/remote/check_research_backend.py \
  --index data/indexes/hotpotqa --min-documents 1000000 \
  --examples data/deepresearch/hotpot_fullwiki.jsonl \
  --examples data/deepresearch/hotpot_validation.jsonl
python scripts/remote/check_research_backend.py \
  --index data/indexes/2wiki --min-documents 1000 \
  --examples data/deepresearch/2wiki.jsonl \
  --examples data/deepresearch/2wiki_validation.jsonl
export NO_PROXY="${NO_PROXY:-127.0.0.1,localhost}"
export no_proxy="${no_proxy:-127.0.0.1,localhost}"

run_stage() {
  local stage="$1"
  local train_file="$2"
  local val_file="$3"
  local retrieval_mode="$4"
  local index_path="$5"
  local adapter_path="${6:-}"
  local output="outputs/deepresearch/${stage}"
  mkdir -p "${output}"
  export VERL_FILE_LOGGER_PATH="${output}/metrics.jsonl"
  local args=(
    --config configs/deepresearch/train.yaml
    --override "data.train_files=[${train_file}]"
    --override "data.val_files=[${val_file}]"
    --override "agentflow.deepresearch.retrieval_mode=${retrieval_mode}"
    --override "agentflow.deepresearch.index_path=${index_path}"
    --override "trainer.experiment_name=deepresearch_${stage}"
    --override "trainer.rollout_data_dir=${output}/rollouts"
    --override "trainer.validation_data_dir=${output}/validation"
    --override "trainer.default_local_dir=${output}"
    --override trainer.resume_mode=auto
  )
  if [[ -n "${adapter_path}" ]]; then
    args+=(--adapter-path "${adapter_path}")
  fi
  CUDA_VISIBLE_DEVICES=1 python -m agentflow_rl.verl.main "${args[@]}" "${@:7}"
}

run_stage hotpot_distractor data/verl/deepresearch/hotpot_distractor.parquet \
  data/verl/deepresearch/hotpot_validation.parquet local_context \
  data/indexes/hotpotqa "" "$@"
stage1_adapter="$(python scripts/find_latest_adapter.py outputs/deepresearch/hotpot_distractor)"
run_stage two_wiki data/verl/deepresearch/2wiki.parquet \
  data/verl/deepresearch/2wiki_validation.parquet global data/indexes/2wiki \
  "${stage1_adapter}" "$@"
stage2_adapter="$(python scripts/find_latest_adapter.py outputs/deepresearch/two_wiki)"
run_stage hotpot_fullwiki data/verl/deepresearch/hotpot_fullwiki.parquet \
  data/verl/deepresearch/hotpot_validation.parquet global data/indexes/hotpotqa \
  "${stage2_adapter}" "$@"
