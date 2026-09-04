#!/usr/bin/env bash
set -euo pipefail
python scripts/remote/check_frozen_server.py
export JAVA_TOOL_OPTIONS="${JAVA_TOOL_OPTIONS:--Xms1g -Xmx8g}"
export NO_PROXY="${NO_PROXY:-127.0.0.1,localhost}"
export no_proxy="${no_proxy:-127.0.0.1,localhost}"

run_baseline() {
  local benchmark="$1"
  local split="$2"
  local index_path="$3"
  local min_documents="$4"
  local output="outputs/deepresearch/baseline/${benchmark}"
  python scripts/remote/check_research_backend.py \
    --index "${index_path}" --min-documents "${min_documents}" \
    --examples "data/deepresearch/${split}.jsonl"
  mkdir -p "${output}"
  export VERL_FILE_LOGGER_PATH="${output}/metrics.jsonl"
  CUDA_VISIBLE_DEVICES=1 python -m agentflow_rl.verl.main \
    --config configs/deepresearch/baseline.yaml \
    --override "data.val_files=[data/verl/deepresearch/${split}.parquet]" \
    --override "agentflow.deepresearch.index_path=${index_path}" \
    --override "trainer.experiment_name=deepresearch_baseline_${benchmark}" \
    --override "trainer.validation_data_dir=${output}/validation" \
    --override "trainer.default_local_dir=${output}" \
    "${@:5}"
}

run_baseline hotpotqa hotpot_validation data/indexes/hotpotqa 1000000 "$@"
run_baseline 2wiki 2wiki_validation data/indexes/2wiki 1000 "$@"
