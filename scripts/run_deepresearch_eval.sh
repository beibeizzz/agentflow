#!/usr/bin/env bash
set -euo pipefail
python scripts/remote/check_frozen_server.py
export JAVA_TOOL_OPTIONS="${JAVA_TOOL_OPTIONS:--Xms1g -Xmx8g}"
ADAPTER_PATH="${ADAPTER_PATH:-$(python scripts/find_latest_adapter.py outputs/deepresearch/hotpot_fullwiki)}"
export NO_PROXY="${NO_PROXY:-127.0.0.1,localhost}"
export no_proxy="${no_proxy:-127.0.0.1,localhost}"

run_eval() {
  local benchmark="$1"
  local split="$2"
  local index_path="$3"
  local min_documents="$4"
  local output="outputs/deepresearch/eval/${benchmark}"
  python scripts/remote/check_research_backend.py \
    --index "${index_path}" --min-documents "${min_documents}" \
    --examples "data/deepresearch/${split}.jsonl"
  mkdir -p "${output}"
  export VERL_FILE_LOGGER_PATH="${output}/metrics.jsonl"
  CUDA_VISIBLE_DEVICES=1 python -m agentflow_rl.verl.main \
    --config configs/deepresearch/eval.yaml \
    --adapter-path "${ADAPTER_PATH}" \
    --override "data.val_files=[data/verl/deepresearch/${split}.parquet]" \
    --override "agentflow.deepresearch.index_path=${index_path}" \
    --override "trainer.experiment_name=deepresearch_eval_${benchmark}" \
    --override "trainer.validation_data_dir=${output}/validation" \
    --override "trainer.default_local_dir=${output}" \
    "${@:5}"
}

run_eval hotpotqa hotpot_test data/indexes/hotpotqa 1000000 "$@"
run_eval 2wiki 2wiki_test data/indexes/2wiki 1000 "$@"
