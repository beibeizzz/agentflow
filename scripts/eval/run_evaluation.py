from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path

import pyarrow.parquet as pq
from openai import AsyncOpenAI
from transformers import AutoTokenizer

from agentflow_rl.backends.frozen_llm import OpenAICompatibleDirectGateway
from agentflow_rl.evaluation import (
    ACTIVE_EVALUATION_CONDITIONS,
    EvaluationCondition,
    assert_run_matches_shared_manifest,
    build_run_manifest,
    evaluate_agentflow,
    evaluate_direct,
    read_shared_manifest,
    write_evaluation,
    write_run_manifest,
    write_shared_manifest,
)
from agentflow_rl.integrations.bootstrap import build_runtime_bundle
from agentflow_rl.integrations.planner import OpenAICompatiblePlannerGateway
from agentflow_rl.integrations.verl_agent_loop import task_envelope_from_extra_info
from agentflow_rl.roles.prompts import PROMPT_REVISION
from agentflow_rl.tasks import TERMINAL_EVALUATOR_REVISION
from agentflow_rl.tools import TOOL_CONTRACT_REVISION


def load_envelopes(paths: list[Path]):
    envelopes = []
    for path in paths:
        for row in pq.read_table(path, columns=["extra_info"]).to_pylist():
            envelopes.append(task_envelope_from_extra_info(dict(row["extra_info"])))
    return tuple(envelopes)


def runtime_config(args: argparse.Namespace) -> dict:
    return {
        "data": {"max_prompt_length": args.max_prompt_tokens},
        "agentflow": {
            "environment_mode": "formal",
            "frozen_base_url": args.frozen_base_url,
            "frozen_model": args.frozen_model,
            "frozen_revision": args.frozen_revision,
            "frozen_max_input_tokens": args.frozen_max_prompt_tokens,
            "role_max_input_tokens": {
                "query_analyzer": args.frozen_max_prompt_tokens,
                "executor": args.frozen_max_prompt_tokens,
                "verifier": args.frozen_max_prompt_tokens,
                "generator": args.generator_max_prompt_tokens,
            },
            "private_records_path": str(args.private_records),
            "max_turns": args.max_turns,
            "role_max_tokens": {
                "query_analyzer": args.query_analyzer_max_tokens,
                "executor": args.executor_max_tokens,
                "verifier": args.verifier_max_tokens,
                "generator": args.generator_max_tokens,
            },
            "query_analyzer_cache": {
                "enabled": True,
                "path": str(args.output / "query_analyzer_cache.sqlite3"),
                "namespace": "query-analyzer-v1",
                "lock_timeout_s": 120.0,
            },
            "trajectory_timeout_s": args.trajectory_timeout_s,
            "http_timeout_s": args.http_timeout_s,
            "process_reward": {"mode": "none"},
            "tools": {
                "retrieval_read_top_k": args.retrieval_read_top_k,
                "base_generator_max_tokens": args.base_generator_max_tokens,
                "retrieval_max_passages": args.retrieval_max_passages,
                "retrieval_max_source_chars": args.retrieval_max_source_chars,
                "retrieval_read_timeout_s": args.retrieval_read_timeout_s,
                "sandbox_image": args.sandbox_image,
                "bigcodebench_image": args.bigcodebench_image,
                "bigcodebench_revision": args.bigcodebench_revision,
                "serper_key_env": "SERPER_API_KEY",
                "wikipedia_service_url": args.wikipedia_service_url,
                "wikipedia_revision": args.wikipedia_revision,
                "serper_cache_path": str(args.serper_cache),
                "serper_revision": args.serper_revision,
            },
        },
    }


async def run(args: argparse.Namespace) -> None:
    envelopes = load_envelopes(args.input)
    condition = EvaluationCondition(args.condition)
    shared_manifest = read_shared_manifest(args.shared_manifest)
    assert_run_matches_shared_manifest(
        shared_manifest,
        condition=condition,
        planner_revision=args.planner_revision,
        input_paths=tuple(args.input),
        private_records_path=args.private_records,
        task_keys=tuple(
            (envelope.public.task_name.value, envelope.public.task_id)
            for envelope in envelopes
        ),
        seeds=tuple(args.seed),
        settings={
            "planner_model": args.planner_model,
            "frozen_model": args.frozen_model,
            "frozen_revision": args.frozen_revision,
            "prompt_revision": PROMPT_REVISION,
            "evaluator_revision": TERMINAL_EVALUATOR_REVISION,
            "tool_contract_revision": TOOL_CONTRACT_REVISION,
            "serper_revision": args.serper_revision,
            "wikipedia_revision": args.wikipedia_revision,
            "wikipedia_index_type": args.wikipedia_index_type,
            "wikipedia_hnsw_m": args.wikipedia_hnsw_m,
            "wikipedia_hnsw_ef_search": args.wikipedia_hnsw_ef_search,
            "wikipedia_index_sha256": args.wikipedia_index_sha256,
            "wikipedia_corpus_sha256": args.wikipedia_corpus_sha256,
            "wikipedia_arrow_fingerprint": args.wikipedia_arrow_fingerprint,
            "wikipedia_encoder_revision": args.wikipedia_encoder_revision,
            "wikipedia_benchmark_sha256": args.wikipedia_benchmark_sha256,
            "sandbox_image": args.sandbox_image,
            "bigcodebench_image": args.bigcodebench_image,
            "bigcodebench_revision": args.bigcodebench_revision,
            "decoding": {
                "temperature": args.temperature,
                "top_p": args.top_p,
                "top_k": args.top_k,
                "direct_enable_thinking": args.direct_enable_thinking,
            },
            "budgets": {
                "retrieval_read_top_k": args.retrieval_read_top_k,
                "retrieval_max_passages": args.retrieval_max_passages,
                "retrieval_max_source_chars": args.retrieval_max_source_chars,
                "retrieval_read_timeout_s": args.retrieval_read_timeout_s,
                "max_turns": args.max_turns,
                "max_prompt_tokens": args.max_prompt_tokens,
                "max_output_tokens": args.max_tokens,
                "direct_max_output_tokens": args.direct_max_tokens,
                "frozen_max_prompt_tokens": args.frozen_max_prompt_tokens,
                "frozen_generator_max_prompt_tokens": args.generator_max_prompt_tokens,
                "frozen_max_output_tokens": {
                    "query_analyzer": args.query_analyzer_max_tokens,
                    "executor": args.executor_max_tokens,
                    "verifier": args.verifier_max_tokens,
                    "generator": args.generator_max_tokens,
                    "base_generator": args.base_generator_max_tokens,
                },
            },
            "execution": {
                "max_concurrency": args.max_concurrency,
                "trajectory_timeout_s": args.trajectory_timeout_s,
                "http_timeout_s": args.http_timeout_s,
            },
        },
    )
    tokenizer = AutoTokenizer.from_pretrained(args.planner_tokenizer, use_fast=True)
    planner_client = AsyncOpenAI(
        base_url=args.planner_base_url,
        api_key="not-required",
        timeout=args.http_timeout_s,
    )
    planner = OpenAICompatiblePlannerGateway(
        planner_client,
        tokenizer,
        model=args.planner_model,
        revision=args.planner_revision,
        max_prompt_tokens=args.max_prompt_tokens,
    )
    bundle = build_runtime_bundle(runtime_config(args), planner=planner)
    if condition is EvaluationCondition.DIRECT:
        direct = OpenAICompatibleDirectGateway(
            planner_client,
            tokenizer=tokenizer,
            model=args.planner_model,
            revision=args.planner_revision,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            max_input_tokens=args.max_prompt_tokens,
            enable_thinking=args.direct_enable_thinking,
        )
        records = await evaluate_direct(
            generator=direct,
            tasks=bundle.loop.tasks,
            envelopes=envelopes,
            seeds=args.seed,
            max_tokens=args.direct_max_tokens,
            max_concurrency=args.max_concurrency,
            trajectory_timeout_s=args.trajectory_timeout_s,
        )
    else:
        records = await evaluate_agentflow(
            loop=bundle.loop,
            envelopes=envelopes,
            condition=condition,
            model_revision=args.planner_revision,
            seeds=args.seed,
            sampling_params={
                "temperature": args.temperature,
                "top_p": args.top_p,
                "top_k": args.top_k,
                "max_tokens": args.max_tokens,
            },
            max_concurrency=args.max_concurrency,
            trajectory_timeout_s=args.trajectory_timeout_s,
        )
    expected_sample_keys = {
        (envelope.public.task_id, seed)
        for envelope in envelopes
        for seed in args.seed
    }
    actual_sample_keys = {(record.task_id, record.seed) for record in records}
    if (
        len(records) != len(expected_sample_keys)
        or actual_sample_keys != expected_sample_keys
    ):
        raise RuntimeError("evaluation output does not cover the frozen task/seed matrix")
    args.output.mkdir(parents=True, exist_ok=True)
    write_evaluation(
        records,
        samples_path=args.output / "eval_samples.jsonl",
        metrics_path=args.output / "eval_metrics.json",
    )
    write_shared_manifest(
        shared_manifest, args.output / "evaluation_shared_manifest.json"
    )
    write_run_manifest(
        build_run_manifest(
            shared_manifest,
            condition=condition,
            planner_revision=args.planner_revision,
            records=records,
        ),
        args.output / "evaluation_run_manifest.json",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one active AgentFlow E0-E3 condition")
    parser.add_argument(
        "--condition",
        choices=[item.value for item in ACTIVE_EVALUATION_CONDITIONS],
        required=True,
    )
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--private-records", type=Path, required=True)
    parser.add_argument("--shared-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--planner-base-url", default="http://127.0.0.1:8010/v1")
    parser.add_argument("--planner-model", default="Qwen3-4B-Planner")
    parser.add_argument("--planner-tokenizer", required=True)
    parser.add_argument("--planner-revision", required=True)
    parser.add_argument("--frozen-base-url", default="http://127.0.0.1:8001/v1")
    parser.add_argument("--frozen-model", default="Qwen3-8B")
    parser.add_argument(
        "--frozen-revision",
        default=os.environ.get("FROZEN_MODEL_REVISION", "unresolved"),
    )
    parser.add_argument("--serper-revision", default="serper-search-v1")
    parser.add_argument("--wikipedia-service-url", default="http://127.0.0.1:8002")
    parser.add_argument("--wikipedia-revision", required=True)
    parser.add_argument("--wikipedia-index-type", default="hnsw64")
    parser.add_argument("--wikipedia-hnsw-m", type=int, default=64)
    parser.add_argument("--wikipedia-hnsw-ef-search", type=int, default=256)
    parser.add_argument(
        "--wikipedia-index-sha256",
        default=os.environ.get("WIKIPEDIA_INDEX_SHA256", "unresolved"),
    )
    parser.add_argument(
        "--wikipedia-corpus-sha256",
        default=os.environ.get("WIKIPEDIA_CORPUS_SHA256", "unresolved"),
    )
    parser.add_argument(
        "--wikipedia-arrow-fingerprint",
        default=os.environ.get("WIKIPEDIA_ARROW_FINGERPRINT", "unresolved"),
    )
    parser.add_argument(
        "--wikipedia-encoder-revision",
        default=os.environ.get("WIKIPEDIA_E5_REVISION", "unresolved"),
    )
    parser.add_argument(
        "--wikipedia-benchmark-sha256",
        default=os.environ.get("WIKIPEDIA_BENCHMARK_SHA256", "unresolved"),
    )
    parser.add_argument("--serper-cache", type=Path, required=True)
    parser.add_argument("--sandbox-image", required=True)
    parser.add_argument("--bigcodebench-image", required=True)
    parser.add_argument("--bigcodebench-revision", required=True)
    parser.add_argument("--seed", type=int, action="append", default=[])
    parser.add_argument("--max-concurrency", type=int, default=8)
    parser.add_argument("--max-turns", type=int, default=5)
    parser.add_argument("--max-prompt-tokens", type=int, default=4096)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--direct-max-tokens", type=int, default=12288)
    parser.add_argument(
        "--direct-enable-thinking",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--frozen-max-prompt-tokens", type=int, default=4096)
    parser.add_argument("--generator-max-prompt-tokens", type=int, default=8192)
    parser.add_argument("--query-analyzer-max-tokens", type=int, default=1024)
    parser.add_argument("--executor-max-tokens", type=int, default=1024)
    parser.add_argument("--verifier-max-tokens", type=int, default=512)
    parser.add_argument("--generator-max-tokens", type=int, default=2048)
    parser.add_argument("--base-generator-max-tokens", type=int, default=768)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--trajectory-timeout-s", type=float, default=600.0)
    parser.add_argument("--http-timeout-s", type=float, default=120.0)
    parser.add_argument("--retrieval-read-top-k", type=int, default=2)
    parser.add_argument("--retrieval-max-passages", type=int, default=3)
    parser.add_argument("--retrieval-max-source-chars", type=int, default=4000)
    parser.add_argument("--retrieval-read-timeout-s", type=float, default=8.0)
    args = parser.parse_args()
    if not args.seed:
        args.seed = [1]
    return args


def main() -> int:
    asyncio.run(run(parse_args()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
