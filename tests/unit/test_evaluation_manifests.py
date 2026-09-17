from pathlib import Path

import pytest
import yaml

from pydantic import ValidationError

from agentflow_rl.utils.fingerprint import sha256_json
from agentflow_rl.evaluation import (
    ACTIVE_EVALUATION_CONDITIONS,
    ArtifactDigest,
    EvaluationCondition,
    SharedEvaluationManifest,
    assert_run_matches_shared_manifest,
    freeze_shared_manifest,
)
from agentflow_rl.evaluation.manifests import sha256_file


def manifest(input_path: Path, private_path: Path) -> SharedEvaluationManifest:
    return freeze_shared_manifest(
        SharedEvaluationManifest(
            evaluation_id="fixture",
            inputs=(ArtifactDigest(name=input_path.name, sha256=sha256_file(input_path)),),
            private_records=ArtifactDigest(
                name=private_path.name, sha256=sha256_file(private_path)
            ),
            task_keys_sha256=sha256_json([("aime", "aime-1")]),
            seeds=(1, 2),
            condition_planner_revisions={
                condition: f"{condition.value}-revision"
                for condition in ACTIVE_EVALUATION_CONDITIONS
            },
            planner_model="planner",
            frozen_model="frozen",
            frozen_revision="frozen-r1",
            prompt_revision="prompts-r1",
            evaluator_revision="evaluators-r1",
            tool_contract_revision="tools-r1",
            serper_revision="serper-r1",
            wikipedia_revision="wiki-r1",
            wikipedia_index_type="hnsw64",
            wikipedia_hnsw_m=64,
            wikipedia_hnsw_ef_search=256,
            wikipedia_index_sha256="1" * 64,
            wikipedia_corpus_sha256="2" * 64,
            wikipedia_arrow_fingerprint="arrow-r1",
            wikipedia_encoder_revision="e5-r1",
            wikipedia_benchmark_sha256="3" * 64,
            sandbox_image="sandbox@sha256:1",
            sandbox_service={"revision": "sandbox-service-r1", "timeout_s": 30.0},
            bigcodebench_image="bigcode@sha256:1",
            bigcodebench_revision="bigcode-r1",
            decoding={"temperature": 1.0, "top_p": 1.0},
            budgets=budgets(),
            execution={
                "max_concurrency": 4,
                "trajectory_timeout_s": 600.0,
                "http_timeout_s": 120.0,
            },
        )
    )


def settings() -> dict:
    return {
        "planner_model": "planner",
        "frozen_model": "frozen",
        "frozen_revision": "frozen-r1",
        "prompt_revision": "prompts-r1",
        "evaluator_revision": "evaluators-r1",
        "tool_contract_revision": "tools-r1",
        "serper_revision": "serper-r1",
        "wikipedia_revision": "wiki-r1",
        "wikipedia_index_type": "hnsw64",
        "wikipedia_hnsw_m": 64,
        "wikipedia_hnsw_ef_search": 256,
        "wikipedia_index_sha256": "1" * 64,
        "wikipedia_corpus_sha256": "2" * 64,
        "wikipedia_arrow_fingerprint": "arrow-r1",
        "wikipedia_encoder_revision": "e5-r1",
        "wikipedia_benchmark_sha256": "3" * 64,
        "sandbox_image": "sandbox@sha256:1",
        "sandbox_service": {"revision": "sandbox-service-r1", "timeout_s": 30.0},
        "bigcodebench_image": "bigcode@sha256:1",
        "bigcodebench_revision": "bigcode-r1",
        "decoding": {"temperature": 1.0, "top_p": 1.0},
        "budgets": budgets(),
        "execution": {"max_concurrency": 4, "trajectory_timeout_s": 600.0, "http_timeout_s": 120.0},
    }


def budgets() -> dict:
    return {
        "retrieval_read_top_k": 2,
        "retrieval_max_passages": 3,
        "retrieval_max_source_chars": 4000,
        "retrieval_read_timeout_s": 8.0,
        "max_turns": 5,
        "max_prompt_tokens": 4096,
        "max_output_tokens": 1024,
        "direct_max_output_tokens": 12288,
        "frozen_max_prompt_tokens": 4096,
        "frozen_generator_max_prompt_tokens": 8192,
        "frozen_max_output_tokens": {
            "query_analyzer": 1024,
            "executor": 1024,
            "verifier": 512,
            "generator": 2048,
            "base_generator": 768,
        },
    }


def test_run_settings_match_frozen_shared_manifest(tmp_path: Path) -> None:
    input_path = tmp_path / "test.parquet"
    private_path = tmp_path / "private.jsonl"
    input_path.write_bytes(b"public")
    private_path.write_bytes(b"private")
    value = manifest(input_path, private_path)
    assert_run_matches_shared_manifest(
        value,
        condition=EvaluationCondition.DIRECT,
        planner_revision="E0_direct-revision",
        input_paths=(input_path,),
        private_records_path=private_path,
        task_keys=(("aime", "aime-1"),),
        seeds=(1, 2),
        settings=settings(),
    )


def test_changed_shared_condition_is_rejected(tmp_path: Path) -> None:
    input_path = tmp_path / "test.parquet"
    private_path = tmp_path / "private.jsonl"
    input_path.write_bytes(b"public")
    private_path.write_bytes(b"private")
    changed = settings()
    changed["wikipedia_revision"] = "wiki-r2"
    with pytest.raises(ValueError, match="wikipedia_revision"):
        assert_run_matches_shared_manifest(
            manifest(input_path, private_path),
            condition=EvaluationCondition.DIRECT,
            planner_revision="E0_direct-revision",
            input_paths=(input_path,),
            private_records_path=private_path,
            task_keys=(("aime", "aime-1"),),
            seeds=(1, 2),
            settings=changed,
        )


def test_nested_budgets_round_trip_and_compare_as_json_payload(tmp_path: Path) -> None:
    input_path = tmp_path / "test.parquet"
    private_path = tmp_path / "private.jsonl"
    input_path.write_bytes(b"public")
    private_path.write_bytes(b"private")
    value = manifest(input_path, private_path)
    assert value.budgets.frozen_max_output_tokens.generator == 2048
    assert value.model_validate_json(value.model_dump_json()) == value
    assert_run_matches_shared_manifest(
        value,
        condition=EvaluationCondition.DIRECT,
        planner_revision="E0_direct-revision",
        input_paths=(input_path,),
        private_records_path=private_path,
        task_keys=(("aime", "aime-1"),),
        seeds=(1, 2),
        settings=settings(),
    )


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value["frozen_max_output_tokens"].pop("generator"),
        lambda value: value["frozen_max_output_tokens"].update({"judge": 128}),
        lambda value: value.update({"max_turns": 0}),
        lambda value: value.update({"retrieval_read_timeout_s": float("inf")}),
    ],
)
def test_nested_budgets_reject_missing_unknown_and_invalid_values(
    tmp_path: Path, mutation
) -> None:
    input_path = tmp_path / "test.parquet"
    private_path = tmp_path / "private.jsonl"
    input_path.write_bytes(b"public")
    private_path.write_bytes(b"private")
    value = budgets()
    mutation(value)
    with pytest.raises(ValidationError):
        SharedEvaluationManifest(
            **{
                **manifest(input_path, private_path).model_dump(
                    exclude={"budgets", "manifest_sha256"}
                ),
                "budgets": value,
            }
        )


def test_canonical_json_rejects_nan_and_unsupported_objects() -> None:
    with pytest.raises(ValueError):
        sha256_json({"value": float("nan")})
    with pytest.raises(TypeError):
        sha256_json({"value": object()})


def test_shared_manifest_example_matches_schema_v2() -> None:
    root = Path(__file__).resolve().parents[2]
    values = yaml.safe_load(
        (root / "configs/eval/shared_manifest.example.yaml").read_text(
            encoding="utf-8"
        )
    )
    values.pop("input_paths")
    values.pop("private_records_path")
    parsed = SharedEvaluationManifest(
        **values,
        inputs=(ArtifactDigest(name="test.parquet", sha256="4" * 64),),
        private_records=ArtifactDigest(name="private.jsonl", sha256="5" * 64),
        task_keys_sha256="6" * 64,
    )
    assert parsed.schema_version == "2"
    assert parsed.budgets.frozen_max_output_tokens.generator == 2048
