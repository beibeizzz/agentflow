from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from agentflow_rl.data.fingerprint import sha256_json

from .runner import ACTIVE_EVALUATION_CONDITIONS, EvaluationCondition, EvaluationRecord


class StrictFrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ArtifactDigest(StrictFrozenModel):
    name: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class SharedEvaluationManifest(StrictFrozenModel):
    schema_version: str = "1"
    evaluation_id: str = Field(min_length=1)
    inputs: tuple[ArtifactDigest, ...]
    private_records: ArtifactDigest
    task_keys_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    seeds: tuple[int, ...]
    condition_planner_revisions: dict[EvaluationCondition, str]
    planner_model: str = Field(min_length=1)
    frozen_model: str = Field(min_length=1)
    frozen_revision: str = Field(min_length=1)
    prompt_revision: str = Field(min_length=1)
    evaluator_revision: str = Field(min_length=1)
    tool_contract_revision: str = Field(min_length=1)
    serper_revision: str = Field(min_length=1)
    wikipedia_revision: str = Field(min_length=1)
    wikipedia_index_type: str = Field(pattern=r"^hnsw64$")
    wikipedia_hnsw_m: int = Field(ge=64, le=64)
    wikipedia_hnsw_ef_search: int = Field(ge=256, le=256)
    wikipedia_index_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    wikipedia_corpus_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    wikipedia_arrow_fingerprint: str = Field(min_length=1)
    wikipedia_encoder_revision: str = Field(min_length=1)
    wikipedia_benchmark_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    sandbox_image: str = Field(min_length=1)
    bigcodebench_image: str = Field(min_length=1)
    bigcodebench_revision: str = Field(min_length=1)
    decoding: dict[str, float | int | bool]
    budgets: dict[str, float | int]
    execution: dict[str, float | int]
    manifest_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_complete_matrix(self) -> "SharedEvaluationManifest":
        if set(self.condition_planner_revisions) != set(ACTIVE_EVALUATION_CONDITIONS):
            raise ValueError("all active E0-E3 Planner revisions are required")
        if not self.inputs or not self.seeds or len(self.seeds) != len(set(self.seeds)):
            raise ValueError("evaluation inputs and unique seeds are required")
        if any(not revision for revision in self.condition_planner_revisions.values()):
            raise ValueError("Planner revisions must be non-empty")
        return self


class EvaluationRunManifest(StrictFrozenModel):
    schema_version: str = "1"
    shared_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    condition: EvaluationCondition
    planner_revision: str = Field(min_length=1)
    record_count: int = Field(ge=0)
    sample_keys_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    valid_sample_keys_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def freeze_shared_manifest(manifest: SharedEvaluationManifest) -> SharedEvaluationManifest:
    payload = manifest.model_dump(mode="json", exclude={"manifest_sha256"})
    return manifest.model_copy(update={"manifest_sha256": sha256_json(payload)})


def write_shared_manifest(
    manifest: SharedEvaluationManifest, path: str | Path
) -> SharedEvaluationManifest:
    frozen = freeze_shared_manifest(manifest)
    _write_json(Path(path), frozen.model_dump(mode="json"))
    return frozen


def read_shared_manifest(path: str | Path) -> SharedEvaluationManifest:
    manifest = SharedEvaluationManifest.model_validate_json(
        Path(path).read_text(encoding="utf-8")
    )
    if manifest.manifest_sha256 != freeze_shared_manifest(manifest).manifest_sha256:
        raise ValueError("shared evaluation manifest hash mismatch")
    return manifest


def assert_run_matches_shared_manifest(
    manifest: SharedEvaluationManifest,
    *,
    condition: EvaluationCondition,
    planner_revision: str,
    input_paths: tuple[Path, ...],
    private_records_path: Path,
    task_keys: tuple[tuple[str, str], ...],
    seeds: tuple[int, ...],
    settings: dict[str, Any],
) -> None:
    actual_inputs = tuple(
        ArtifactDigest(name=path.name, sha256=sha256_file(path)) for path in input_paths
    )
    actual_private = ArtifactDigest(
        name=private_records_path.name, sha256=sha256_file(private_records_path)
    )
    checks = {
        "inputs": (actual_inputs, manifest.inputs),
        "private records": (actual_private, manifest.private_records),
        "task keys": (sha256_json(sorted(task_keys)), manifest.task_keys_sha256),
        "seeds": (seeds, manifest.seeds),
        "Planner revision": (
            planner_revision,
            manifest.condition_planner_revisions[condition],
        ),
    }
    for key, expected in settings.items():
        checks[key] = (expected, getattr(manifest, key))
    mismatches = [name for name, (actual, expected) in checks.items() if actual != expected]
    if mismatches:
        raise ValueError(
            "evaluation run differs from the frozen shared manifest: "
            + ", ".join(mismatches)
        )


def build_run_manifest(
    manifest: SharedEvaluationManifest,
    *,
    condition: EvaluationCondition,
    planner_revision: str,
    records: list[EvaluationRecord],
) -> EvaluationRunManifest:
    if manifest.manifest_sha256 is None:
        raise ValueError("shared evaluation manifest must be frozen")
    keys = sorted((item.task_id, item.seed) for item in records)
    if any(item.condition != condition for item in records):
        raise ValueError("evaluation records must match the run condition")
    if len(keys) != len(set(keys)):
        raise ValueError("evaluation sample keys must be unique within a condition")
    valid_keys = sorted((item.task_id, item.seed) for item in records if item.valid)
    return EvaluationRunManifest(
        shared_manifest_sha256=manifest.manifest_sha256,
        condition=condition,
        planner_revision=planner_revision,
        record_count=len(records),
        sample_keys_sha256=sha256_json(keys),
        valid_sample_keys_sha256=sha256_json(valid_keys),
    )


def write_run_manifest(manifest: EvaluationRunManifest, path: str | Path) -> None:
    _write_json(Path(path), manifest.model_dump(mode="json"))


def read_run_manifest(path: str | Path) -> EvaluationRunManifest:
    return EvaluationRunManifest.model_validate_json(Path(path).read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


__all__ = [
    "ArtifactDigest",
    "EvaluationRunManifest",
    "SharedEvaluationManifest",
    "assert_run_matches_shared_manifest",
    "build_run_manifest",
    "freeze_shared_manifest",
    "read_run_manifest",
    "read_shared_manifest",
    "sha256_file",
    "write_run_manifest",
    "write_shared_manifest",
]
