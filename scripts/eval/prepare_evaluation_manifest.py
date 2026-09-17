from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq
import yaml

from agentflow_rl.utils.fingerprint import sha256_json
from agentflow_rl.evaluation import (
    ArtifactDigest,
    SharedEvaluationManifest,
    write_shared_manifest,
)
from agentflow_rl.evaluation.manifests import sha256_file


def resolve(base: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base / path).resolve()


def task_keys(paths: tuple[Path, ...]) -> tuple[tuple[str, str], ...]:
    values = []
    for path in paths:
        for row in pq.read_table(path, columns=["extra_info"]).to_pylist():
            public = json.loads(dict(row["extra_info"])["public_task"])
            values.append((str(public["task_name"]), str(public["task_id"])))
    if len(values) != len(set(values)):
        raise ValueError("evaluation task keys must be unique across input files")
    return tuple(values)


def main() -> int:
    parser = argparse.ArgumentParser(description="Freeze the shared active E0-E3 evaluation manifest")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config_path = args.config.resolve()
    values = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    base = config_path.parent
    inputs = tuple(resolve(base, value) for value in values.pop("input_paths"))
    private_records = resolve(base, values.pop("private_records_path"))
    manifest = SharedEvaluationManifest(
        **values,
        inputs=tuple(
            ArtifactDigest(name=path.name, sha256=sha256_file(path)) for path in inputs
        ),
        private_records=ArtifactDigest(
            name=private_records.name, sha256=sha256_file(private_records)
        ),
        task_keys_sha256=sha256_json(sorted(task_keys(inputs))),
    )
    frozen = write_shared_manifest(manifest, args.output)
    print(frozen.manifest_sha256)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
