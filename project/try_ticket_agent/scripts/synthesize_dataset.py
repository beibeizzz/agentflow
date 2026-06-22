from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import types
from typing import Any

PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))
if "agentflow" not in sys.modules:
    agentflow_core = PROJECT_DIR / "agentflow" / "agentflow"
    agentflow_package = types.ModuleType("agentflow")
    agentflow_package.__path__ = [str(agentflow_core)]
    agentflow_package.__file__ = str(agentflow_core / "__init__.py")
    sys.modules["agentflow"] = agentflow_package

from try_ticket_agent.data_synthesis.api_client import DeepSeekClient
from try_ticket_agent.data_synthesis.pipeline import PipelinePaths, TicketSynthesisPipeline
from try_ticket_agent.data_synthesis.schemas import EpisodeBlueprint


def synthesis_paths_for_split(output_dir: Path, split: str) -> PipelinePaths:
    return PipelinePaths(
        dataset=output_dir / f"{split}.jsonl",
        rejected=output_dir / f"{split}.rejected.jsonl",
        progress=output_dir / f"{split}.progress.jsonl",
        summary=output_dir / f"{split}.summary.json",
    )


def load_blueprints(path: Path) -> list[EpisodeBlueprint]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(EpisodeBlueprint.from_dict(json.loads(line)))
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(f"Invalid blueprint at {path}:{line_number}") from exc
    return rows


def load_config(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to read synthesis config") from exc
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("synthesis config must be a YAML object")
    return payload


def write_synthesis_manifest(
    output_dir: Path, *, splits: list[str], summaries: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    hashes = {}
    counts = {}
    for split in splits:
        path = output_dir / f"{split}.jsonl"
        if not path.is_file():
            raise FileNotFoundError(path)
        hashes[split] = hashlib.sha256(path.read_bytes()).hexdigest()
        counts[split] = int(summaries[split]["accepted"])
    manifest = {"splits": splits, "counts": counts, "sha256": hashes, "summaries": summaries}
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "manifest.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, path)
    return manifest


def run_config(config: dict[str, Any]) -> dict[str, Any]:
    blueprints_dir = Path(config["blueprints_dir"])
    output_dir = Path(config["output_dir"])
    splits = list(config.get("splits", ["train", "validation", "test"]))
    client = DeepSeekClient(
        base_url=str(config.get("base_url", "https://api.deepseek.com")),
        rewrite_model=str(config.get("rewrite_model", "deepseek-v4-flash")),
        judge_model=str(config.get("judge_model", "deepseek-v4-pro")),
        max_transport_attempts=int(config.get("max_transport_attempts", 4)),
    )
    pipeline = TicketSynthesisPipeline(
        client,
        max_attempts=int(config.get("max_semantic_attempts", 3)),
        rewrite_temperature=float(config.get("rewrite_temperature", 0.3)),
    )
    summaries = {}
    for split in splits:
        blueprints = load_blueprints(blueprints_dir / f"{split}.jsonl")
        summaries[split] = pipeline.run(
            blueprints,
            paths=synthesis_paths_for_split(output_dir, split),
            resume=bool(config.get("resume", True)),
            concurrency=int(config.get("concurrency", 8)),
        )
    write_synthesis_manifest(output_dir, splits=splits, summaries=summaries)
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthesize validated Ticket requests with DeepSeek.")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    summaries = run_config(load_config(args.config))
    print(json.dumps(summaries, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
