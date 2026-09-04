from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Literal


TaskName = Literal["ticket", "gsm8k", "deepresearch", "coding"]


def _base_row(
    *, task: TaskName, prompt: str, ground_truth: str, extra_info: dict[str, Any]
) -> dict[str, Any]:
    return {
        "data_source": task,
        "prompt": [{"role": "user", "content": prompt}],
        "ability": task,
        "reward_model": {"style": "rule", "ground_truth": ground_truth},
        "extra_info": extra_info,
        "agent_name": f"agentflow_{task}",
    }


def ticket_to_verl_row(source: dict[str, Any], *, index: int) -> dict[str, Any]:
    if not source.get("episode_id") or not source.get("user_request"):
        raise ValueError("Ticket row requires episode_id and user_request")
    extra_info = dict(source)
    extra_info["index"] = int(index)
    return _base_row(
        task="ticket",
        prompt=str(source["user_request"]),
        ground_truth="1",
        extra_info=extra_info,
    )


def gsm8k_to_verl_row(source: dict[str, Any], *, index: int) -> dict[str, Any]:
    identity = source.get("episode_id", source.get("id", source.get("pid")))
    if identity is None or not source.get("question") or source.get("gold_answer") is None:
        raise ValueError("GSM8K row requires pid/id/episode_id, question, and gold_answer")
    extra_info = dict(source)
    extra_info["episode_id"] = str(identity)
    extra_info["gold_answer"] = str(source["gold_answer"])
    extra_info["index"] = int(index)
    return _base_row(
        task="gsm8k",
        prompt=str(source["question"]),
        ground_truth=str(source["gold_answer"]),
        extra_info=extra_info,
    )


def deepresearch_to_verl_row(source: dict[str, Any], *, index: int) -> dict[str, Any]:
    identity = source.get("episode_id", source.get("id", source.get("_id")))
    if identity is None or not source.get("question") or source.get("answer") is None:
        raise ValueError("DeepResearch row requires identity, question, and answer")
    extra_info = dict(source)
    extra_info["episode_id"] = str(identity)
    extra_info["index"] = int(index)
    return _base_row(
        task="deepresearch",
        prompt=str(source["question"]),
        ground_truth=str(source["answer"]),
        extra_info=extra_info,
    )


def coding_to_verl_row(source: dict[str, Any], *, index: int) -> dict[str, Any]:
    identity = source.get("episode_id", source.get("id", source.get("task_id")))
    question = source.get("question", source.get("prompt"))
    if identity is None or not question or not source.get("public_tests") or not source.get("hidden_tests"):
        raise ValueError("Coding row requires identity, question, public_tests, and hidden_tests")
    extra_info = dict(source)
    extra_info["episode_id"] = str(identity)
    extra_info["question"] = str(question)
    extra_info["index"] = int(index)
    return _base_row(
        task="coding",
        prompt=str(question),
        ground_truth="hidden_tests",
        extra_info=extra_info,
    )


def load_rows(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    if source.suffix == ".jsonl":
        return [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"JSON dataset must contain a list: {source}")
    return [dict(row) for row in payload]


def convert_rows(task: TaskName, rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    converters = {
        "ticket": ticket_to_verl_row,
        "gsm8k": gsm8k_to_verl_row,
        "deepresearch": deepresearch_to_verl_row,
        "coding": coding_to_verl_row,
    }
    converter = converters[task]
    return [converter(dict(row), index=index) for index, row in enumerate(rows)]


def write_parquet(rows: list[dict[str, Any]], path: str | Path) -> str:
    import pyarrow as pa
    import pyarrow.parquet as pq

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), target, compression="zstd")
    return hashlib.sha256(target.read_bytes()).hexdigest()


def convert_file(task: TaskName, source: str | Path, target: str | Path) -> dict[str, Any]:
    rows = convert_rows(task, load_rows(source))
    digest = write_parquet(rows, target)
    return {"task": task, "source": str(source), "target": str(target), "rows": len(rows), "sha256": digest}


__all__ = [
    "convert_file",
    "convert_rows",
    "coding_to_verl_row",
    "deepresearch_to_verl_row",
    "gsm8k_to_verl_row",
    "load_rows",
    "ticket_to_verl_row",
    "write_parquet",
]
