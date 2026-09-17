from __future__ import annotations

from pathlib import Path
from typing import Iterable

from .dataset import ProcessLabel
from agentflow_rl.rewards.rubric import validate_rubric_revision
from agentflow_rl.rewards.transition_view import PROCESS_VIEW_REVISION
from agentflow_rl.runtime.contracts import TaskName


def validate_label_protocol(labels: Iterable[ProcessLabel]) -> None:
    for item in labels:
        if item.transition.task.task_name not in {TaskName.AIME, TaskName.TWOWIKI, TaskName.TACO}:
            raise ValueError("Evaluation-only tasks are excluded from PRM training")
        if item.input_view_revision != PROCESS_VIEW_REVISION:
            raise ValueError("PRM labels use a different input view; regenerate labels for the current protocol")
        validate_rubric_revision(item.rubric_revision)


def load_process_labels(path: str | Path) -> list[ProcessLabel]:
    return [
        ProcessLabel.model_validate_json(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_process_labels(labels: Iterable[ProcessLabel], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="\n") as handle:
        for label in labels:
            handle.write(label.model_dump_json() + "\n")


def _average_ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        stop = start + 1
        while stop < len(order) and values[order[stop]] == values[order[start]]:
            stop += 1
        rank = (start + stop - 1) / 2.0
        for index in order[start:stop]:
            ranks[index] = rank
        start = stop
    return ranks


def _pearson(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or not left:
        raise ValueError("metric vectors must be non-empty and aligned")
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    numerator = sum(
        (x - left_mean) * (y - right_mean) for x, y in zip(left, right, strict=True)
    )
    left_norm = sum((value - left_mean) ** 2 for value in left) ** 0.5
    right_norm = sum((value - right_mean) ** 2 for value in right) ** 0.5
    denominator = left_norm * right_norm
    return numerator / denominator if denominator else 0.0


def regression_metrics(
    predictions: Iterable[float], targets: Iterable[float], *, calibration_bins: int = 10
) -> dict[str, float]:
    predicted = [min(1.0, max(0.0, float(value))) for value in predictions]
    expected = [float(value) for value in targets]
    if len(predicted) != len(expected) or not predicted or calibration_bins <= 0:
        raise ValueError("metric inputs must be non-empty, aligned, and have valid bins")
    mae = sum(abs(x - y) for x, y in zip(predicted, expected, strict=True)) / len(expected)
    spearman = _pearson(_average_ranks(predicted), _average_ranks(expected))
    calibration = 0.0
    for bin_index in range(calibration_bins):
        low = bin_index / calibration_bins
        high = (bin_index + 1) / calibration_bins
        indices = [
            index
            for index, value in enumerate(predicted)
            if low <= value < high or (bin_index == calibration_bins - 1 and value == 1.0)
        ]
        if indices:
            mean_prediction = sum(predicted[index] for index in indices) / len(indices)
            mean_target = sum(expected[index] for index in indices) / len(indices)
            calibration += len(indices) / len(expected) * abs(mean_prediction - mean_target)
    comparable = correct = 0
    for left in range(len(expected)):
        for right in range(left + 1, len(expected)):
            if expected[left] == expected[right]:
                continue
            comparable += 1
            correct += int(
                (predicted[left] - predicted[right])
                * (expected[left] - expected[right])
                > 0
            )
    return {
        "mae": mae,
        "spearman": spearman,
        "calibration_error": calibration,
        "pairwise_ranking_accuracy": correct / comparable if comparable else 0.0,
        "count": float(len(expected)),
    }


__all__ = ["load_process_labels", "regression_metrics", "write_process_labels", "validate_label_protocol"]
