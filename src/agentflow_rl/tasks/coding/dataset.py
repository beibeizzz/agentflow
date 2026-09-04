from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from .schemas import CodeExample, CodeTest, split_tests


UNSUPPORTED_PATTERNS = re.compile(r"\b(interactive|special judge|output any|see image|see figure)\b", re.IGNORECASE)


def _stdio_text(value: Any) -> str:
    if isinstance(value, list):
        return "\n".join(str(item) for item in value)
    return str(value)


def parse_taco_tests(input_output: str | dict[str, Any]) -> list[CodeTest]:
    payload = json.loads(input_output) if isinstance(input_output, str) else input_output
    inputs = payload.get("inputs", ())
    outputs = payload.get("outputs", ())
    if len(inputs) != len(outputs):
        raise ValueError("TACO inputs and outputs must align")
    fn_name = payload.get("fn_name")
    if fn_name:
        return [
            CodeTest(fn_name=str(fn_name), args=arguments, expected=expected)
            for arguments, expected in zip(inputs, outputs, strict=True)
        ]
    return [
        CodeTest(stdin=_stdio_text(stdin), expected_stdout=_stdio_text(expected))
        for stdin, expected in zip(inputs, outputs, strict=True)
    ]


def problem_fingerprint(row: dict[str, Any]) -> str:
    question = " ".join(str(row.get("question", "")).lower().split())
    tests = row.get("input_output", row.get("tests", ""))
    if isinstance(tests, str):
        try:
            tests = json.loads(tests)
        except json.JSONDecodeError:
            pass
    canonical_tests = json.dumps(tests, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(f"{question}\n{canonical_tests}".encode("utf-8")).hexdigest()


def question_fingerprint(row: dict[str, Any]) -> str:
    question = " ".join(str(row.get("question", "")).lower().split())
    return hashlib.sha256(question.encode("utf-8")).hexdigest()


def split_verified_rows(
    rows: list[dict[str, Any]],
    *,
    validation_fraction: float,
    test_fraction: float,
    seed: int,
) -> dict[str, list[dict[str, Any]]]:
    if validation_fraction <= 0 or test_fraction <= 0:
        raise ValueError("validation and test fractions must be positive")
    if validation_fraction + test_fraction >= 1:
        raise ValueError("validation and test fractions must sum to less than one")
    ranked = sorted(
        rows,
        key=lambda row: hashlib.sha256(
            f"{seed}:{row['metadata']['question_fingerprint']}".encode("utf-8")
        ).digest(),
    )
    validation_count = max(1, round(len(ranked) * validation_fraction))
    test_count = max(1, round(len(ranked) * test_fraction))
    if validation_count + test_count >= len(ranked):
        raise ValueError("coding source is too small for train/validation/test splitting")
    train_end = len(ranked) - validation_count - test_count
    validation_end = train_end + validation_count
    return {
        "train": ranked[:train_end],
        "validation": ranked[train_end:validation_end],
        "test": ranked[validation_end:],
    }


def deterministic_limit(
    rows: list[dict[str, Any]], count: int | None
) -> list[dict[str, Any]]:
    if count is None:
        return rows
    if count <= 0:
        raise ValueError("split limit must be positive")
    return sorted(
        rows,
        key=lambda row: hashlib.sha256(str(row["episode_id"]).encode("utf-8")).digest(),
    )[:count]


def standardize_taco_row(row: dict[str, Any], *, public_fraction: float = 0.2) -> dict[str, Any] | None:
    difficulty = str(row.get("difficulty", "")).upper()
    question = str(row.get("question", ""))
    if difficulty not in {"EASY", "MEDIUM"} or not question:
        return None
    if str(row.get("picture_num", "0")) not in {"", "0", "None", "none"}:
        return None
    if UNSUPPORTED_PATTERNS.search(question):
        return None
    try:
        tests = parse_taco_tests(row["input_output"])
    except (KeyError, json.JSONDecodeError, TypeError, ValueError):
        return None
    if len(tests) < 2:
        return None
    identity = str(row.get("id", problem_fingerprint(row)))
    public, hidden = split_tests(tests, identity=identity, public_fraction=public_fraction)
    example = CodeExample(
        episode_id=identity,
        question=question,
        difficulty=difficulty,
        public_tests=public,
        hidden_tests=hidden,
        starter_code=str(row.get("starter_code", "")),
        source=str(row.get("source", "taco-verified")),
        metadata={
            "fingerprint": problem_fingerprint(row),
            "question_fingerprint": question_fingerprint(row),
            "url": str(row.get("url", "")),
        },
    )
    return example.model_dump(mode="json")


__all__ = [
    "parse_taco_tests",
    "deterministic_limit",
    "problem_fingerprint",
    "question_fingerprint",
    "split_verified_rows",
    "standardize_taco_row",
]
