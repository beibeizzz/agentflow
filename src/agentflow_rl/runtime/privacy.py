from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .errors import PrivacyBoundaryError


PRIVATE_FIELD_NAMES = frozenset(
    {
        "answer",
        "answers",
        "answer_aliases",
        "canonical_answer",
        "gold",
        "gold_answer",
        "gold_supporting_facts",
        "hidden_tests",
        "private_payload",
        "reference_answer",
        "reference_solution",
        "supporting_facts",
    }
)


def assert_public_payload(value: Any, *, path: str = "payload") -> None:
    """Reject evaluator-private field names from model-visible structures."""
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = str(key).strip().casefold()
            if normalized in PRIVATE_FIELD_NAMES:
                raise PrivacyBoundaryError(
                    f"evaluator-private field {key!r} found at {path}"
                )
            assert_public_payload(item, path=f"{path}.{key}")
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, item in enumerate(value):
            assert_public_payload(item, path=f"{path}[{index}]")


__all__ = ["PRIVATE_FIELD_NAMES", "assert_public_payload"]
