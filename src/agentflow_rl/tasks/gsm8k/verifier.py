from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from fractions import Fraction


NUMBER_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?(?:/\d[\d,]*)?")


def extract_numeric_answer(value: object) -> str | None:
    matches = NUMBER_RE.findall(str(value))
    return matches[-1].replace(",", "").rstrip(".") if matches else None


def _number(value: str) -> Fraction:
    cleaned = value.replace(",", "").strip().rstrip(".")
    if "/" in cleaned:
        left, right = cleaned.split("/", 1)
        return Fraction(int(left), int(right))
    return Fraction(Decimal(cleaned))


def answers_match(predicted: str | None, gold: str) -> bool:
    if predicted is None:
        return False
    try:
        return abs(_number(predicted) - _number(gold)) <= Fraction(1, 1_000_000)
    except (ValueError, ZeroDivisionError, InvalidOperation):
        return False


def extract_verifier_conclusion(response: str) -> str:
    matches = re.findall(r"Conclusion:\s*(STOP|CONTINUE)\b", response, re.IGNORECASE)
    if not matches:
        raise ValueError("verifier response has no explicit conclusion")
    return matches[-1].upper()
