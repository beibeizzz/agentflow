from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from typing import Optional


NUMBER_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?(?:/\d[\d,]*)?")


def clean_numeric_text(value: str) -> str:
    return value.strip().replace(",", "").rstrip(".")


def extract_gold_answer(answer: str) -> str:
    marker = "####"
    if marker not in answer:
        raise ValueError("GSM8K answer does not contain the final answer marker '####'.")
    final_part = answer.rsplit(marker, 1)[1]
    match = NUMBER_RE.search(final_part)
    if not match:
        raise ValueError(f"Could not extract numeric gold answer from: {final_part!r}")
    return clean_numeric_text(match.group(0))


def extract_predicted_answer(response: object) -> Optional[str]:
    text = str(response)

    answer_tag_matches = re.findall(r"<answer>\s*([^<>]+?)\s*</answer>", text, re.IGNORECASE | re.DOTALL)
    if answer_tag_matches:
        tagged_number = NUMBER_RE.search(answer_tag_matches[-1])
        if tagged_number:
            return clean_numeric_text(tagged_number.group(0))

    generator_match = re.search(r"Generator:\s*(" + NUMBER_RE.pattern + r")\s*$", text, re.IGNORECASE)
    if generator_match:
        return clean_numeric_text(generator_match.group(1))

    hash_matches = re.findall(r"####\s*(" + NUMBER_RE.pattern + r")", text)
    if hash_matches:
        return clean_numeric_text(hash_matches[-1])

    boxed_matches = re.findall(r"\\boxed\{([^{}]+)\}", text)
    if boxed_matches:
        boxed_number = NUMBER_RE.search(boxed_matches[-1])
        if boxed_number:
            return clean_numeric_text(boxed_number.group(0))

    matches = NUMBER_RE.findall(text)
    if not matches:
        return None
    return clean_numeric_text(matches[-1])


def normalize_numeric_answer(value: str) -> Fraction:
    cleaned = clean_numeric_text(value)
    if "/" in cleaned:
        numerator, denominator = cleaned.split("/", 1)
        return Fraction(int(numerator), int(denominator))
    try:
        return Fraction(Decimal(cleaned))
    except InvalidOperation as exc:
        raise ValueError(f"Could not normalize numeric answer: {value!r}") from exc


def answers_match(predicted: Optional[str], gold: str) -> bool:
    if predicted is None:
        return False
    try:
        return normalize_numeric_answer(predicted) == normalize_numeric_answer(gold)
    except (ValueError, ZeroDivisionError, InvalidOperation):
        return clean_numeric_text(str(predicted)) == clean_numeric_text(str(gold))
