from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from typing import Any

from .schemas import ValidationIssue, ValidationResult


_FORMAT_PATTERN = re.compile(
    r"\AKnown facts:\n(?P<facts>(?:- [^\n]+\n)+)\nQuestion:\n- (?P<question>[^\n]+)\Z"
)
_NUMBER_PATTERN = re.compile(r"(?<![\w.])(?:\d+\s*/\s*\d+|\d+(?:,\d{3})*(?:\.\d+)?%?)(?![\w.])")
_EXPLICIT_ARITHMETIC_PATTERN = re.compile(
    r"(?<!\w)\d+(?:\.\d+)?\s*(?:[+*/]|\s-\s)\s*\d+(?:\.\d+)?"
)
_SOLUTION_RESULT_PATTERN = re.compile(r"<<[^<>]*=([^<>]+)>>")
_FORBIDDEN_MARKERS = ("<<", ">>", "####", "```")
_FORBIDDEN_INSTRUCTIONS = re.compile(
    r"\b(?:solve|calculate|compute|use (?:a |the )?calculator|step[- ]by[- ]step|"
    r"first calculate|then calculate|final answer)\b",
    re.IGNORECASE,
)


def _normalize_number(token: str) -> str:
    compact = token.replace(",", "").replace(" ", "")
    if compact.endswith("%"):
        compact = compact[:-1]
    if "/" in compact:
        numerator, denominator = compact.split("/", 1)
        try:
            return f"fraction:{Decimal(numerator).normalize()}/{Decimal(denominator).normalize()}"
        except InvalidOperation:
            return compact
    try:
        return str(Decimal(compact).normalize())
    except InvalidOperation:
        return compact


def _numbers(text: str) -> set[str]:
    return {_normalize_number(match.group(0)) for match in _NUMBER_PATTERN.finditer(text)}


def _solution_values(answer: str) -> set[str]:
    values: set[str] = set()
    for raw in _SOLUTION_RESULT_PATTERN.findall(answer):
        match = _NUMBER_PATTERN.search(raw.strip())
        if match:
            values.add(_normalize_number(match.group(0)))
    return values


def validate_rewrite(source: dict[str, Any], rewritten_question: str) -> ValidationResult:
    issues: list[ValidationIssue] = []
    text = rewritten_question.strip().replace("\r\n", "\n")
    match = _FORMAT_PATTERN.fullmatch(text)
    facts: tuple[str, ...] = ()
    question = ""

    if match is None:
        issues.append(
            ValidationIssue(
                "invalid_format",
                "Output must contain only Known facts bullets, one blank line, and one Question bullet.",
            )
        )
    else:
        facts = tuple(
            line[2:].strip()
            for line in match.group("facts").rstrip("\n").splitlines()
            if line.startswith("- ")
        )
        question = match.group("question").strip()
        if not facts or any(not fact for fact in facts) or not question:
            issues.append(ValidationIssue("empty_content", "Facts and question must be non-empty."))

    lowered = text.lower()
    if any(marker in text for marker in _FORBIDDEN_MARKERS) or "final answer" in lowered:
        issues.append(ValidationIssue("answer_leak", "Rewrite contains an answer or solution marker."))

    if "=" in text or _EXPLICIT_ARITHMETIC_PATTERN.search(text):
        issues.append(
            ValidationIssue(
                "arithmetic_leak",
                "Rewrite contains an explicit equation or arithmetic expression.",
            )
        )

    if _FORBIDDEN_INSTRUCTIONS.search(text):
        issues.append(
            ValidationIssue(
                "instruction_leak",
                "Rewrite contains solving instructions instead of only facts and a question.",
            )
        )

    source_question = str(source.get("question") or "")
    source_numbers = _numbers(source_question)
    rewritten_numbers = _numbers(text)
    new_numbers = rewritten_numbers - source_numbers
    missing_numbers = source_numbers - rewritten_numbers
    if new_numbers:
        issues.append(
            ValidationIssue(
                "new_number",
                f"Rewrite introduced numeric values absent from the source question: {sorted(new_numbers)}",
            )
        )
    if missing_numbers:
        issues.append(
            ValidationIssue(
                "missing_number",
                f"Rewrite omitted numeric values present in the source question: {sorted(missing_numbers)}",
            )
        )

    leaked_solution_values = new_numbers & _solution_values(str(source.get("answer") or ""))
    gold_values = _numbers(str(source.get("gold_answer") or ""))
    leaked_solution_values |= new_numbers & gold_values
    if leaked_solution_values:
        issues.append(
            ValidationIssue(
                "solution_value_leak",
                f"Rewrite exposed solution-only values: {sorted(leaked_solution_values)}",
            )
        )

    return ValidationResult(issues=tuple(issues), facts=facts, question=question)
