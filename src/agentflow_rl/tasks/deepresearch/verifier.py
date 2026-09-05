from __future__ import annotations

import re
import string
from collections import Counter
from typing import Iterable

from agentflow_rl.tasks.base import VerificationResult

from .schemas import Citation, DeepResearchExample, ResearchFinalAnswer


def normalize_answer(text: str) -> str:
    lowered = text.lower()
    lowered = "".join(character for character in lowered if character not in string.punctuation)
    lowered = re.sub(r"\b(a|an|the)\b", " ", lowered)
    return " ".join(lowered.split())


def answer_scores(prediction: str, gold: str) -> tuple[float, float, float, float]:
    predicted = normalize_answer(prediction).split()
    expected = normalize_answer(gold).split()
    exact = float(predicted == expected)
    if not predicted or not expected:
        same_empty = float(predicted == expected)
        return exact, same_empty, same_empty, same_empty
    overlap = sum((Counter(predicted) & Counter(expected)).values())
    precision = overlap / len(predicted)
    recall = overlap / len(expected)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return exact, precision, recall, f1


def supporting_fact_scores(
    prediction: Iterable[Citation], gold: Iterable[Citation]
) -> tuple[float, float, float, float]:
    predicted = {(item.title, item.sentence_id) for item in prediction}
    expected = {(item.title, item.sentence_id) for item in gold}
    exact = float(predicted == expected)
    if not predicted or not expected:
        same_empty = float(predicted == expected)
        return exact, same_empty, same_empty, same_empty
    overlap = len(predicted & expected)
    precision = overlap / len(predicted)
    recall = overlap / len(expected)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return exact, precision, recall, f1


def evaluate_research_answer(
    prediction: ResearchFinalAnswer,
    example: DeepResearchExample,
    *,
    observed_citations: Iterable[Citation] | None = None,
) -> VerificationResult:
    answer_em, answer_precision, answer_recall, answer_f1 = answer_scores(
        prediction.answer, example.answer
    )
    sp_em, sp_precision, sp_recall, sp_f1 = supporting_fact_scores(
        prediction.citations, example.supporting_facts
    )
    joint_precision = answer_precision * sp_precision
    joint_recall = answer_recall * sp_recall
    joint_f1 = (
        2 * joint_precision * joint_recall / (joint_precision + joint_recall)
        if joint_precision + joint_recall
        else 0.0
    )
    joint_em = answer_em * sp_em
    failures = []
    if answer_em == 0.0:
        failures.append("ANSWER_MISMATCH")
    if sp_em == 0.0:
        failures.append("SUPPORTING_FACT_MISMATCH")
    metrics = {
        "answer_em": answer_em,
        "answer_f1": answer_f1,
        "supporting_fact_em": sp_em,
        "supporting_fact_f1": sp_f1,
        "joint_em": joint_em,
        "joint_f1": joint_f1,
    }
    if observed_citations is not None:
        predicted_facts = {(item.title, item.sentence_id) for item in prediction.citations}
        observed_facts = {(item.title, item.sentence_id) for item in observed_citations}
        grounded = len(predicted_facts & observed_facts)
        metrics.update({
            "observed_citation_count": float(len(observed_facts)),
            "citation_grounded_fraction": (
                grounded / len(predicted_facts) if predicted_facts else 0.0
            ),
            "citation_grounded_exact": float(
                bool(predicted_facts) and predicted_facts <= observed_facts
            ),
        })
    return VerificationResult(
        success=joint_em == 1.0,
        reward=joint_f1,
        failure_codes=tuple(failures),
        metrics=metrics,
    )


__all__ = ["answer_scores", "evaluate_research_answer", "normalize_answer", "supporting_fact_scores"]
