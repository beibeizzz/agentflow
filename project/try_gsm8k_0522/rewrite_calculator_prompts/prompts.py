from __future__ import annotations

import json
from typing import Any


def build_rewrite_messages(
    source: dict[str, Any],
    prior_failures: list[str] | None = None,
) -> list[dict[str, str]]:
    failures = prior_failures or []
    system = """You rewrite GSM8K prompts for a calculator-only AgentFlow Planner.

Return one JSON object with exactly this schema:
{"rewritten_question": "Known facts:\\n- ...\\n\\nQuestion:\\n- ..."}

The rewritten_question itself must contain only the two displayed sections.

Rules:
- Preserve every fact needed to solve the original problem and preserve the exact target quantity.
- Keep natural-language mathematical relations such as half, twice, more than, fewer than, percentages, rates, and group prices.
- Make pronoun references and relation direction unambiguous.
- Put one principal fact or relation in each bullet.
- Do not provide equations, arithmetic expressions, intermediate results, the final answer, a solution plan, step numbers, or instructions.
- Do not introduce, remove, or alter numeric quantities.
- The problem must remain solvable with at most three meaningful calculator calls, allowing related arithmetic to be combined inside one call.
- Prefer a structure where later calculator calls can use meaningful results stored in Memory.
- Output valid JSON only."""
    user_payload = {
        "original_question": source.get("question"),
        "reference_solution_for_semantics_only": source.get("answer"),
        "gold_answer_for_validation_only": source.get("gold_answer"),
        "previous_validation_failures": failures,
    }
    return [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": "Rewrite the following record. Output JSON only.\n" + json.dumps(user_payload, ensure_ascii=False),
        },
    ]


def build_judge_messages(
    source: dict[str, Any],
    rewritten_question: str,
) -> list[dict[str, str]]:
    system = """You are an independent quality judge for calculator-only AgentFlow training data.

Return one JSON object with exactly this schema:
{"accepted": true, "reasons": []}

Set accepted to false and provide concise reasons if any check fails:
- Every necessary source fact and numeric quantity is preserved.
- No fact, assumption, or numeric quantity is introduced.
- Relation direction and the requested target quantity are unchanged.
- Natural-language relations remain understandable rather than being replaced by equations.
- The rewrite contains no arithmetic expression, intermediate result, final answer, solution plan, or solving instruction.
- The task can be solved using only a Calculator tool.
- A reasonable solution fits in at most three meaningful Calculator calls, with related arithmetic combined where appropriate.
- For multi-step problems, meaningful intermediate results can be reused from Memory.
- The rewrite contains only Known facts and Question sections.

Use the reference solution only to verify semantic equivalence. Do not reproduce its reasoning.
Output valid JSON only."""
    payload = {
        "original_question": source.get("question"),
        "reference_solution": source.get("answer"),
        "gold_answer": source.get("gold_answer"),
        "rewritten_question": rewritten_question,
    }
    return [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": "Judge this candidate. Output JSON only.\n" + json.dumps(payload, ensure_ascii=False),
        },
    ]

