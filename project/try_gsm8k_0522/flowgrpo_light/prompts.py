from __future__ import annotations

from typing import Any


QUERY_ANALYSIS_SYSTEM_PROMPT = (
    "You are a careful grade-school math problem solver. Use only the information in the problem. "
    "Keep the reasoning concise and arithmetic-focused."
)


def build_query_analysis_prompt(question: str) -> str:
    return f"""
Task:
Solve the following GSM8K math word problem. Explain the general solution approach step by step and the final goal to this problem in a concise manner, without delving into specific calculations.

Inputs:
- Problem: {question}

Rules:
- Do not calculate numerically.
- Do not give the final answer or generator number.
- Do not determine the specific values of the intermediate variables or the final target.
"""


def build_planner_prompt(
    *,
    question: str,
    query_analysis: str,
    memory: dict[str, Any],
    step_count: int,
    max_steps: int,
) -> str:
    return f"""
You are Planner and you should plan the next calculator step and provide the arithmetic expression.

Problem: {question}
Query Analysis: {query_analysis}
Memory: {memory}
Step: {step_count} of {max_steps}

Memory Notes:
- If Memory is not empty, inspect previous calculations and results.
- Do not repeat any previous Calculation or Sub_goal.
- Prefer the next missing arithmetic step required to solve the problem.

Rules:
- Return only one JSON object.
- "Sub_goal": briefly say what this calculation computes.
- "Calculation": write only the arithmetic expression and must match this regex: ^[0-9+\\-*/().% ]+$
- Use only numbers from the problem or previous results.
- In Calculation, use only digits, +, -, *, /, %, parentheses, and decimals.
- Do not include variables, words, units, "=", currency symbols, commas, explanatory text, or the result in Calculation.

JSON example:
{{
  "Sub_goal": "Calculate reading time per night",
  "Calculation": "2 / 2"
}}
"""


def build_final_prompt(question: str, query_analysis: str, memory: dict[str, Any]) -> str:
    return f"""
Task:
Return the final numeric answer based on the Analysis and Memory.

Problem:
{question}

Analysis:
{query_analysis}

Memory:
{memory}

Rules:
- If Memory contains complete and consistent calculator results, use the final relevant result.
- If Memory is incomplete or inconsistent, solve from the problem directly.
- Do not explain.
- Output one number only.
"""
