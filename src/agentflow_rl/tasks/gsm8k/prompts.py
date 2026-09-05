from __future__ import annotations

from typing import Any


GSM8K_SYSTEM_PROMPT = (
    "You are good at math problems. Use only the information in the problem. "
    "Keep the reasoning concise and arithmetic-focused."
)

QUERY_SYSTEM = GSM8K_SYSTEM_PROMPT
PLANNER_SYSTEM = GSM8K_SYSTEM_PROMPT
EXECUTOR_SYSTEM = GSM8K_SYSTEM_PROMPT
VERIFIER_SYSTEM = GSM8K_SYSTEM_PROMPT
FINAL_SYSTEM = GSM8K_SYSTEM_PROMPT
BASE_GENERATOR_SYSTEM = GSM8K_SYSTEM_PROMPT

CALCULATOR_TOOL_METADATA: dict[str, Any] = {
    "tool_name": "Calculator_Tool",
    "tool_description": "Evaluate an elementary arithmetic expression deterministically.",
    "input_types": {"expression": "str"},
    "output_type": "str",
    "require_llm_engine": False,
}


def query_prompt(question: str) -> str:
    return f"""
Explain the general solution approach step by step and the final goal to the problem in a concise manner, without delving into specific calculations.

Inputs:
- Problem: {question}

Rules:
- Do not calculate numerically.
- Do not give the final answer or generator number.
- Do not determine the specific values of the intermediate variables or the final target.

"""


def planner_prompt(question: str, memory: Any) -> str:
    return f"""
/no_think
Choose one action that advances the arithmetic proof.
Use the latest Verifier judgement first.
Preserve successful results and choose the next unfinished sub-goal.
Problem: {question}
Memory: {memory}

Rules:
- Return only one JSON object.
- Calculator_Tool arguments are {{"expression":"..."}} and the expression uses digits, +, -, *, /, parentheses, and decimals.
- Base_Generator_Tool arguments are {{}} and it supplies concise reasoning for an unresolved sub-goal.
- Output exactly {{"sub_goal":"...","tool_name":"Calculator_Tool|Base_Generator_Tool","arguments":{{...}}}}.

JSON example:
{{
  "sub_goal": "Calculate reading time per night",
  "tool_name": "Calculator_Tool",
  "arguments": {{"expression": "2 / 2"}}
}}

Replace the example values with values specific to the current problem.
"""


def executor_prompt(question: str, proposed_action: str, memory: str) -> str:
    return f"""
/no_think
Validate and concretize the proposed action while preserving tool_name.

Problem: {question}
Proposed action: {proposed_action}
Relevant memory: {memory}

Rules:
- Preserve the proposed tool_name.
- Calculator_Tool receives exactly one arithmetic expression.
- Base_Generator_Tool receives an empty arguments object.
- Return exactly one JSON object with sub_goal, tool_name, and arguments.
"""


def verifier_prompt(question: str, memory: Any) -> str:
    return f"""
Decide whether memory has enough proof to solve the entire problem.
Initial Analysis and generated notes are hints. Executed calculator command/result pairs are proof.

Before STOP, check:
- What exact quantity does the problem ask for?
- What exact quantity did the latest command compute?
- Do the executed results cover every required quantity?

Context:
- Problem: {question}
- Memory: {memory}

Rules:
- First line must contain one Conclusion.
- Return CONTINUE with the missing logic and next action when more evidence is required.
- Return STOP when current memory supports the final answer.

Response formats:
Conclusion: CONTINUE
Current issue: ...
Next action: ...
<end>

Conclusion: STOP
Current memory solves the entire problem.
<end>
"""


def final_prompt(question: str, memory: Any) -> str:
    return f"""
Return the final numeric answer based on the Analysis and Memory.

Problem: {question}
Memory: {memory}

Rules:
- Use executed calculator results as arithmetic evidence.
- Complete any remaining reasoning from the problem and Analysis.
- Output one number only.
"""


__all__ = [
    "BASE_GENERATOR_SYSTEM",
    "CALCULATOR_TOOL_METADATA",
    "EXECUTOR_SYSTEM",
    "FINAL_SYSTEM",
    "GSM8K_SYSTEM_PROMPT",
    "PLANNER_SYSTEM",
    "QUERY_SYSTEM",
    "VERIFIER_SYSTEM",
    "executor_prompt",
    "final_prompt",
    "planner_prompt",
    "query_prompt",
    "verifier_prompt",
]
