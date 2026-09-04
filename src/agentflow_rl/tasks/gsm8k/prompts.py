from __future__ import annotations

from typing import Any


GSM8K_SYSTEM_PROMPT = (
    "You are good at math problems. Use only the information in the problem. "
    "Keep the reasoning concise and arithmetic-focused."
)

# The original experiment routes every frozen subagent and the trainable Planner
# through the same GSM8K system prompt.
QUERY_SYSTEM = GSM8K_SYSTEM_PROMPT
PLANNER_SYSTEM = GSM8K_SYSTEM_PROMPT
EXECUTOR_SYSTEM = GSM8K_SYSTEM_PROMPT
VERIFIER_SYSTEM = GSM8K_SYSTEM_PROMPT
FINAL_SYSTEM = GSM8K_SYSTEM_PROMPT

CALCULATOR_TOOL_METADATA: dict[str, Any] = {
    "tool_name": "Calculator_Tool",
    "tool_description": (
        "A deterministic calculator supporting multi-step operations, percentages, "
        "and elementary arithmetic. "
    ),
    "tool_version": None,
    "input_types": {
        "expression": (
            "str - Arithmetic expression using +, -, *, /, parentheses, decimals. "
        )
    },
    "output_type": "str - The numeric result of the evaluated expression.",
    "demo_commands": [],
    "require_llm_engine": False,
    "user_metadata": {
        "limitations": (
            'Only arithmetic expressions are allowed. Variables, functions, text, '
            'units, and "=" signs are not allowed.'
        )
    },
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


def planner_prompt(question: str, analysis: str, memory: dict[str, Any]) -> str:
    return f"""
/no_think
You should choose the next calculator step and provide the arithmetic expression.
You are strict to output a JSON.
Use the Judge feedback first.
Do not repeat any previous Calculation or Sub_goal in Memory.
Problem: {question}
Query Analysis: {analysis}
Memory: {memory}

Rules:
- Return only one JSON object.
- "Sub_goal": briefly say what this calculation computes.
- "Calculation": write only the arithmetic expression and must match this regex: ^[0-9+\\-*/(). ]+$
- In calculation, use only digits, +, -, *, /, parentheses, and decimals.
- In calculation, do not include variables, words, units, "=", , currency symbols, commas, explanatory text, or the result.

JSON example:
{{
  "Sub_goal": "Calculate reading time per night",
  "Calculation": "2 / 2"
}}

Important:
- Replace the placeholder contents with values specific to the current problem.
- Do not copy the example text or expression.

"""


def executor_prompt(
    question: str,
    context: str,
    tool_metadata: dict[str, Any] = CALCULATOR_TOOL_METADATA,
) -> str:
    return f"""
/no_think
Your should extract the expression of Context into one executable Calculator_Tool command.

Problem: {question}
Context: {context}
Tool: Calculator_Tool
Tool description: {tool_metadata}

Rules:
- Extract and copy the original arithmetic expression from Context.
- Do not calculate the result.
- Do not add words, "=" or units inside expression.
- Return only one Python code block and no prose and follow the format below strictly.
- Do not add numbers from the problem.
- Do not combine Context with other calculations.
- Do not change the operation order.

Output Format:

```python
execution = tool.execute(expression="<raw arithmetic expression>")
```

"""


def verifier_prompt(question: str, analysis: str, memory: dict[str, Any]) -> str:
    return f"""
Decide whether memory has enough proof to solve the entire problem.
Initial Analysis and Memory's action_predictor_response is only a hint, not proof.
Command/result pairs from executed tools count as proof.

Before STOP, check:
- What exact quantity does the problem ask for?
- What exact quantity did the latest command compute?
- Are they the same quantity?
If you are not sure about confirming the above questions, output Conclusion: CONTINUE.

Context:
- Problem: {question}
- Initial Analysis: {analysis}
- Memory: {memory}

Rules:
- First line must be Conclusion. Do not write any other Conclusion in the response.
- Do not solve the problem or repeat the raw problem.
- Analyse the missing logic if neccessary.
- Follow the formats above. Response only one of the two formats below.
- Do not write another Conclusion later.


Response Format:
Format1 (When memory not solves the entire problem):
Conclusion: CONTINUE
Current memory can't solve the problem.
Current issue: ...
Next action:...
<end>

Format2 (Only when memory solves the entire problem):
Conclusion: STOP
Current memory solves the entire problem.
<end>





"""


def final_prompt(question: str, analysis: str, memory: dict[str, Any]) -> str:
    return f"""
Return the final numeric answer based on the comprehensive Analysis and Memory.

Problem:{question}
Analysis: {analysis}
Memory:{memory}

Rules:
- Memory contains the previous sub-goals and command actions.
- Memory may be unreliable because commands can be incomplete, repeated, or based on a wrong expression.
- Check whether the commands cover every required quantity to solve the problem.
- If Memory are complete and consistent, refer to the final relevant calculator result.
- If Memory are incomplete or inconsistent, refer to the the problem and Analysis.
- Do not explain.
- Output one number only.
"""
