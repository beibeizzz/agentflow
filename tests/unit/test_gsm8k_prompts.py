from __future__ import annotations

from textwrap import dedent


def clean(value: str) -> str:
    return "\n".join(line.rstrip() for line in dedent(value).strip().splitlines())


def test_gsm8k_uses_the_original_shared_system_prompt() -> None:
    from agentflow_rl.tasks.gsm8k.prompts import GSM8K_SYSTEM_PROMPT

    assert GSM8K_SYSTEM_PROMPT == (
        "You are good at math problems. Use only the information in the problem. "
        "Keep the reasoning concise and arithmetic-focused."
    )


def test_query_analyzer_prompt_matches_the_original_calculator_prompt() -> None:
    from agentflow_rl.tasks.gsm8k.prompts import query_prompt

    assert clean(query_prompt("Alice has 2 apples.")) == clean("""
        Explain the general solution approach step by step and the final goal to the problem in a concise manner, without delving into specific calculations.

        Inputs:
        - Problem: Alice has 2 apples.

        Rules:
        - Do not calculate numerically.
        - Do not give the final answer or generator number.
        - Do not determine the specific values of the intermediate variables or the final target.
    """)


def test_planner_prompt_restores_original_rules_and_serializes_judge_memory() -> None:
    from agentflow_rl.tasks.gsm8k.prompts import planner_prompt

    memory = {
        "Action Step 1": {
            "tool_name": "Calculator_Tool",
            "sub_goal": "subtotal",
            "command": 'execution = tool.execute(expression="2 + 3")',
            "result": "5",
            "judge": "Conclusion: CONTINUE\nNext action: multiply by 4",
        }
    }
    prompt = planner_prompt("Find the total.", "First add, then multiply.", memory)

    assert clean(prompt) == clean(f"""
        /no_think
        You should choose the next calculator step and provide the arithmetic expression.
        You are strict to output a JSON.
        Use the Judge feedback first.
        Do not repeat any previous Calculation or Sub_goal in Memory.
        Problem: Find the total.
        Query Analysis: First add, then multiply.
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
    """)


def test_verifier_and_generator_prompts_match_original_direct_output_path() -> None:
    from agentflow_rl.tasks.gsm8k.prompts import final_prompt, verifier_prompt

    memory = {"Action Step 1": {"command": "2 + 3", "result": "5"}}
    verifier = verifier_prompt("Find the total.", "Add the values.", memory)
    final = final_prompt("Find the total.", "Add the values.", memory)

    assert "Decide whether memory has enough proof to solve the entire problem." in verifier
    assert "Initial Analysis and Memory's action_predictor_response is only a hint, not proof." in verifier
    assert "- Initial Analysis: Add the values." in verifier
    assert f"- Memory: {memory}" in verifier
    assert "Conclusion: CONTINUE" in verifier
    assert "Conclusion: STOP" in verifier
    assert clean(final) == clean(f"""
        Return the final numeric answer based on the comprehensive Analysis and Memory.

        Problem:Find the total.
        Analysis: Add the values.
        Memory:{memory}

        Rules:
        - Memory contains the previous sub-goals and command actions.
        - Memory may be unreliable because commands can be incomplete, repeated, or based on a wrong expression.
        - Check whether the commands cover every required quantity to solve the problem.
        - If Memory are complete and consistent, refer to the final relevant calculator result.
        - If Memory are incomplete or inconsistent, refer to the the problem and Analysis.
        - Do not explain.
        - Output one number only.
    """)


def test_legacy_executor_prompt_restores_original_command_contract() -> None:
    from agentflow_rl.tasks.gsm8k.prompts import executor_prompt

    prompt = executor_prompt("Find the total.", "2 + 3")
    assert prompt.startswith("\n/no_think\n")
    assert "Context: 2 + 3" in prompt
    assert "Tool: Calculator_Tool" in prompt
    assert 'execution = tool.execute(expression="<raw arithmetic expression>")' in prompt
    assert "Do not change the operation order." in prompt


def test_legacy_executor_accepts_the_original_structured_command_response() -> None:
    from agentflow_rl.verl.agent_loops.gsm8k import extract_legacy_expression

    response = (
        '{"analysis":"copy expression","explanation":"dispatch",'
        '"command":"execution = tool.execute(expression=\\"2 + 3\\")"}'
    )
    assert extract_legacy_expression(response) == "2 + 3"
