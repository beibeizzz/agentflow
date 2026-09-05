from __future__ import annotations


def test_gsm8k_uses_the_shared_system_prompt_for_every_role() -> None:
    from agentflow_rl.tasks.gsm8k.prompts import (
        BASE_GENERATOR_SYSTEM,
        EXECUTOR_SYSTEM,
        FINAL_SYSTEM,
        GSM8K_SYSTEM_PROMPT,
        PLANNER_SYSTEM,
        QUERY_SYSTEM,
        VERIFIER_SYSTEM,
    )

    assert GSM8K_SYSTEM_PROMPT == (
        "You are good at math problems. Use only the information in the problem. "
        "Keep the reasoning concise and arithmetic-focused."
    )
    assert {
        QUERY_SYSTEM,
        PLANNER_SYSTEM,
        EXECUTOR_SYSTEM,
        VERIFIER_SYSTEM,
        FINAL_SYSTEM,
        BASE_GENERATOR_SYSTEM,
    } == {GSM8K_SYSTEM_PROMPT}


def test_query_analyzer_keeps_specific_arithmetic_out_of_initial_plan() -> None:
    from agentflow_rl.tasks.gsm8k.prompts import query_prompt

    prompt = query_prompt("Alice has 2 apples.")
    assert "Explain the general solution approach" in prompt
    assert "Do not calculate numerically." in prompt
    assert "Alice has 2 apples." in prompt


def test_planner_exposes_calculator_and_base_generator_actions_with_memory() -> None:
    from agentflow_rl.tasks.gsm8k.prompts import planner_prompt

    memory = "Conclusion: CONTINUE\nNext action: multiply by 4"
    prompt = planner_prompt("Find the total.", memory)

    assert "Calculator_Tool|Base_Generator_Tool" in prompt
    assert '"sub_goal":"..."' in prompt
    assert memory in prompt
    assert "latest Verifier judgement" in prompt


def test_verifier_and_generator_separate_evidence_from_final_answer() -> None:
    from agentflow_rl.tasks.gsm8k.prompts import final_prompt, verifier_prompt

    memory = '{"tool_name":"Calculator_Tool","result":{"value":"5"}}'
    verifier = verifier_prompt("Find the total.", memory)
    final = final_prompt("Find the total.", memory)

    assert "generated notes are hints" in verifier
    assert "Executed calculator command/result pairs are proof." in verifier
    assert "Conclusion: CONTINUE" in verifier
    assert "Conclusion: STOP" in verifier
    assert "Return the final numeric answer" in final
    assert "Output one number only." in final
    assert memory in final


def test_executor_preserves_planner_tool_selection() -> None:
    from agentflow_rl.tasks.gsm8k.prompts import executor_prompt

    action = '{"sub_goal":"subtotal","tool_name":"Calculator_Tool","arguments":{"expression":"2 + 3"}}'
    prompt = executor_prompt("Find the total.", action, "prior memory")

    assert prompt.startswith("\n/no_think\n")
    assert action in prompt
    assert "preserving tool_name" in prompt
    assert "Base_Generator_Tool receives an empty arguments object" in prompt


def test_legacy_executor_response_remains_accepted_for_checkpoint_compatibility() -> None:
    from agentflow_rl.verl.agent_loops.gsm8k import extract_legacy_expression

    response = (
        '{"analysis":"copy expression","explanation":"dispatch",'
        '"command":"execution = tool.execute(expression=\\"2 + 3\\")"}'
    )
    assert extract_legacy_expression(response) == "2 + 3"


def test_gsm8k_action_accepts_legacy_calculator_and_shared_base_generator() -> None:
    from agentflow_rl.tasks.gsm8k.schemas import GSM8KAction

    legacy = GSM8KAction.parse('{"Sub_goal":"sum","Calculation":"2 + 3"}')
    generated = GSM8KAction.parse(
        '{"sub_goal":"derive the equation","tool_name":"Base_Generator_Tool","arguments":{}}'
    )

    assert legacy.tool_name == "Calculator_Tool"
    assert legacy.arguments == {"expression": "2 + 3"}
    assert generated.tool_name == "Base_Generator_Tool"
