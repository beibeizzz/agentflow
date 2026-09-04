from __future__ import annotations

import asyncio
from types import SimpleNamespace


class FakeTokenizer:
    def __init__(self, responses):
        self.responses = responses
        self.counter = 0

    def apply_chat_template(self, messages, **kwargs):
        self.counter += 1
        return [100 + self.counter, 200 + self.counter]

    def decode(self, token_ids, skip_special_tokens=True):
        return self.responses[tuple(token_ids)]


class FakeServer:
    def __init__(self, outputs):
        self.outputs = list(outputs)

    async def generate(self, **kwargs):
        token_ids = self.outputs.pop(0)
        return SimpleNamespace(
            token_ids=list(token_ids),
            log_probs=[-0.1] * len(token_ids),
            num_preempted=0,
            extra_fields={},
        )


class FakeFrozen:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = []

    async def generate(self, **kwargs):
        self.calls.append(kwargs)
        output = self.outputs.pop(0)
        if isinstance(output, BaseException):
            raise output
        return output


def config(*, executor_mode="deterministic"):
    return SimpleNamespace(
        agentflow=SimpleNamespace(
            max_time_s=120,
            max_steps=3,
            gsm8k=SimpleNamespace(executor_mode=executor_mode),
        ),
        actor_rollout_ref=SimpleNamespace(
            rollout=SimpleNamespace(prompt_length=2048, response_length=512)
        ),
    )


def ticket_row():
    return {
        "episode_id": "ticket-1",
        "user_request": "Set T-1 priority high and complete.",
        "lookup_mode": "ticket_id",
        "max_steps": 2,
        "curriculum_mode": "direct",
        "initial_state": {"tickets": [{
            "ticket_id": "T-1", "customer_id": "C-1", "order_id": "O-1",
            "subject": "Payment", "status": "open", "assigned_team": "support",
            "priority": "normal",
        }]},
        "goal_spec": {"target_ticket_id": "T-1", "field": "priority", "value": "high"},
    }


def indirect_ticket_row():
    return {
        "episode_id": "ticket-indirect",
        "user_request": "Find order O-2, set its ticket priority high, and complete.",
        "lookup_mode": "order_id",
        "max_steps": 3,
        "curriculum_mode": "indirect",
        "initial_state": {"tickets": [
            {
                "ticket_id": "T-1", "customer_id": "C-1", "order_id": "O-1",
                "subject": "Payment", "status": "open", "assigned_team": "support",
                "priority": "normal",
            },
            {
                "ticket_id": "T-2", "customer_id": "C-2", "order_id": "O-2",
                "subject": "Delivery", "status": "open", "assigned_team": "billing",
                "priority": "normal",
            },
        ]},
        "goal_spec": {"target_ticket_id": "T-2", "field": "priority", "value": "high"},
    }


def test_ticket_agent_loop_returns_one_verl_output_per_real_planner_turn() -> None:
    from agentflow_rl.verl.agent_loops.ticket import TicketAgentLoop

    responses = {
        (11, 2): '{"tool_name":"Ticket_Update_Tool","arguments":{"ticket_id":"T-1","field":"priority","value":"high"}}',
        (12, 2): '{"tool_name":"Ticket_Finish_Tool","arguments":{"ticket_id":"T-1","outcome":"completed"}}',
    }
    loop = TicketAgentLoop(
        trainer_config=config(), server_manager=FakeServer([(11, 2), (12, 2)]),
        tokenizer=FakeTokenizer(responses), processor=None, dataset_cls=object,
        data_config=SimpleNamespace(config={}), frozen_model=FakeFrozen(["plan"]),
    )
    outputs = asyncio.run(loop.run(
        {"temperature": 1.2, "top_p": 1.0, "top_k": -1, "logprobs": True},
        uid="q", session_id=0, extra_info=ticket_row(), raw_prompt=[],
    ))

    assert len(outputs) == 2
    assert outputs[0].reward_score is None
    assert outputs[-1].reward_score == 1.0
    assert outputs[0].response_ids == [11, 2]
    assert outputs[1].extra_fields["terminal_reason"] == "finish_submitted"
    assert outputs[-1].extra_fields["reward_extra_info"]["success"] == 1.0
    assert all(output.extra_fields["valid_for_training"] for output in outputs)


def test_invalid_planner_output_is_not_counted_as_a_tool_call() -> None:
    from agentflow_rl.verl.agent_loops.ticket import TicketAgentLoop

    loop = TicketAgentLoop(
        trainer_config=config(),
        server_manager=FakeServer([(21, 2)]),
        tokenizer=FakeTokenizer({(21, 2): "invalid action"}),
        processor=None,
        dataset_cls=object,
        data_config=SimpleNamespace(config={}),
        frozen_model=FakeFrozen(["plan"]),
    )

    outputs = asyncio.run(loop.run(
        {"temperature": 1.2, "top_p": 1.0, "top_k": -1, "logprobs": True},
        uid="invalid",
        session_id=0,
        extra_info=ticket_row(),
        raw_prompt=[],
    ))

    assert outputs[0].metrics.tool_calls == 0.0
    assert outputs[0].extra_fields["terminal_reason"] == "invalid_action"


def test_gsm8k_agent_loop_restores_judge_memory_and_legacy_executor() -> None:
    from agentflow_rl.verl.agent_loops.gsm8k import GSM8KAgentLoop

    responses = {(31, 2): '{"Sub_goal":"add values","Calculation":"2 + 3"}'}
    frozen = FakeFrozen([
        "Add the values.",
        '```python\nexecution = tool.execute(expression="2 + 3")\n```',
        "Conclusion: STOP\nCurrent memory solves the entire problem.\n<end>",
        "5",
    ])
    loop = GSM8KAgentLoop(
        trainer_config=config(executor_mode="legacy_llm"),
        server_manager=FakeServer([(31, 2)]), tokenizer=FakeTokenizer(responses),
        processor=None, dataset_cls=object, data_config=SimpleNamespace(config={}),
        frozen_model=frozen,
    )
    outputs = asyncio.run(loop.run(
        {"temperature": 1.2, "top_p": 1.0, "top_k": -1, "logprobs": True},
        uid="g", session_id=0,
        extra_info={"episode_id": "g1", "question": "What is 2 plus 3?", "gold_answer": "5"},
        raw_prompt=[],
    ))

    assert len(outputs) == 1
    assert outputs[0].reward_score == 1.0
    memory = outputs[0].extra_fields["memory_actions"]
    assert memory["Action Step 1"]["result"] == ["5"]
    assert memory["Action Step 1"]["judge"].startswith("Conclusion: STOP")
    assert frozen.calls[1]["think_mode"] == "off"


def test_ticket_indirect_agent_loop_preserves_query_update_finish_parity() -> None:
    from agentflow_rl.verl.agent_loops.ticket import TicketAgentLoop

    responses = {
        (41, 2): '{"tool_name":"Ticket_Query_Tool","arguments":{"lookup_by":"order_id","value":"O-2"}}',
        (42, 2): '{"tool_name":"Ticket_Update_Tool","arguments":{"ticket_id":"T-2","field":"priority","value":"high"}}',
        (43, 2): '{"tool_name":"Ticket_Finish_Tool","arguments":{"ticket_id":"T-2","outcome":"completed"}}',
    }
    loop = TicketAgentLoop(
        trainer_config=config(),
        server_manager=FakeServer([(41, 2), (42, 2), (43, 2)]),
        tokenizer=FakeTokenizer(responses),
        processor=None,
        dataset_cls=object,
        data_config=SimpleNamespace(config={}),
        frozen_model=FakeFrozen(["query, update, finish"]),
    )
    outputs = asyncio.run(loop.run(
        {"temperature": 1.2, "top_p": 1.0, "top_k": -1, "logprobs": True},
        uid="indirect",
        session_id=0,
        extra_info=indirect_ticket_row(),
        raw_prompt=[],
    ))

    events = outputs[-1].extra_fields["tool_events"]
    assert [event["tool_name"] for event in events] == [
        "Ticket_Query_Tool", "Ticket_Update_Tool", "Ticket_Finish_Tool"
    ]
    assert events[0]["result"]["data"]["ticket_id"] == "T-2"
    assert outputs[-1].reward_score == 1.0


def test_gsm8k_three_turn_agent_loop_passes_every_judge_to_next_prompt() -> None:
    from agentflow_rl.verl.agent_loops.gsm8k import GSM8KAgentLoop

    responses = {
        (51, 2): '{"Sub_goal":"first","Calculation":"2 + 3"}',
        (52, 2): '{"Sub_goal":"second","Calculation":"5 * 4"}',
        (53, 2): '{"Sub_goal":"confirm","Calculation":"20 + 0"}',
    }
    frozen = FakeFrozen([
        "Solve with arithmetic.",
        "Conclusion: CONTINUE\nNeed the total.\n<end>",
        "Conclusion: CONTINUE\nConfirm it.\n<end>",
        "Conclusion: STOP\nCurrent memory solves the entire problem.\n<end>",
        "20",
    ])
    loop = GSM8KAgentLoop(
        trainer_config=config(executor_mode="deterministic"),
        server_manager=FakeServer([(51, 2), (52, 2), (53, 2)]),
        tokenizer=FakeTokenizer(responses),
        processor=None,
        dataset_cls=object,
        data_config=SimpleNamespace(config={}),
        frozen_model=frozen,
    )
    outputs = asyncio.run(loop.run(
        {"temperature": 1.2, "top_p": 1.0, "top_k": -1, "logprobs": True},
        uid="gsm-three",
        session_id=0,
        extra_info={"episode_id": "g3", "question": "Compute twenty.", "gold_answer": "20"},
        raw_prompt=[],
    ))

    assert len(outputs) == 3
    assert outputs[-1].reward_score == 1.0
    assert "Need the total." in outputs[1].extra_fields["planner_prompt"]
    assert "Confirm it." in outputs[2].extra_fields["planner_prompt"]
    assert outputs[-1].extra_fields["memory_actions"]["Action Step 3"]["judge"].startswith(
        "Conclusion: STOP"
    )


def test_episode_deadline_is_a_valid_zero_reward_sample_not_infrastructure_invalid() -> None:
    from agentflow_rl.verl.agent_loops.gsm8k import GSM8KAgentLoop

    responses = {(61, 2): '{"Sub_goal":"add","Calculation":"2 + 3"}'}
    frozen = FakeFrozen(["analysis", TimeoutError("deadline")])
    loop = GSM8KAgentLoop(
        trainer_config=config(executor_mode="deterministic"),
        server_manager=FakeServer([(61, 2)]),
        tokenizer=FakeTokenizer(responses),
        processor=None,
        dataset_cls=object,
        data_config=SimpleNamespace(config={}),
        frozen_model=frozen,
    )
    outputs = asyncio.run(loop.run(
        {"temperature": 1.2, "top_p": 1.0, "top_k": -1, "logprobs": True},
        uid="timeout",
        session_id=0,
        extra_info={"episode_id": "timeout", "question": "2+3?", "gold_answer": "5"},
        raw_prompt=[],
    ))

    assert outputs[-1].reward_score == 0.0
    assert outputs[-1].extra_fields["terminal_reason"] == "time_limit"
    assert outputs[-1].extra_fields["valid_for_training"] is True


def test_failure_before_first_planner_turn_emits_non_trainable_diagnostic_row() -> None:
    from agentflow_rl.verl.agent_loops.ticket import TicketAgentLoop

    loop = TicketAgentLoop(
        trainer_config=config(),
        server_manager=FakeServer([]),
        tokenizer=FakeTokenizer({}),
        processor=None,
        dataset_cls=object,
        data_config=SimpleNamespace(config={}),
        frozen_model=FakeFrozen([OSError("frozen service unavailable")]),
    )
    outputs = asyncio.run(loop.run(
        {"temperature": 1.2, "top_p": 1.0, "top_k": -1, "logprobs": True},
        uid="infra",
        session_id=0,
        extra_info=ticket_row(),
        raw_prompt=[],
    ))

    assert len(outputs) == 1
    assert outputs[0].response_mask == [0]
    assert outputs[0].reward_score == 0.0
    assert outputs[0].extra_fields["synthetic_diagnostic"] is True
    assert outputs[0].extra_fields["valid_for_training"] is False


def test_deepresearch_agent_loop_returns_joint_terminal_reward() -> None:
    from agentflow_rl.tasks.deepresearch.retrieval import InMemoryBM25Index, ResearchDocument
    from agentflow_rl.verl.agent_loops.deepresearch import DeepResearchAgentLoop

    action = '{"sub_goal":"read evidence","tool_name":"Research_Read_Tool","arguments":{"doc_id":"paris"}}'
    responses = {(71, 2): action}
    frozen = FakeFrozen([
        "Find direct evidence for the capital.",
        action,
        "Evidence is sufficient. Conclusion: STOP",
        '{"answer":"Paris","report":"Paris is the capital.","citations":[{"title":"Paris","sentence_id":0}]}',
    ])
    loop = DeepResearchAgentLoop(
        trainer_config=config(),
        server_manager=FakeServer([(71, 2)]),
        tokenizer=FakeTokenizer(responses),
        processor=None,
        dataset_cls=object,
        data_config=SimpleNamespace(config={}),
        frozen_model=frozen,
        research_index=InMemoryBM25Index([
            ResearchDocument("paris", "Paris", ("Paris is the capital of France.",))
        ]),
    )

    outputs = asyncio.run(loop.run(
        {"temperature": 1.0, "top_p": 1.0, "top_k": -1, "logprobs": True},
        uid="research",
        session_id=0,
        extra_info={
            "episode_id": "research-1",
            "dataset": "hotpotqa",
            "question": "What is the capital of France?",
            "answer": "Paris",
            "supporting_facts": [["Paris", 0]],
        },
        raw_prompt=[],
    ))

    assert len(outputs) == 1
    assert outputs[-1].reward_score == 1.0
    assert outputs[-1].extra_fields["verification"]["metrics"]["joint_f1"] == 1.0
    assert outputs[-1].extra_fields["terminal_reason"] == "verifier_stop"


def test_coding_agent_loop_uses_public_tools_and_hidden_terminal_tests() -> None:
    from agentflow_rl.tasks.coding.sandbox import LocalPythonSandbox
    from agentflow_rl.verl.agent_loops.coding import CodingAgentLoop

    code = "a, b = map(int, input().split())\\nprint(a + b)\\n"
    action = '{"sub_goal":"write solution","tool_name":"Code_Write_Tool","arguments":{"code":"' + code + '"}}'
    responses = {(81, 2): action}
    frozen = FakeFrozen([
        "Parse two integers and print their sum.",
        action,
        "The code is ready. Conclusion: STOP",
        '{"code":"' + code + '"}',
    ])
    loop = CodingAgentLoop(
        trainer_config=config(),
        server_manager=FakeServer([(81, 2)]),
        tokenizer=FakeTokenizer(responses),
        processor=None,
        dataset_cls=object,
        data_config=SimpleNamespace(config={}),
        frozen_model=frozen,
        code_sandbox=LocalPythonSandbox(),
    )

    outputs = asyncio.run(loop.run(
        {"temperature": 1.0, "top_p": 1.0, "top_k": -1, "logprobs": True},
        uid="coding",
        session_id=0,
        extra_info={
            "episode_id": "sum",
            "question": "Read two integers and print their sum.",
            "difficulty": "EASY",
            "public_tests": [{"stdin": "1 2\n", "expected_stdout": "3\n"}],
            "hidden_tests": [{"stdin": "8 9\n", "expected_stdout": "17\n"}],
        },
        raw_prompt=[],
    ))

    assert len(outputs) == 1
    assert outputs[-1].reward_score == 1.0
    assert outputs[-1].extra_fields["verification"]["metrics"]["hidden_pass_rate"] == 1.0
    memory_text = str(outputs[-1].extra_fields["memory"])
    assert "8 9" not in memory_text
