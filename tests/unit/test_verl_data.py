from __future__ import annotations


def test_ticket_row_conversion_keeps_full_isolated_episode() -> None:
    from agentflow_rl.verl.data import ticket_to_verl_row

    source = {
        "episode_id": "ticket-1",
        "user_request": "Set T-1 priority high.",
        "lookup_mode": "ticket_id",
        "max_steps": 2,
        "curriculum_mode": "direct",
        "initial_state": {"tickets": [{"ticket_id": "T-1"}]},
        "goal_spec": {"target_ticket_id": "T-1", "field": "priority", "value": "high"},
    }
    row = ticket_to_verl_row(source, index=3)

    assert row["data_source"] == "ticket"
    assert row["agent_name"] == "agentflow_ticket"
    assert row["prompt"] == [{"role": "user", "content": source["user_request"]}]
    assert row["extra_info"]["episode_id"] == "ticket-1"
    assert row["extra_info"]["initial_state"] == source["initial_state"]
    assert row["extra_info"]["index"] == 3
    assert row["reward_model"]["ground_truth"] == "1"


def test_gsm8k_row_conversion_keeps_gold_answer() -> None:
    from agentflow_rl.verl.data import gsm8k_to_verl_row

    source = {"pid": 7, "question": "2+3?", "gold_answer": "5", "answer": "#### 5"}
    row = gsm8k_to_verl_row(source, index=4)

    assert row["data_source"] == "gsm8k"
    assert row["agent_name"] == "agentflow_gsm8k"
    assert row["extra_info"]["episode_id"] == "7"
    assert row["extra_info"]["gold_answer"] == "5"
    assert row["reward_model"]["ground_truth"] == "5"


def test_conversion_rejects_missing_task_identity() -> None:
    import pytest

    from agentflow_rl.verl.data import gsm8k_to_verl_row, ticket_to_verl_row

    with pytest.raises(ValueError):
        ticket_to_verl_row({"user_request": "x"}, index=0)
    with pytest.raises(ValueError):
        gsm8k_to_verl_row({"question": "x", "gold_answer": "1"}, index=0)


def test_deepresearch_conversion_keeps_terminal_evidence_labels() -> None:
    from agentflow_rl.verl.data import deepresearch_to_verl_row

    source = {
        "episode_id": "research-1",
        "dataset": "hotpotqa",
        "question": "Where?",
        "answer": "Paris",
        "supporting_facts": [["Paris", 0]],
        "metadata": {"retrieval_documents": [{"doc_id": "p", "title": "Paris", "sentences": ["text"]}]},
    }

    row = deepresearch_to_verl_row(source, index=1)

    assert row["agent_name"] == "agentflow_deepresearch"
    assert row["extra_info"]["supporting_facts"] == [["Paris", 0]]
    assert row["reward_model"]["ground_truth"] == "Paris"


def test_coding_conversion_keeps_public_and_hidden_partitions() -> None:
    from agentflow_rl.verl.data import coding_to_verl_row

    source = {
        "episode_id": "code-1",
        "question": "sum",
        "difficulty": "EASY",
        "public_tests": [{"stdin": "1 2\n", "expected_stdout": "3\n"}],
        "hidden_tests": [{"stdin": "4 5\n", "expected_stdout": "9\n"}],
    }

    row = coding_to_verl_row(source, index=2)

    assert row["agent_name"] == "agentflow_coding"
    assert row["extra_info"]["public_tests"] == source["public_tests"]
    assert row["extra_info"]["hidden_tests"] == source["hidden_tests"]
    assert row["reward_model"]["ground_truth"] == "hidden_tests"
