from __future__ import annotations

from agentflow_rl.runtime.memory import MemoryStore
from agentflow_rl.verl.agent_loops.base import bounded_memory_text


def test_memory_retains_full_snapshot_and_projects_recent_entries() -> None:
    memory = MemoryStore()
    memory.add(turn_index=-1, role="user", kind="question", content="identity", tags=("identity",))
    memory.add(turn_index=0, role="executor", kind="result", content="old observation")
    memory.add(turn_index=1, role="executor", kind="result", content="recent observation")

    view = memory.project(max_tokens=10, token_counter=lambda text: len(text.split()), required_tags=("identity",))

    assert len(memory.snapshot()) == 3
    assert "identity" in view.text
    assert "recent observation" in view.text
    assert view.omitted_entries >= 1


def test_memory_rejects_non_positive_projection_budget() -> None:
    memory = MemoryStore()
    try:
        memory.project(max_tokens=0)
    except ValueError as exc:
        assert "positive" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_bounded_memory_text_reserves_role_specific_prompt_space() -> None:
    memory = MemoryStore()
    memory.add(turn_index=-1, role="query", kind="analysis", content="a" * 200, tags=("identity",))
    memory.add(turn_index=0, role="executor", kind="tool", content="b" * 200)

    text = bounded_memory_text(
        memory,
        token_counter=len,
        max_prompt_tokens=300,
        max_memory_tokens=250,
        reserve_tokens=50,
        reserved_texts=("q" * 100,),
    )

    assert len(text) <= 150


def test_memory_role_projection_keeps_latest_tagged_state() -> None:
    memory = MemoryStore()
    memory.add(
        turn_index=-1,
        role="query_analyzer",
        kind="analysis",
        content="task analysis",
        tags=("identity",),
    )
    memory.add(
        turn_index=0,
        role="executor",
        kind="tool_event",
        content="old code",
        tags=("current_code",),
    )
    memory.add(
        turn_index=1,
        role="verifier",
        kind="judgement",
        content="old judgement",
    )
    memory.add(
        turn_index=2,
        role="executor",
        kind="tool_event",
        content="new code",
        tags=("current_code",),
    )

    view = memory.project(
        max_tokens=100,
        token_counter=lambda text: len(text.split()),
        required_tags=("identity",),
        required_latest_tags=("current_code",),
        include_roles=("query_analyzer", "executor"),
        max_recent_entries=0,
    )

    assert "task analysis" in view.text
    assert "new code" in view.text
    assert "old code" not in view.text
    assert "old judgement" not in view.text


def test_memory_role_projection_prioritizes_latest_state_under_budget_pressure() -> None:
    memory = MemoryStore()
    memory.add(
        turn_index=-1,
        role="query_analyzer",
        kind="analysis",
        content="long stable identity that fills most of the available memory budget",
        tags=("identity",),
    )
    memory.add(
        turn_index=2,
        role="executor",
        kind="tool_event",
        content="latest tested code passed",
        tags=("latest_test_result",),
    )

    view = memory.project(
        max_tokens=9,
        token_counter=lambda text: len(text.split()),
        required_tags=("identity",),
        required_latest_tags=("latest_test_result",),
        max_recent_entries=0,
    )

    assert "latest tested code passed" in view.text
    assert "long stable identity" not in view.text
