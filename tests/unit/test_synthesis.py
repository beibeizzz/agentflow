from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


def test_blueprints_are_balanced_unique_and_materialize_valid_episodes() -> None:
    from agentflow_rl.synthesis.pipeline import generate_blueprints, materialize_blueprint

    blueprints = generate_blueprints(split="smoke", count=8, seed=42)
    episodes = [materialize_blueprint(item) for item in blueprints]
    assert [item.lookup_mode for item in blueprints].count("ticket_id") == 4
    assert sum(item.lookup_mode in {"customer_id", "order_id"} for item in blueprints) == 4
    assert len({item.episode_id for item in blueprints}) == 8
    assert all(episode.goal_spec.target_ticket_id in episode.user_request
               for episode in episodes if episode.curriculum_mode == "direct")
    for episode in episodes:
        target = next(
            ticket for ticket in episode.initial_state.tickets
            if ticket.ticket_id == episode.goal_spec.target_ticket_id
        )
        assert getattr(target, episode.goal_spec.field) != episode.goal_spec.value


def test_resume_reuses_identical_fingerprint_and_rejects_changed_blueprint(tmp_path) -> None:
    from agentflow_rl.synthesis.pipeline import TicketSynthesisPipeline, generate_blueprints

    calls = []

    def rewrite(blueprint, request):
        calls.append(blueprint.episode_id)
        return request

    output = tmp_path / "rows.jsonl"
    progress = tmp_path / "progress.jsonl"
    blueprints = generate_blueprints(split="smoke", count=2, seed=7)
    pipeline = TicketSynthesisPipeline(rewriter=rewrite)
    pipeline.run(blueprints, output_path=output, progress_path=progress)
    assert len(calls) == 2
    pipeline.run(blueprints, output_path=output, progress_path=progress)
    assert len(calls) == 2

    changed = [blueprints[0].model_copy(update={"target_index": 1}), blueprints[1]]
    with pytest.raises(ValueError, match="fingerprint"):
        pipeline.run(changed, output_path=output, progress_path=progress)
    assert len(calls) == 2
    assert len([json.loads(line) for line in output.read_text().splitlines()]) == 2


def test_synthesis_cli_generates_without_business_or_api_dependency(tmp_path) -> None:
    root = Path(__file__).parents[2]
    output = tmp_path / "smoke.jsonl"
    progress = tmp_path / "smoke.progress.jsonl"
    result = subprocess.run(
        [
            sys.executable, str(root / "scripts" / "synthesize_ticket.py"),
            "--split", "smoke", "--count", "4", "--seed", "42",
            "--output", str(output), "--progress", str(progress),
        ],
        cwd=root, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert len(output.read_text(encoding="utf-8").splitlines()) == 4


def test_synthesis_cli_exposes_optional_offline_api_rewriter() -> None:
    root = Path(__file__).parents[2]
    result = subprocess.run(
        [sys.executable, str(root / "scripts" / "synthesize_ticket.py"), "--help"],
        cwd=root, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0
    assert "--rewrite-model" in result.stdout


def test_rewrite_must_preserve_completion_instruction() -> None:
    from agentflow_rl.synthesis.pipeline import generate_blueprints, materialize_blueprint

    blueprint = generate_blueprints(split="smoke", count=2, seed=7)[0]
    original = materialize_blueprint(blueprint)
    rewritten = original.user_request.replace("then complete the request", "and stop")

    with pytest.raises(ValueError, match="completion instruction"):
        materialize_blueprint(blueprint, user_request=rewritten)


def test_indirect_rewrite_must_not_leak_hidden_ticket_id() -> None:
    from agentflow_rl.synthesis.pipeline import generate_blueprints, materialize_blueprint

    blueprint = next(
        item for item in generate_blueprints(split="smoke", count=4, seed=7)
        if item.lookup_mode != "ticket_id"
    )
    original = materialize_blueprint(blueprint)
    leaked = f"{original.user_request} Hidden ticket: {original.goal_spec.target_ticket_id}."

    with pytest.raises(ValueError, match="hidden ticket"):
        materialize_blueprint(blueprint, user_request=leaked)
