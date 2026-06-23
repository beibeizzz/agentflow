from __future__ import annotations

from collections import Counter
import hashlib
import json
import random
import re
from typing import Any, Iterable

from agentflow.models.executor import Executor
from agentflow.models.memory import Memory
from agentflow.tools.ticket_common.backend import LEGAL_STATUS_TRANSITIONS, TicketBackend
from agentflow.tools.ticket_finish.tool import Ticket_Finish_Tool
from agentflow.tools.ticket_query.tool import Ticket_Query_Tool
from agentflow.tools.ticket_update.tool import Ticket_Update_Tool

from try_ticket_agent.ticket_env.verifier import TicketVerifier, VerificationResult

from .schemas import EpisodeBlueprint


GENERATOR_VERSION = "1.0.0"
SPLIT_CODES = {"train": "tr", "validation": "va", "test": "te", "smoke": "sm"}
SUBJECTS = (
    "Payment review", "Delivery review", "Account review", "Billing review",
    "Refund review", "Order review", "Identity review", "Service review",
)
TEAMS = ("support", "billing", "finance", "logistics", "fraud")
PRIORITIES = ("low", "normal", "high", "urgent")


def stable_integer_hash(*parts: object) -> int:
    text = "\x1f".join(str(part) for part in parts)
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")


def generate_blueprint(*, seed: int, split: str, index: int) -> EpisodeBlueprint:
    if split not in SPLIT_CODES:
        raise ValueError(f"Unknown split: {split}")
    if index < 0:
        raise ValueError("index must be non-negative")
    rng = random.Random(stable_integer_hash(GENERATOR_VERSION, seed, split, index))
    code = SPLIT_CODES[split]
    episode_id = f"ticket-{code}-{index:06d}"
    ticket_count = rng.randint(6, 10)
    target_position = rng.randrange(ticket_count)
    field = ("priority", "assigned_team", "status")[index % 3]
    tickets: list[dict[str, str]] = []
    for position in range(ticket_count):
        tickets.append(
            {
                "ticket_id": f"T-{code.upper()}-{index:06d}-{position:02d}",
                "customer_id": f"C-{code.upper()}-{index:06d}-{position:02d}",
                "order_id": f"O-{code.upper()}-{index:06d}-{position:02d}",
                "subject": SUBJECTS[(index + position) % len(SUBJECTS)],
                "status": "open",
                "assigned_team": TEAMS[(index + position) % len(TEAMS)],
                "priority": PRIORITIES[(index + position) % len(PRIORITIES)],
            }
        )
    target = tickets[target_position]
    if field == "priority":
        choices = [value for value in PRIORITIES if value != target[field]]
    elif field == "assigned_team":
        choices = [value for value in TEAMS if value != target[field]]
    else:
        choices = sorted(LEGAL_STATUS_TRANSITIONS[target["status"]])
    value = rng.choice(choices)

    indirect = index % 5 == 4
    if indirect:
        lookup_mode = "customer_id" if (index // 5) % 2 == 0 else "order_id"
        lookup_value = target[lookup_mode]
        canonical = (
            f"Locate the unique ticket with {lookup_mode} {lookup_value}. "
            f"Set {field} to {value}, then complete the request."
        )
        curriculum_mode = "indirect"
        max_steps = 3
    else:
        lookup_mode = "ticket_id"
        canonical = (
            f"For ticket {target['ticket_id']}, set {field} to {value}, "
            "then complete the request."
        )
        curriculum_mode = "direct"
        max_steps = 2
    goal = {
        "target_ticket_id": target["ticket_id"],
        "field": field,
        "value": value,
        "finish_outcome": "completed",
    }
    return EpisodeBlueprint(
        generator_version=GENERATOR_VERSION,
        seed=int(seed),
        split=split,
        index=index,
        episode_id=episode_id,
        curriculum_mode=curriculum_mode,
        lookup_mode=lookup_mode,
        user_request=canonical,
        canonical_request=canonical,
        initial_state={"tickets": tickets},
        goal_spec=goal,
        max_steps=max_steps,
    )


def execute_reference_actions(blueprint: EpisodeBlueprint) -> VerificationResult:
    backend = TicketBackend()
    backend.reset(blueprint.initial_state, blueprint.goal_spec)
    tools = {
        "Ticket_Query_Tool": Ticket_Query_Tool(),
        "Ticket_Update_Tool": Ticket_Update_Tool(),
        "Ticket_Finish_Tool": Ticket_Finish_Tool(),
    }
    for tool in tools.values():
        tool.bind_backend(backend)
    executor = Executor.__new__(Executor)
    executor.execution_mode = "structured"
    executor.tool_instances_cache = tools
    memory = Memory()
    step = 0

    def call(tool_name: str, arguments: dict[str, str]) -> None:
        nonlocal step
        step += 1
        command = json.dumps(arguments, sort_keys=True, separators=(",", ":"))
        output = executor.execute_tool_command(tool_name, command)
        memory.add_action(step, tool_name, f"Reference {tool_name}", command, output)

    target_id = blueprint.goal_spec["target_ticket_id"]
    target = next(row for row in blueprint.initial_state["tickets"] if row["ticket_id"] == target_id)
    if blueprint.curriculum_mode == "indirect":
        call(
            "Ticket_Query_Tool",
            {"lookup_by": blueprint.lookup_mode, "value": target[blueprint.lookup_mode]},
        )
    call(
        "Ticket_Update_Tool",
        {
            "ticket_id": target_id,
            "field": blueprint.goal_spec["field"],
            "value": blueprint.goal_spec["value"],
        },
    )
    call(
        "Ticket_Finish_Tool",
        {"ticket_id": target_id, "outcome": blueprint.goal_spec["finish_outcome"]},
    )
    return TicketVerifier(backend, max_steps=blueprint.max_steps).verify_final(memory, step)


def _digest(payload: Any) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def blueprint_fingerprint(blueprint: EpisodeBlueprint) -> str:
    return _digest(blueprint.to_dict())


def validate_blueprint_collection(items: Iterable[EpisodeBlueprint]) -> dict[str, Any]:
    rows = list(items)
    episode_ids = [item.episode_id for item in rows]
    state_hashes = [_digest(item.initial_state) for item in rows]
    signatures = [
        _digest(
            {
                "request": re.sub(r"\s+", " ", item.canonical_request).strip().lower(),
                "goal": item.goal_spec,
            }
        )
        for item in rows
    ]

    def duplicates(values: list[str]) -> list[str]:
        counts = Counter(values)
        return sorted(value for value, count in counts.items() if count > 1)

    report = {
        "duplicate_episode_ids": duplicates(episode_ids),
        "duplicate_state_hashes": duplicates(state_hashes),
        "duplicate_request_goal_signatures": duplicates(signatures),
    }
    report["ok"] = not any(report.values())
    return report
