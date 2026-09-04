from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Callable, Iterable

from agentflow_rl.tasks.ticket.schemas import TicketEpisode

from .schemas import SCHEMA_VERSION, TicketBlueprint, blueprint_fingerprint
from .validators import validate_synthesized_episode


FIELDS = (
    ("priority", ("low", "high", "urgent")),
    ("assigned_team", ("support", "billing", "finance", "logistics")),
    ("status", ("pending_customer", "pending_finance", "resolved")),
)


def generate_blueprints(*, split: str, count: int, seed: int) -> list[TicketBlueprint]:
    if count <= 0 or count % 2:
        raise ValueError("balanced synthesis count must be a positive even number")
    rng = random.Random(seed)
    prefix = {"smoke": "SM", "train": "TR", "validation": "VA", "test": "TE"}.get(
        split, split[:2].upper()
    )
    modes = ["ticket_id"] * (count // 2)
    modes += ["customer_id" if index % 2 == 0 else "order_id" for index in range(count // 2)]
    rng.shuffle(modes)
    result = []
    for index, lookup_mode in enumerate(modes):
        field, values = FIELDS[index % len(FIELDS)]
        target_index = index % 4
        initial = {
            "priority": ("normal", "low", "normal", "high")[target_index],
            "assigned_team": ("support", "billing", "finance", "logistics")[target_index],
            "status": "open",
        }[field]
        changing_values = tuple(value for value in values if value != initial)
        result.append(
            TicketBlueprint(
                episode_id=f"ticket-v2-{split}-{index:06d}",
                split=split,
                seed=seed,
                lookup_mode=lookup_mode,
                target_index=target_index,
                field=field,
                value=changing_values[(index // len(FIELDS)) % len(changing_values)],
            )
        )
    return result


def _ticket(blueprint: TicketBlueprint, index: int) -> dict[str, str]:
    stem = f"{blueprint.split[:2].upper()}-{blueprint.seed:04d}-{blueprint.episode_id[-6:]}"
    return {
        "ticket_id": f"T-{stem}-{index:02d}",
        "customer_id": f"C-{stem}-{index:02d}",
        "order_id": f"O-{stem}-{index:02d}",
        "subject": ("Payment", "Delivery", "Account", "Refund")[index % 4] + " review",
        "status": "open",
        "assigned_team": ("support", "billing", "finance", "logistics")[index % 4],
        "priority": ("normal", "low", "normal", "high")[index % 4],
    }


def _request(blueprint: TicketBlueprint, target: dict[str, str]) -> str:
    identifier = target[blueprint.lookup_mode]
    if blueprint.lookup_mode == "ticket_id":
        return (
            f"For ticket {identifier}, set {blueprint.field} to {blueprint.value}, "
            "then complete the request."
        )
    key = blueprint.lookup_mode
    noun = "customer" if key == "customer_id" else "order"
    return (
        f"Find the ticket for {noun} {identifier}, set {blueprint.field} to "
        f"{blueprint.value}, then complete the request."
    )


def materialize_blueprint(
    blueprint: TicketBlueprint, *, user_request: str | None = None
) -> TicketEpisode:
    tickets = tuple(_ticket(blueprint, index) for index in range(4))
    if not 0 <= blueprint.target_index < len(tickets):
        raise ValueError("target_index is outside the generated environment")
    target = tickets[blueprint.target_index]
    direct = blueprint.lookup_mode == "ticket_id"
    episode = TicketEpisode.from_row(
        {
            "episode_id": blueprint.episode_id,
            "user_request": user_request or _request(blueprint, target),
            "lookup_mode": blueprint.lookup_mode,
            "max_steps": 2 if direct else 3,
            "initial_state": {"tickets": tickets},
            "goal_spec": {
                "target_ticket_id": target["ticket_id"],
                "field": blueprint.field,
                "value": blueprint.value,
                "finish_outcome": "completed",
            },
            "curriculum_mode": "direct" if direct else "indirect",
            "generator_version": SCHEMA_VERSION,
            "blueprint_fingerprint": blueprint_fingerprint(blueprint),
            "split": blueprint.split,
            "seed": blueprint.seed,
        }
    )
    validate_synthesized_episode(episode)
    return episode


class TicketSynthesisPipeline:
    def __init__(
        self,
        *,
        rewriter: Callable[[TicketBlueprint, str], str] | None = None,
    ) -> None:
        self.rewriter = rewriter or (lambda _blueprint, request: request)

    def run(
        self,
        blueprints: Iterable[TicketBlueprint],
        *,
        output_path: str | Path,
        progress_path: str | Path,
    ) -> list[TicketEpisode]:
        items = tuple(blueprints)
        progress_file = Path(progress_path)
        existing: dict[str, dict] = {}
        if progress_file.exists():
            for line in progress_file.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    record = json.loads(line)
                    existing[record["episode_id"]] = record

        for item in items:
            record = existing.get(item.episode_id)
            if record and record["blueprint_fingerprint"] != blueprint_fingerprint(item):
                raise ValueError(f"blueprint fingerprint changed for {item.episode_id}")

        episodes = []
        records = []
        for item in items:
            record = existing.get(item.episode_id)
            if record:
                episode = TicketEpisode.from_row(record["row"])
            else:
                template = materialize_blueprint(item)
                rewritten = self.rewriter(item, template.user_request)
                episode = materialize_blueprint(item, user_request=rewritten)
                record = {
                    "episode_id": item.episode_id,
                    "blueprint_fingerprint": blueprint_fingerprint(item),
                    "schema_version": SCHEMA_VERSION,
                    "row": episode.model_dump(mode="json", exclude={"metadata"})
                    | episode.metadata,
                }
            episodes.append(episode)
            records.append(record)

        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        progress_file.parent.mkdir(parents=True, exist_ok=True)
        progress_file.write_text(
            "".join(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in records),
            encoding="utf-8",
        )
        output.write_text(
            "".join(
                json.dumps(
                    episode.model_dump(mode="json", exclude={"metadata"}) | episode.metadata,
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
                for episode in episodes
            ),
            encoding="utf-8",
        )
        return episodes
