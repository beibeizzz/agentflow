from __future__ import annotations

from dataclasses import dataclass
import re

from .schemas import EpisodeBlueprint


@dataclass(frozen=True)
class CandidateValidation:
    codes: tuple[str, ...]
    messages: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.codes


def validate_candidate(blueprint: EpisodeBlueprint, user_request: str) -> CandidateValidation:
    codes: list[str] = []
    messages: list[str] = []

    def reject(code: str, message: str) -> None:
        if code not in codes:
            codes.append(code)
            messages.append(message)

    if not isinstance(user_request, str) or not user_request.strip():
        reject("empty_request", "user_request must be a non-empty string")
        return CandidateValidation(tuple(codes), tuple(messages))
    request = " ".join(user_request.split())
    lowered = request.lower()
    if len(request) < 20 or len(request) > 600:
        reject("invalid_length", "request length must be between 20 and 600 characters")
    if any(ord(character) < 32 for character in user_request if character not in "\n\t"):
        reject("invalid_character", "request contains a control character")

    tickets = blueprint.initial_state["tickets"]
    allowed_ids = {
        str(row[key]).upper()
        for row in tickets
        for key in ("ticket_id", "customer_id", "order_id")
    }
    mentioned_ids = {match.upper() for match in re.findall(r"\b(?:T|C|O)-[A-Z0-9-]+\b", request, re.I)}
    if mentioned_ids - allowed_ids:
        reject("introduced_identifier", "request introduces an identifier absent from the episode")

    target_id = blueprint.goal_spec["target_ticket_id"]
    target = next(row for row in tickets if row["ticket_id"] == target_id)
    if blueprint.curriculum_mode == "indirect":
        if target_id.lower() in lowered:
            reject("target_ticket_leak", "indirect request exposes the hidden target ticket ID")
        if str(target[blueprint.lookup_mode]).lower() not in lowered:
            reject("missing_lookup", "indirect request omits the required lookup value")
    elif target_id.lower() not in lowered:
        reject("missing_lookup", "direct request omits the target ticket ID")

    field = blueprint.goal_spec["field"]
    field_labels = {
        "priority": ("priority",),
        "assigned_team": ("assigned_team", "assigned team", "team"),
        "status": ("status",),
    }
    mentioned_fields = {
        candidate
        for candidate, labels in field_labels.items()
        if any(label in lowered for label in labels)
    }
    if field not in mentioned_fields:
        reject("missing_mutation", "request omits the intended field mutation")
    if len(mentioned_fields) > 1:
        reject("multiple_mutations", "request asks for more than one field mutation")
    if blueprint.goal_spec["value"].lower() not in lowered:
        reject("missing_value", "request omits the intended new value")
    enum_values = {
        "priority": {"low", "normal", "high", "urgent"},
        "assigned_team": {"support", "billing", "finance", "logistics", "fraud"},
        "status": {"open", "in_progress", "resolved", "closed"},
    }
    patterns = {
        "priority": r"(?:set|change|make|mark|update)?\s*priority\s+(?:to|as|is|becomes?)\s+([a-z_]+)",
        "assigned_team": r"(?:set|change|make|mark|update)?\s*(?:assigned_team|assigned team|team)\s+(?:to|as|is|becomes?)\s+([a-z_]+)",
        "status": r"(?:set|change|make|mark|update)?\s*status\s+(?:to|as|is|becomes?)\s+([a-z_]+)",
    }
    observed_values = re.findall(patterns[field], lowered)
    if any(value not in enum_values[field] for value in observed_values):
        reject("introduced_enum", "request introduces an unsupported enum value")
    if not any(word in lowered for word in ("complete", "finish", "done", "close out")):
        reject("missing_finish", "request omits completion intent")
    if re.search(r"ticket_(?:query|update|finish)_tool|tool_name|\bjson\b|goal_spec|hidden goal", lowered):
        reject("tool_hint", "request exposes tool, JSON, or hidden-goal instructions")
    return CandidateValidation(tuple(codes), tuple(messages))
