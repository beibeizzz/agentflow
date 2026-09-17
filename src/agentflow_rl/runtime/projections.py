from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Callable

from agentflow_rl.roles.schemas import RoleName

from .contracts import MemoryAudience
from .memory import MemoryStore, MemoryView
from .observations import role_event_content

MEMORY_VIEW_REVISION = "role-memory-view-v4"


@dataclass(frozen=True)
class RoleViewSpec:
    max_tokens: int
    include_roles: tuple[str, ...]
    include_kinds: tuple[str, ...]
    required_tags: tuple[str, ...] = ("identity", "analysis")
    required_latest_tags: tuple[str, ...] = ("latest_result", "latest_judgement", "executed_action", "planner_action")
    max_recent_events: int = 1000


DEFAULT_ROLE_VIEWS = {
    RoleName.PLANNER: RoleViewSpec(
        max_tokens=6144,
        include_roles=("user", "query_analyzer", "planner", "executor", "verifier"),
        include_kinds=(
            "task",
            "analysis",
            "action",
            "executed_action",
            "tool_result",
            "action_error",
            "verifier_decision",
        ),
    ),
    RoleName.EXECUTOR: RoleViewSpec(
        max_tokens=4096,
        include_roles=("user", "query_analyzer", "executor", "verifier"),
        include_kinds=(
            "task",
            "analysis",
            "executed_action",
            "tool_result",
            "action_error",
            "verifier_decision",
        ),
        max_recent_events=6,
    ),
    RoleName.VERIFIER: RoleViewSpec(
        max_tokens=6144,
        include_roles=("user", "query_analyzer", "planner", "executor", "verifier"),
        include_kinds=(
            "task",
            "analysis",
            "action",
            "executed_action",
            "tool_result",
            "action_error",
            "verifier_decision",
        ),
    ),
    RoleName.GENERATOR: RoleViewSpec(
        max_tokens=6144,
        include_roles=("user", "query_analyzer", "planner", "executor", "verifier"),
        include_kinds=(
            "task",
            "analysis",
            "action",
            "executed_action",
            "tool_result",
            "action_error",
            "verifier_decision",
        ),
    ),
}


class RoleMemoryProjector:
    def __init__(
        self,
        specs: dict[RoleName, RoleViewSpec] | None = None,
        *,
        token_counter: Callable[[str], int] | None = None,
    ) -> None:
        self.specs = dict(specs or DEFAULT_ROLE_VIEWS)
        self.token_counter = token_counter
        missing = {
            RoleName.PLANNER,
            RoleName.EXECUTOR,
            RoleName.VERIFIER,
            RoleName.GENERATOR,
        } - set(self.specs)
        if missing:
            raise ValueError(f"missing role Memory view specs: {sorted(missing)}")

    def project(self, memory: MemoryStore, role: RoleName, *, max_tokens: int | None = None,
                token_counter: Callable[[str], int] | None = None,
                omit_task: bool = False) -> MemoryView:
        spec = self.specs[role]
        counter = token_counter or self.token_counter
        kwargs = {"token_counter": counter} if counter else {}
        kinds = tuple(kind for kind in spec.include_kinds if not (omit_task and kind == "task"))
        selected_ids = None
        header = ""
        current_code = None
        if role is RoleName.GENERATOR:
            events = memory.events
            judgements = [e for e in events if e.kind == "verifier_decision"]
            results = [e for e in events if e.kind == "tool_result"]
            refs = {ref for e in judgements if isinstance(e.content, dict)
                    for ref in e.content.get("evidence_ids", [])}
            selected_ids = {e.event_id for e in events if e.kind in {"task", "analysis"}}
            selected_ids.update(refs)
            if judgements:
                selected_ids.add(judgements[-1].event_id)
            if results:
                selected_ids.add(results[-1].event_id)
            coded = [e for e in results if isinstance(e.content, dict)
                     and isinstance(e.content.get("data"), dict) and e.content["data"].get("code")]
            if coded:
                selected_ids.add(coded[-1].event_id)
                current_code = coded[-1].content["data"]["code"]
            selected_turns = {e.turn_index for e in events if e.event_id in selected_ids}
            selected_ids.update(e.event_id for e in events
                                if e.kind in {"action", "executed_action", "action_error"} and e.turn_index in selected_turns)
            header = ("Evidence selected through Verifier evidence references and latest feedback. "
                      "Some historical evidence may be omitted by the context budget. "
                      "Unselected evidence is unverified. Match test feedback to its code_revision.")
        view = memory.project(
            audience=MemoryAudience.MODEL,
            max_tokens=spec.max_tokens if max_tokens is None else max_tokens,
            required_tags=spec.required_tags,
            required_latest_tags=spec.required_latest_tags,
            include_roles=spec.include_roles,
            include_kinds=kinds,
            max_recent_events=spec.max_recent_events,
            include_event_ids=selected_ids,
            header=header,
            content_projector=lambda kind, content: role_event_content(role.value, kind, content),
            **kwargs,
        )
        if current_code and json.dumps(current_code, ensure_ascii=False) not in view.text:
            raise ValueError("memory budget cannot retain complete current code for Generator")
        return view

    def render(self, memory: MemoryStore, role: RoleName) -> str:
        return self.project(memory, role).text


__all__ = ["DEFAULT_ROLE_VIEWS", "RoleMemoryProjector", "RoleViewSpec"]
