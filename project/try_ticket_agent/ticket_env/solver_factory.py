from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable

from agentflow.models.memory import Memory
from agentflow.solver import construct_solver
from agentflow.tools.ticket_common.backend import TicketBackend

from .episode_io import EpisodeSpec, parse_episode
from .verifier import TicketVerifier

TICKET_QUERY_ANALYSIS_PROMPT = """Summarize the ticket workflow in one short plan.
User request: {question}
Available tools: {available_tools}

Rules:
- Direct: update the ticket, then finish.
- Indirect: query by customer_id or order_id, update the returned ticket, then finish.
- Track only lookup key, target field/value, and finish outcome completed.
- Keep it concise.
"""

TICKET_QUERY_ANALYSIS_SYSTEM_PROMPT = "Plan concise ticket workflows."

TICKET_NEXT_STEP_PROMPT = """{think_directive}Next ticket action.
Request: {question}
Plan: {query_analysis}
Previous steps: {memory_actions}

Choose exactly one tool.
Return exactly one JSON object with exactly these top-level keys:
{{
  "tool_name": "Ticket_Query_Tool | Ticket_Update_Tool | Ticket_Finish_Tool",
  "arguments": {{}}
}}

Do not output multiple tools.
Do not use tool names as top-level keys.
Do not output markdown or prose.

Argument formats:
- Ticket_Query_Tool: {{"lookup_by": "ticket_id|customer_id|order_id", "value": "..."}}
- Ticket_Update_Tool: {{"ticket_id": "...", "field": "priority|assigned_team|status", "value": "..."}}
- Ticket_Finish_Tool: {{"ticket_id": "...", "outcome": "completed"}}

Rules:
- If no previous step and request has ticket_id: use Ticket_Update_Tool.
- If no previous step and request has customer_id or order_id: use Ticket_Query_Tool.
- If query OK: use Ticket_Update_Tool with result data.ticket_id.
- If update OK: use Ticket_Finish_Tool with same ticket_id and outcome completed.
- Never repeat an OK query or OK update.
"""

TICKET_NEXT_STEP_SYSTEM_PROMPT = "Choose the next ticket tool call."


TICKET_TOOL_NAMES = ["Ticket_Query_Tool", "Ticket_Update_Tool", "Ticket_Finish_Tool"]


@dataclass
class TicketRuntime:
    solver: Any
    backend: TicketBackend
    verifier: TicketVerifier

    def reset_episode(self, episode: EpisodeSpec) -> None:
        self.backend.reset(episode.initial_state, episode.goal_spec)
        self.solver.memory = Memory()
        self.solver.max_steps = episode.max_steps
        self.verifier.max_steps = episode.max_steps

    def run_episode(self, row: dict[str, Any]) -> dict[str, Any]:
        episode = parse_episode(row)
        self.reset_episode(episode)
        solver_result = self.solver.solve(episode.user_request)
        step_count = int(solver_result.get("step_count", 0))
        verification = self.verifier.verify_final(self.solver.memory, step_count)
        return {
            **solver_result,
            "episode_id": episode.episode_id,
            "reward": 1.0 if verification.success else 0.0,
            "verification": verification.to_dict(),
            "state_diff": self.backend.state_diff(),
            "action_log": [asdict(event) for event in self.backend.action_log],
        }


def construct_ticket_runtime(
    *,
    llm_engine_name: str,
    base_url: str,
    max_steps: int = 3,
    max_time: int = 120,
    max_tokens: int = 512,
    temperature: float = 0.0,
    think_mode: str = "off",
    query_analysis_think_mode: str = "on",
    solver_builder: Callable[..., Any] = construct_solver,
) -> TicketRuntime:
    solver = solver_builder(
        llm_engine_name=llm_engine_name,
        base_url=base_url,
        enabled_tools=list(TICKET_TOOL_NAMES),
        tool_engine=["Default", "Default", "Default"],
        model_engine=["trainable", "trainable", "trainable", "trainable"],
        output_types="workflow",
        max_steps=max_steps,
        max_time=max_time,
        max_tokens=max_tokens,
        temperature=temperature,
        think_mode=think_mode,
        query_analysis_think_mode=query_analysis_think_mode,
        planner_action_mode="structured",
        executor_mode="structured",
        verbose=False,
    )

    planner = getattr(solver, "planner", None)
    if planner is not None:
        generation_configs = dict(getattr(planner, "generation_configs", {}) or {})
        generation_configs["query_analysis"] = {
            "prompt_template": TICKET_QUERY_ANALYSIS_PROMPT,
            "system_prompt": TICKET_QUERY_ANALYSIS_SYSTEM_PROMPT,
            "max_tokens": 192,
            "temperature": 0.0,
            "top_p": 0.95,
        }
        generation_configs["planner_next_step"] = {
            "prompt_template": TICKET_NEXT_STEP_PROMPT,
            "system_prompt": TICKET_NEXT_STEP_SYSTEM_PROMPT,
            "max_tokens": 192,
            "temperature": 0.0,
            "top_p": 0.95,
        }
        planner.generation_configs = generation_configs

    backend = TicketBackend()
    for tool_name in TICKET_TOOL_NAMES:
        tool = solver.executor.tool_instances_cache.get(tool_name)
        if tool is None or not hasattr(tool, "bind_backend"):
            raise RuntimeError(f"Ticket tool {tool_name!r} is unavailable or cannot bind a backend")
        tool.bind_backend(backend)
    verifier = TicketVerifier(backend, max_steps=max_steps)
    solver.verifier = verifier
    return TicketRuntime(solver=solver, backend=backend, verifier=verifier)
