from __future__ import annotations

from agentflow_rl.tasks.ticket.schemas import TicketEpisode


def validate_synthesized_episode(episode: TicketEpisode) -> None:
    tickets = episode.initial_state.tickets
    targets = [ticket for ticket in tickets if ticket.ticket_id == episode.goal_spec.target_ticket_id]
    if len(targets) != 1:
        raise ValueError("hidden goal must identify exactly one ticket")
    target = targets[0]
    lookup_value = getattr(target, episode.lookup_mode)
    matches = [ticket for ticket in tickets if getattr(ticket, episode.lookup_mode) == lookup_value]
    if len(matches) != 1:
        raise ValueError("lookup key must identify exactly one ticket")
    if lookup_value not in episode.user_request:
        raise ValueError("request must preserve the exact lookup identifier")
    if episode.goal_spec.field not in episode.user_request:
        raise ValueError("request must state the target field")
    if episode.goal_spec.value not in episode.user_request:
        raise ValueError("request must state the target value")
    if "complete" not in episode.user_request.lower():
        raise ValueError("request must preserve the completion instruction")
    if (
        episode.lookup_mode != "ticket_id"
        and episode.goal_spec.target_ticket_id in episode.user_request
    ):
        raise ValueError("indirect request must not reveal the hidden ticket identifier")
