"""Stable failure categories shared by tasks, tools, and training."""


class AgentFlowError(Exception):
    """Base error for the AgentFlow runtime."""


class ModelValidError(AgentFlowError):
    """A model-visible failure that remains valid for optimization."""


class ActionParseError(ModelValidError):
    """A generated action or final answer violates its public schema."""


class ToolDispatchError(ModelValidError):
    """A valid action requests an unregistered tool."""


class InfrastructureError(AgentFlowError):
    """A backend failure invalidates the rollout for optimization."""


class RateLimitError(InfrastructureError):
    """A shared external service rejected work because of rate limits."""


class PromptBudgetError(InfrastructureError):
    """Required model context cannot fit without losing essential input."""


class RevisionMismatchError(InfrastructureError):
    """A service response came from a different immutable environment revision."""


class PrivacyBoundaryError(AgentFlowError):
    """Evaluator-private content crossed into a model-visible record."""


__all__ = [
    "ActionParseError",
    "AgentFlowError",
    "InfrastructureError",
    "ModelValidError",
    "PrivacyBoundaryError",
    "PromptBudgetError",
    "RateLimitError",
    "RevisionMismatchError",
    "ToolDispatchError",
]
