from __future__ import annotations

from enum import Enum


class FailureKind(str, Enum):
    MODEL_VALID = "model_valid"
    INFRASTRUCTURE_INVALID = "infrastructure_invalid"


class AgentFlowRuntimeError(Exception):
    failure_kind: FailureKind


class ModelValidError(AgentFlowRuntimeError):
    failure_kind = FailureKind.MODEL_VALID


class InfrastructureInvalidError(AgentFlowRuntimeError):
    failure_kind = FailureKind.INFRASTRUCTURE_INVALID


class ActionParseError(ModelValidError):
    """The model response is not one strict structured action."""
