from __future__ import annotations

from typing import Any, Protocol

from .schemas import PlannerGeneration, RoleName


class PlannerGateway(Protocol):
    revision: str

    async def generate(
        self,
        *,
        request_id: str,
        system_prompt: str,
        prompt: str,
        sampling_params: dict[str, Any],
    ) -> PlannerGeneration: ...


class FrozenRoleGateway(Protocol):
    revision: str

    async def generate(
        self,
        *,
        role: RoleName,
        system_prompt: str,
        prompt: str,
        max_tokens: int,
    ) -> str: ...


__all__ = ["FrozenRoleGateway", "PlannerGateway"]
