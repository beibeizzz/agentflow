"""Task-independent execution backend contracts."""

from .bigcodebench import (
    BigCodeBenchBackend,
    BigCodeBenchExecution,
    DockerBigCodeBenchBackend,
)
from .contracts import (
    BaseGeneratorBackend,
    PageDocument,
    PageReaderBackend,
    SandboxBackend,
    SandboxRequest,
    SandboxResult,
    SearchBackend,
    SearchHit,
    WikipediaBackend,
    WikipediaDocument,
    WikipediaHit,
)
from .docker_sandbox import DockerSandboxBackend
from .persistent_sandbox import PersistentDockerSandboxPool
from .sandbox_http import HttpSandboxBackend

__all__ = [
    "BigCodeBenchBackend",
    "BigCodeBenchExecution",
    "DockerBigCodeBenchBackend",
    "BaseGeneratorBackend",
    "PageDocument",
    "PageReaderBackend",
    "SandboxBackend",
    "SandboxRequest",
    "SandboxResult",
    "SearchBackend",
    "SearchHit",
    "WikipediaBackend",
    "WikipediaDocument",
    "WikipediaHit",
    "DockerSandboxBackend",
    "PersistentDockerSandboxPool",
    "HttpSandboxBackend",
]
