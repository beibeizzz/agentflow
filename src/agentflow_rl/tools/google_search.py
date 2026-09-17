from __future__ import annotations

import asyncio
from time import monotonic
from agentflow_rl.backends.contracts import PageReaderBackend, SearchBackend
from agentflow_rl.runtime.contracts import ToolName
from .catalog import SearchArguments
from .contracts import BackendCall, ToolFailureCode, ToolRequest, ToolResult, ToolStatus
from .retrieval import RetrievalPolicy, read_source


class GoogleSearchTool(RetrievalPolicy):
    name = ToolName.GOOGLE_SEARCH

    def __init__(self, search_backend: SearchBackend, reader_backend: PageReaderBackend, **policy) -> None:
        super().__init__(**policy)
        self.search_backend, self.reader_backend = search_backend, reader_backend
        self.revision = f"google-search-tool-v2:{self.policy_revision}:{search_backend.revision}:{reader_backend.revision}"

    async def execute(self, request: ToolRequest) -> ToolResult:
        args = SearchArguments.model_validate(request.arguments)
        started = monotonic()
        hits = await self.search_backend.search(query=args.query, top_k=args.top_k)
        calls = [BackendCall(stage="search", arguments=args.model_dump(), latency_ms=(monotonic() - started) * 1000)]
        selected, seen = [], set()
        for hit in hits:
            if hit.url not in seen:
                seen.add(hit.url)
                selected.append(hit)
            if len(selected) == self.read_top_k:
                break
        rows = await asyncio.gather(*(read_source(
            self.reader_backend.open, {"url": hit.url, "result_id": hit.result_id},
            {"result_id": hit.result_id, "url": hit.url, "title": hit.title},
            **self.read_kwargs(request)) for hit in selected))
        documents = [row[0] for row in rows]
        calls.extend(row[1] for row in rows)
        failed = next((call for call in calls if call.status is ToolStatus.INFRASTRUCTURE_ERROR), None)
        failure = failed.failure_code if failed else (ToolFailureCode.RETRIEVAL_MISS if not hits else None)
        status = ToolStatus.INFRASTRUCTURE_ERROR if failed else ToolStatus.MODEL_ERROR if failure else ToolStatus.SUCCESS
        return ToolResult(request_id=request.request_id, tool_name=self.name, status=status, failure_code=failure,
                          data={"hits": [hit.model_dump(mode="json") for hit in hits], "documents": documents},
                          backend_revision=self.revision, latency_ms=(monotonic() - started) * 1000,
                          truncated=any(doc.get("truncated", False) for doc in documents), backend_calls=tuple(calls))
