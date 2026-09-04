from __future__ import annotations

from typing import Any

from .retrieval import ResearchIndex
from .schemas import ResearchAction


class DeepResearchEnvironment:
    def __init__(self, index: ResearchIndex, *, top_k: int = 10) -> None:
        self.index = index
        self.top_k = top_k

    def execute(self, action: ResearchAction) -> dict[str, Any]:
        if action.tool_name == "Research_Search_Tool":
            query = str(action.arguments.get("query", "")).strip()
            if not query:
                return {"ok": False, "code": "EMPTY_QUERY"}
            hits = self.index.search(query, top_k=self.top_k)
            return {
                "ok": True,
                "data": [
                    {"doc_id": hit.doc_id, "title": hit.title, "score": hit.score, "snippet": hit.snippet}
                    for hit in hits
                ],
            }
        if action.tool_name == "Research_Read_Tool":
            doc_id = str(action.arguments.get("doc_id", "")).strip()
            start_sentence = int(action.arguments.get("start_sentence", 0))
            max_sentences = int(action.arguments.get("max_sentences", 20))
            if start_sentence < 0 or not 1 <= max_sentences <= 40:
                return {"ok": False, "code": "INVALID_READ_RANGE"}
            try:
                document = self.index.read(doc_id)
            except KeyError:
                return {"ok": False, "code": "DOCUMENT_NOT_FOUND", "doc_id": doc_id}
            stop_sentence = min(len(document.sentences), start_sentence + max_sentences)
            return {
                "ok": True,
                "data": {
                    "doc_id": document.doc_id,
                    "title": document.title,
                    "sentences": [
                        {"sentence_id": index, "text": sentence}
                        for index, sentence in enumerate(
                            document.sentences[start_sentence:stop_sentence],
                            start=start_sentence,
                        )
                    ],
                    "start_sentence": start_sentence,
                    "next_start_sentence": (
                        stop_sentence if stop_sentence < len(document.sentences) else None
                    ),
                    "total_sentences": len(document.sentences),
                },
            }
        if action.tool_name == "Research_Finish_Tool":
            return {"ok": True, "data": {"finish": True}}
        return {"ok": False, "code": "EXTERNAL_GENERATOR_REQUIRED"}
