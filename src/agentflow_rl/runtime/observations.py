"""Pure model views. Execution records remain intact in Memory/artifact storage."""
from __future__ import annotations

from copy import deepcopy
from typing import Any

OBSERVATION_REVISION = "tool-observation-v3"

_PLANNER_HISTORY_KINDS = {
    "analysis",
    "action",
    "executed_action",
    "tool_result",
    "action_error",
    "verifier_decision",
}
_COMPACTED_VISIBLE_FIELDS = {
    "tool_name",
    "status",
    "failure_code",
    "outcome",
    "evidence_ids",
}


def pick(value: dict, fields: tuple[str, ...]) -> dict:
    return {key: deepcopy(value[key]) for key in fields if key in value}


def execution_view(value: dict) -> dict:
    result = pick(value, ("ok", "stdout", "stderr", "exit_code", "timed_out", "passed", "total"))
    if "failures" in value:
        result["failures"] = [pick(item, ("test_index", "error_type", "actual", "expected", "stderr"))
                              for item in value["failures"] if isinstance(item, dict)]
    return result


def action_view(value: dict | None) -> dict | None:
    return None if value is None else pick(value, ("sub_goal", "tool_name", "arguments"))


def tool_observation(value: dict | None) -> dict | None:
    if value is None:
        return None
    result = pick(value, ("tool_name", "status", "failure_code", "truncated"))
    failure = value.get("failure_code")
    messages = {
        "INVALID_ARGUMENTS": "Arguments failed validation; check the selected tool schema and supply complete inputs.",
        "EXECUTION_FAILED": "Execution or a supplied public test failed; inspect execution feedback.",
        "RETRIEVAL_MISS": "Search returned no results.",
        "EMPTY_RESULT": "The tool returned no usable content.",
        "TIMEOUT": "The tool service exceeded its time budget.",
        "RATE_LIMITED": "The tool service is rate limited.",
        "BACKEND_UNAVAILABLE": "The tool service is unavailable.",
        "INTERNAL_ERROR": "The tool service failed.",
    }
    if failure:
        result["message"] = messages.get(failure, "The tool call failed.")
        if value.get("message") == "Supplied tool context exceeds its input token budget; shorten the prompt.":
            result["message"] = value["message"]
    data = value.get("data", {})
    name = value.get("tool_name")
    clean: dict[str, Any] = {}
    if name == "Base_Generator_Tool":
        clean = pick(data, ("text",))
    elif name == "Python_Coder_Tool":
        clean = pick(data, ("code", "code_revision"))
        for key in ("execution",):
            if isinstance(data.get(key), dict):
                clean[key] = execution_view(data[key])
    elif name in {"Google_Search_Tool", "Wikipedia_Search_Tool"}:
        fields = (("result_id", "title", "url", "snippet", "published_at")
                  if name == "Google_Search_Tool" else ("doc_id", "passage_id", "title", "snippet"))
        clean = {"hits": [pick(hit, fields) for hit in data.get("hits", [])],
                 "documents": [pick(doc, ("result_id", "url", "canonical_url", "doc_id", "title",
                                           "passages", "passage_ids", "read_status", "failure_code",
                                           "message", "truncated")) for doc in data.get("documents", [])]}
    result["data"] = clean
    return result


def event_content(kind: str, content: Any) -> Any:
    if not isinstance(content, dict):
        return deepcopy(content)
    if kind == "tool_result" and "tool_name" in content:
        return tool_observation(content)
    if kind in {"action", "executed_action"}:
        return action_view(content)
    if kind == "task":
        return pick(content, ("task_name", "prompt", "public_payload"))
    if kind == "verifier_decision":
        return pick(content, ("outcome", "rationale", "evidence_ids", "failure_codes"))
    if kind == "action_error":
        return {"failure_code": content.get("failure_code", "INVALID_ACTION"),
                "message": "Action failed validation; use the tool schema and preserve the selected tool and sub-goal."}
    return deepcopy(content)


def event_view(event: dict) -> dict:
    result = pick(event, ("event_id", "turn_index", "role", "kind"))
    result["content"] = event_content(event.get("kind", ""), event.get("content"))
    return result


def role_event_content(role: str, kind: str, content: Any) -> Any:
    """Deterministic role-specific evidence; full records remain in Memory."""
    clean = event_content(kind, content)
    if not isinstance(clean, dict):
        return clean
    if kind == "executed_action":
        args = clean.get("arguments", {})
        if "code" in args:
            # Full code is carried once, in the corresponding tool result.
            clean["arguments"] = {"mode": "public_tests" if args.get("tests") else "run",
                                  "stdin": args.get("stdin", ""), "public_test_count": len(args.get("tests", []))}
    if kind != "tool_result":
        return clean
    data = clean.get("data", {})
    if role in {"planner", "verifier"}:
        data.pop("code", None)
    execution = data.get("execution")
    if isinstance(execution, dict):
        data["execution"] = {k: v for k, v in execution.items() if v is not None and v != "" and v != []}
    # Successful document reads carry evidence once. Unread hits retain snippets.
    documents = data.get("documents", [])
    if "hits" in data:
        def duplicated(hit):
            return any(d.get("read_status") == "success"
                       and d.get("result_id", d.get("doc_id")) == hit.get("result_id", hit.get("doc_id"))
                       and hit.get("snippet") and hit["snippet"] in "\n".join(d.get("passages", []))
                       for d in documents)
        data["hits"] = [hit for hit in data["hits"] if not duplicated(hit)]
    # Planner and Verifier consume bounded outcome evidence, with explicit cuts.
    if role in {"planner", "verifier"}:
        clipped = False
        remaining = 3000 if role == "planner" else 4000
        def bound(value):
            nonlocal remaining, clipped
            if isinstance(value, str):
                used = min(len(value), remaining)
                remaining -= used
                if used < len(value):
                    clipped = True
                    return {"text": value[:used], "truncated": True}
            return value
        if "text" in data:
            data["text"] = bound(data["text"])
        if isinstance(data.get("execution"), dict):
            for key in ("stdout", "stderr"):
                if key in data["execution"]:
                    data["execution"][key] = bound(data["execution"][key])
            failures = data["execution"].get("failures", [])
            if len(failures) > 3:
                data["execution"]["failures_omitted"] = len(failures) - 3
                data["execution"]["failures"] = failures[:3]
                clipped = True
            for failure in data["execution"].get("failures", []):
                for key in ("actual", "expected", "stderr"):
                    if key in failure:
                        value = failure[key]
                        if not isinstance(value, str):
                            import json
                            value = json.dumps(value, ensure_ascii=False)
                        bounded = bound(value)
                        if isinstance(bounded, dict):
                            failure[key] = bounded
        for doc in documents:
            doc["passages"] = [bound(p) for p in doc.get("passages", [])]
        for hit in data.get("hits", []):
            if "snippet" in hit:
                hit["snippet"] = bound(hit["snippet"])
        if clipped:
            clean["truncated"] = True
    return clean


def planner_core_history(
    events: tuple[dict[str, Any], ...],
    *,
    included_event_ids: tuple[str, ...],
    compacted_event_ids: tuple[str, ...],
) -> tuple[dict[str, Any], ...]:
    """Freeze the semantic subset that was visible to the Planner pre-action."""
    included = set(included_event_ids)
    compacted = set(compacted_event_ids)
    history: list[dict[str, Any]] = []
    for event in events:
        event_id = str(event.get("event_id", ""))
        kind = str(event.get("kind", ""))
        if event_id not in included or kind not in _PLANNER_HISTORY_KINDS:
            continue
        content = role_event_content("planner", kind, event.get("content"))
        row = pick(event, ("event_id", "turn_index", "role", "kind"))
        if event_id in compacted:
            visible = (
                {key: value for key, value in content.items() if key in _COMPACTED_VISIBLE_FIELDS}
                if isinstance(content, dict)
                else {}
            )
            row["content"] = {
                "compacted": True,
                "visible_fields": visible,
                "content_omitted": True,
            }
        else:
            row["content"] = content
        history.append(row)
    return tuple(history)
