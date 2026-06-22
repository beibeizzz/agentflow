from __future__ import annotations

import json
import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from try_ticket_agent.data_synthesis.api_client import DeepSeekClient


def response(payload, *, finish_reason="stop", prompt_tokens=10, completion_tokens=5):
    content = payload if isinstance(payload, str) else json.dumps(payload)
    return SimpleNamespace(
        choices=[SimpleNamespace(finish_reason=finish_reason, message=SimpleNamespace(content=content))],
        usage=SimpleNamespace(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )


class FakeCompletions:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class FakeSdk:
    def __init__(self, outcomes):
        self.completions = FakeCompletions(outcomes)
        self.chat = SimpleNamespace(completions=self.completions)


class SynthesisClientTests(unittest.TestCase):
    def test_rewrite_requests_strict_json_object_with_thinking_disabled(self) -> None:
        sdk = FakeSdk([response({"user_request": "Please update the ticket."})])
        client = DeepSeekClient(sdk_client=sdk, sleep=lambda _: None)
        result = client.rewrite({"episode_id": "e-1"}, prior_failures=["leaked target"])
        self.assertEqual(result.payload, {"user_request": "Please update the ticket."})
        call = sdk.completions.calls[0]
        self.assertEqual(call["response_format"], {"type": "json_object"})
        self.assertEqual(call["extra_body"], {"thinking": {"type": "disabled"}})

    def test_judge_uses_separate_model_and_high_reasoning(self) -> None:
        sdk = FakeSdk([response({"accepted": True, "reasons": []})])
        client = DeepSeekClient(sdk_client=sdk, sleep=lambda _: None)
        result = client.judge({"episode_id": "e-1"}, "Please update the ticket.")
        self.assertTrue(result.payload["accepted"])
        call = sdk.completions.calls[0]
        self.assertEqual(call["model"], "deepseek-v4-pro")
        self.assertEqual(call["reasoning_effort"], "high")
        self.assertEqual(call["extra_body"], {"thinking": {"type": "enabled"}})

    def test_transport_empty_and_truncated_responses_retry_and_accumulate_usage(self) -> None:
        sdk = FakeSdk(
            [
                response("", prompt_tokens=2, completion_tokens=1),
                response('{"user_request":', finish_reason="length", prompt_tokens=3, completion_tokens=2),
                response({"user_request": "Valid request"}, prompt_tokens=5, completion_tokens=4),
            ]
        )
        client = DeepSeekClient(sdk_client=sdk, max_transport_attempts=3, sleep=lambda _: None)
        result = client.rewrite({"episode_id": "e-1"})
        self.assertEqual(result.usage.total_tokens, 17)
        self.assertEqual(len(sdk.completions.calls), 3)
        self.assertGreater(sdk.completions.calls[2]["max_tokens"], sdk.completions.calls[1]["max_tokens"])

    def test_non_object_json_exhausts_transport_attempts(self) -> None:
        sdk = FakeSdk([response([]), response([])])
        client = DeepSeekClient(sdk_client=sdk, max_transport_attempts=2, sleep=lambda _: None)
        with self.assertRaisesRegex(RuntimeError, "JSON response must be an object"):
            client.rewrite({"episode_id": "e-1"})

    def test_client_without_injected_sdk_requires_environment_key(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "DEEPSEEK_API_KEY is not set"):
                DeepSeekClient()


if __name__ == "__main__":
    unittest.main()
