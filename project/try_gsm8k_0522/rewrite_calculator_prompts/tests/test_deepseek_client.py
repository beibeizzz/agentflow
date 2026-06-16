import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


PACKAGE_PARENT = Path(__file__).resolve().parents[2]
if str(PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PARENT))

from rewrite_calculator_prompts.deepseek_client import DeepSeekClient
from rewrite_calculator_prompts.prompts import build_judge_messages, build_rewrite_messages


SOURCE = {
    "question": "Weng earns $12 an hour and works for 50 minutes. How much does she earn?",
    "answer": "12/60=<<12/60=0.2>>0.2 and 0.2*50=<<0.2*50=10>>10. #### 10",
    "gold_answer": "10",
}


def response(payload, *, finish_reason="stop", prompt_tokens=10, completion_tokens=5):
    content = payload if isinstance(payload, str) else json.dumps(payload)
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                finish_reason=finish_reason,
                message=SimpleNamespace(content=content),
            )
        ],
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


class FakeSdkClient:
    def __init__(self, outcomes):
        self.completions = FakeCompletions(outcomes)
        self.chat = SimpleNamespace(completions=self.completions)


class PromptTests(unittest.TestCase):
    def test_rewrite_prompt_uses_solution_as_hidden_reference(self) -> None:
        messages = build_rewrite_messages(SOURCE, ["Previous output leaked an equation."])
        combined = "\n".join(message["content"] for message in messages)

        self.assertIn(SOURCE["answer"], combined)
        self.assertIn("rewritten_question", combined)
        self.assertIn("Known facts:", combined)
        self.assertIn("Previous output leaked an equation.", combined)
        self.assertIn("JSON", combined)

    def test_judge_prompt_requests_strict_decision_schema(self) -> None:
        messages = build_judge_messages(
            SOURCE,
            "Known facts:\n- Weng earns $12 per hour.\n- Weng works for 50 minutes.\n\nQuestion:\n- How much does Weng earn?",
        )
        combined = "\n".join(message["content"] for message in messages)

        self.assertIn('"accepted"', combined)
        self.assertIn('"reasons"', combined)
        self.assertIn("at most three", combined)
        self.assertIn("Calculator", combined)


class DeepSeekClientTests(unittest.TestCase):
    def test_rewrite_uses_flash_with_thinking_disabled(self) -> None:
        sdk = FakeSdkClient([response({"rewritten_question": "Known facts:\n- A.\n\nQuestion:\n- B?"})])
        client = DeepSeekClient(sdk_client=sdk, sleep=lambda _: None)

        result = client.rewrite(SOURCE)

        self.assertEqual(result.payload["rewritten_question"], "Known facts:\n- A.\n\nQuestion:\n- B?")
        call = sdk.completions.calls[0]
        self.assertEqual(call["model"], "deepseek-v4-flash")
        self.assertEqual(call["response_format"], {"type": "json_object"})
        self.assertEqual(call["extra_body"]["thinking"]["type"], "disabled")

    def test_judge_uses_pro_with_thinking_enabled(self) -> None:
        sdk = FakeSdkClient([response({"accepted": True, "reasons": []})])
        client = DeepSeekClient(sdk_client=sdk, sleep=lambda _: None)

        result = client.judge(SOURCE, "Known facts:\n- A.\n\nQuestion:\n- B?")

        self.assertTrue(result.payload["accepted"])
        call = sdk.completions.calls[0]
        self.assertEqual(call["model"], "deepseek-v4-pro")
        self.assertEqual(call["extra_body"]["thinking"]["type"], "enabled")
        self.assertEqual(call["reasoning_effort"], "high")

    def test_empty_content_retries(self) -> None:
        sdk = FakeSdkClient(
            [
                response(""),
                response({"rewritten_question": "Known facts:\n- A.\n\nQuestion:\n- B?"}),
            ]
        )
        client = DeepSeekClient(sdk_client=sdk, max_transport_attempts=2, sleep=lambda _: None)

        result = client.rewrite(SOURCE)

        self.assertIn("rewritten_question", result.payload)
        self.assertEqual(len(sdk.completions.calls), 2)

    def test_transient_exception_retries(self) -> None:
        sdk = FakeSdkClient(
            [
                RuntimeError("temporary"),
                response({"accepted": False, "reasons": ["changed target"]}),
            ]
        )
        client = DeepSeekClient(sdk_client=sdk, max_transport_attempts=2, sleep=lambda _: None)

        result = client.judge(SOURCE, "Known facts:\n- A.\n\nQuestion:\n- B?")

        self.assertFalse(result.payload["accepted"])
        self.assertEqual(len(sdk.completions.calls), 2)

    def test_length_finish_reason_retries(self) -> None:
        sdk = FakeSdkClient(
            [
                response('{"accepted":', finish_reason="length"),
                response({"accepted": True, "reasons": []}),
            ]
        )
        client = DeepSeekClient(sdk_client=sdk, max_transport_attempts=2, sleep=lambda _: None)

        result = client.judge(SOURCE, "Known facts:\n- A.\n\nQuestion:\n- B?")

        self.assertTrue(result.payload["accepted"])
        self.assertEqual(len(sdk.completions.calls), 2)
        self.assertEqual(sdk.completions.calls[0]["max_tokens"], 8192)
        self.assertEqual(sdk.completions.calls[1]["max_tokens"], 16384)


if __name__ == "__main__":
    unittest.main()
