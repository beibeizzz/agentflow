import json
import sys
import tempfile
import unittest
from pathlib import Path


PACKAGE_PARENT = Path(__file__).resolve().parents[2]
if str(PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PARENT))

from rewrite_calculator_prompts.pipeline import DatasetRewriter, PipelinePaths
from rewrite_calculator_prompts.schemas import ApiUsage, JsonResponse


SOURCE = {
    "pid": 1,
    "question": "Weng earns $12 an hour. She works for 50 minutes. How much does she earn?",
    "query": "old query",
    "answer": "12/60=<<12/60=0.2>>0.2; 0.2*50=<<0.2*50=10>>10. #### 10",
    "gold_answer": "10",
}

VALID_REWRITE = """Known facts:
- Weng earns $12 for one hour of babysitting.
- Weng works for 50 minutes.

Question:
- How many dollars does Weng earn?"""


class FakeClient:
    def __init__(self, rewrites, judges):
        self.rewrites = list(rewrites)
        self.judges = list(judges)
        self.rewrite_calls = []
        self.judge_calls = []

    def rewrite(self, source, prior_failures=None, temperature=0.3):
        self.rewrite_calls.append(list(prior_failures or []))
        payload = self.rewrites.pop(0)
        return JsonResponse(payload=payload, usage=ApiUsage(10, 5, 15))

    def judge(self, source, rewritten_question):
        self.judge_calls.append(rewritten_question)
        payload = self.judges.pop(0)
        return JsonResponse(payload=payload, usage=ApiUsage(8, 3, 11))


class PipelineTests(unittest.TestCase):
    def test_accepts_record_and_synchronizes_query(self) -> None:
        client = FakeClient(
            [{"rewritten_question": VALID_REWRITE}],
            [{"accepted": True, "reasons": []}],
        )
        rewriter = DatasetRewriter(client, max_attempts=3)

        result = rewriter.process_record(SOURCE)

        self.assertTrue(result.accepted)
        self.assertEqual(result.record["question"], VALID_REWRITE)
        self.assertIn(VALID_REWRITE, result.record["query"])
        self.assertNotIn("old query", result.record["query"])
        self.assertEqual(result.attempts, 1)
        self.assertEqual(result.usage.total_tokens, 26)

    def test_local_validation_failure_is_passed_to_next_rewrite(self) -> None:
        client = FakeClient(
            [
                {
                    "rewritten_question": (
                        "Known facts:\n- Weng earns $12 per hour.\n"
                        "- Her pay is 12 / 60 per minute.\n\n"
                        "Question:\n- How much does she earn in 50 minutes?"
                    )
                },
                {"rewritten_question": VALID_REWRITE},
            ],
            [{"accepted": True, "reasons": []}],
        )
        rewriter = DatasetRewriter(client, max_attempts=3)

        result = rewriter.process_record(SOURCE)

        self.assertTrue(result.accepted)
        self.assertEqual(result.attempts, 2)
        self.assertEqual(len(client.judge_calls), 1)
        self.assertTrue(client.rewrite_calls[1])
        self.assertIn("arithmetic", " ".join(client.rewrite_calls[1]).lower())

    def test_judge_rejection_is_passed_to_next_rewrite(self) -> None:
        client = FakeClient(
            [
                {"rewritten_question": VALID_REWRITE},
                {"rewritten_question": VALID_REWRITE},
            ],
            [
                {"accepted": False, "reasons": ["The target quantity changed."]},
                {"accepted": True, "reasons": []},
            ],
        )
        rewriter = DatasetRewriter(client, max_attempts=3)

        result = rewriter.process_record(SOURCE)

        self.assertTrue(result.accepted)
        self.assertEqual(result.attempts, 2)
        self.assertIn("target quantity changed", " ".join(client.rewrite_calls[1]).lower())

    def test_rejects_after_maximum_attempts(self) -> None:
        invalid = {
            "rewritten_question": (
                "Known facts:\n- Weng earns $12 per hour.\n"
                "- Her pay is 12 / 60 per minute.\n\n"
                "Question:\n- How much does she earn in 50 minutes?"
            )
        }
        client = FakeClient([invalid, invalid, invalid], [])
        rewriter = DatasetRewriter(client, max_attempts=3)

        result = rewriter.process_record(SOURCE)

        self.assertFalse(result.accepted)
        self.assertEqual(result.attempts, 3)
        self.assertEqual(len(client.rewrite_calls), 3)

    def test_run_writes_outputs_and_resume_skips_api_calls(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = PipelinePaths.in_directory(Path(tmp))
            first_client = FakeClient(
                [{"rewritten_question": VALID_REWRITE}],
                [{"accepted": True, "reasons": []}],
            )
            first = DatasetRewriter(first_client, max_attempts=3)

            summary = first.run([SOURCE], paths=paths)

            self.assertEqual(summary["accepted"], 1)
            dataset = json.loads(paths.dataset.read_text(encoding="utf-8"))
            self.assertEqual(dataset[0]["question"], VALID_REWRITE)
            self.assertTrue(paths.progress.exists())
            self.assertTrue(paths.summary.exists())

            second_client = FakeClient([], [])
            second = DatasetRewriter(second_client, max_attempts=3)
            resumed = second.run([SOURCE], paths=paths, resume=True)

            self.assertEqual(resumed["accepted"], 1)
            self.assertEqual(second_client.rewrite_calls, [])
            self.assertEqual(second_client.judge_calls, [])


if __name__ == "__main__":
    unittest.main()
