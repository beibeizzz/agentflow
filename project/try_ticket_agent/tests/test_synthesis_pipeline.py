from __future__ import annotations

import json
from pathlib import Path
import unittest

from try_ticket_agent.data_synthesis.api_client import ApiUsage, JsonResponse
from try_ticket_agent.data_synthesis.blueprints import generate_blueprint
from try_ticket_agent.data_synthesis.pipeline import PipelinePaths, TicketSynthesisPipeline
from try_ticket_agent.data_synthesis.validators import validate_candidate
from try_ticket_agent.scripts.synthesize_dataset import (
    synthesis_paths_for_split,
    write_synthesis_manifest,
)
from try_ticket_agent.scripts.validate_dataset import validate_synthesized_directory


class FakeClient:
    def __init__(self, rewrites, judges):
        self.rewrites = list(rewrites)
        self.judges = list(judges)
        self.rewrite_calls = []
        self.judge_calls = []

    def rewrite(self, source, prior_failures=None, temperature=0.3):
        self.rewrite_calls.append(list(prior_failures or []))
        return JsonResponse(self.rewrites.pop(0), ApiUsage(10, 5, 15))

    def judge(self, source, user_request):
        self.judge_calls.append(user_request)
        return JsonResponse(self.judges.pop(0), ApiUsage(8, 3, 11))


class SynthesisPipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.output_dir = Path(__file__).parent / "_synthesis_test_output"
        self.output_dir.mkdir(exist_ok=True)
        for path in self.output_dir.iterdir():
            path.unlink()

    def tearDown(self) -> None:
        for path in self.output_dir.iterdir():
            path.unlink()
        self.output_dir.rmdir()

    def test_validator_rejects_indirect_ticket_leak_and_second_update(self) -> None:
        blueprint = generate_blueprint(seed=42, split="train", index=4)
        leaked = (
            f"For ticket {blueprint.goal_spec['target_ticket_id']}, set priority urgent "
            "and status resolved."
        )
        result = validate_candidate(blueprint, leaked)
        self.assertTrue({"target_ticket_leak", "multiple_mutations"} <= set(result.codes))

    def test_validator_rejects_introduced_identifier_and_tool_hint(self) -> None:
        blueprint = generate_blueprint(seed=42, split="train", index=0)
        request = blueprint.canonical_request + " Use Ticket_Update_Tool on T-FAKE-999."
        result = validate_candidate(blueprint, request)
        self.assertTrue({"introduced_identifier", "tool_hint"} <= set(result.codes))

    def test_validator_rejects_introduced_enum_value(self) -> None:
        blueprint = generate_blueprint(seed=42, split="train", index=0)
        request = blueprint.canonical_request + " Treat the priority as critical."
        result = validate_candidate(blueprint, request)
        self.assertIn("introduced_enum", result.codes)

    def test_validator_allows_field_reference_without_assignment_enum_capture(self) -> None:
        blueprint = generate_blueprint(seed=42, split="train", index=2)
        target_id = blueprint.goal_spec["target_ticket_id"]
        value = blueprint.goal_spec["value"]
        request = (
            f"For ticket {target_id}, update the status of the ticket so it becomes {value}, "
            "then finish the request."
        )
        result = validate_candidate(blueprint, request)
        self.assertNotIn("introduced_enum", result.codes)
        self.assertTrue(result.ok, result.messages)

    def test_validator_still_rejects_explicit_unsupported_assignment_enum(self) -> None:
        blueprint = generate_blueprint(seed=42, split="train", index=2)
        target_id = blueprint.goal_spec["target_ticket_id"]
        request = f"For ticket {target_id}, set status to pending, then finish the request."
        result = validate_candidate(blueprint, request)
        self.assertIn("introduced_enum", result.codes)

    def test_pipeline_retries_with_feedback_and_resumes_without_api_calls(self) -> None:
        blueprint = generate_blueprint(seed=42, split="train", index=0)
        bad = {"user_request": blueprint.canonical_request + " Also set status to resolved."}
        good = {"user_request": blueprint.canonical_request}
        client = FakeClient([bad, good], [{"accepted": True, "reasons": []}])
        pipeline = TicketSynthesisPipeline(client, max_attempts=3)
        paths = PipelinePaths.in_directory(self.output_dir)
        first = pipeline.run([blueprint], paths=paths, resume=True)
        self.assertEqual(first["accepted"], 1)
        self.assertEqual(first["api_calls"], 3)
        self.assertTrue(client.rewrite_calls[1])

        second_client = FakeClient([], [])
        second = TicketSynthesisPipeline(second_client, max_attempts=3)
        resumed = second.run([blueprint], paths=paths, resume=True)
        self.assertEqual(resumed["accepted"], 1)
        self.assertEqual(resumed["api_calls"], 0)

    def test_resume_rejects_same_episode_id_with_changed_blueprint_fingerprint(self) -> None:
        original = generate_blueprint(seed=42, split="train", index=0)
        changed = generate_blueprint(seed=99, split="train", index=0)
        client = FakeClient(
            [{"user_request": original.canonical_request}],
            [{"accepted": True, "reasons": []}],
        )
        paths = PipelinePaths.in_directory(self.output_dir)
        TicketSynthesisPipeline(client).run([original], paths=paths, resume=True)

        second_client = FakeClient(
            [{"user_request": changed.canonical_request}],
            [{"accepted": True, "reasons": []}],
        )
        with self.assertRaisesRegex(RuntimeError, "fingerprint mismatch"):
            TicketSynthesisPipeline(second_client).run([changed], paths=paths, resume=True)
        self.assertEqual(second_client.rewrite_calls, [])
        self.assertEqual(second_client.judge_calls, [])

    def test_resume_rejects_extra_stale_progress_record(self) -> None:
        first = generate_blueprint(seed=42, split="train", index=0)
        second = generate_blueprint(seed=42, split="train", index=1)
        client = FakeClient(
            [
                {"user_request": first.canonical_request},
                {"user_request": second.canonical_request},
            ],
            [
                {"accepted": True, "reasons": []},
                {"accepted": True, "reasons": []},
            ],
        )
        paths = PipelinePaths.in_directory(self.output_dir)
        TicketSynthesisPipeline(client).run([first, second], paths=paths, resume=True)

        resume_client = FakeClient([], [])
        with self.assertRaisesRegex(RuntimeError, "stale progress"):
            TicketSynthesisPipeline(resume_client).run([first], paths=paths, resume=True)
        self.assertEqual(resume_client.rewrite_calls, [])
        self.assertEqual(resume_client.judge_calls, [])

    def test_resume_false_clears_stale_progress_and_recomputes(self) -> None:
        original = generate_blueprint(seed=42, split="train", index=0)
        changed = generate_blueprint(seed=99, split="train", index=0)
        paths = PipelinePaths.in_directory(self.output_dir)
        TicketSynthesisPipeline(
            FakeClient(
                [{"user_request": original.canonical_request}],
                [{"accepted": True, "reasons": []}],
            )
        ).run([original], paths=paths, resume=True)

        client = FakeClient(
            [{"user_request": changed.canonical_request}],
            [{"accepted": True, "reasons": []}],
        )
        summary = TicketSynthesisPipeline(client).run([changed], paths=paths, resume=False)
        rows = [json.loads(line) for line in paths.dataset.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(summary["accepted"], 1)
        self.assertEqual(summary["api_calls"], 2)
        self.assertEqual([row["episode_id"] for row in rows], [changed.episode_id])

    def test_judge_schema_error_is_feedback_for_next_attempt(self) -> None:
        blueprint = generate_blueprint(seed=42, split="train", index=1)
        candidate = {"user_request": blueprint.canonical_request}
        client = FakeClient(
            [candidate, candidate],
            [{"accepted": "yes", "reasons": []}, {"accepted": True, "reasons": []}],
        )
        result = TicketSynthesisPipeline(client, max_attempts=2).process_blueprint(blueprint)
        self.assertTrue(result.accepted)
        self.assertIn("Judge JSON schema error", client.rewrite_calls[1][0])

    def test_judge_rejection_is_feedback_for_next_attempt(self) -> None:
        blueprint = generate_blueprint(seed=42, split="train", index=2)
        candidate = {"user_request": blueprint.canonical_request}
        client = FakeClient(
            [candidate, candidate],
            [
                {"accepted": False, "reasons": ["Completion intent is ambiguous."]},
                {"accepted": True, "reasons": []},
            ],
        )
        result = TicketSynthesisPipeline(client, max_attempts=2).process_blueprint(blueprint)
        self.assertTrue(result.accepted)
        self.assertIn("Completion intent is ambiguous.", client.rewrite_calls[1])

    def test_permanent_rejection_writes_rejected_record_and_usage_total(self) -> None:
        blueprint = generate_blueprint(seed=42, split="train", index=3)
        invalid = {"user_request": "too short"}
        client = FakeClient([invalid, invalid], [])
        paths = PipelinePaths.in_directory(self.output_dir)
        summary = TicketSynthesisPipeline(client, max_attempts=2).run([blueprint], paths=paths)
        rejected = [json.loads(line) for line in paths.rejected.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(summary["rejected"], 1)
        self.assertEqual(summary["api_calls"], 2)
        self.assertEqual(summary["usage"]["total_tokens"], 30)
        self.assertEqual(rejected[0]["episode_id"], blueprint.episode_id)
        self.assertTrue(rejected[0]["reasons"])

    def test_split_paths_do_not_overwrite_other_splits(self) -> None:
        train = synthesis_paths_for_split(self.output_dir, "train")
        validation = synthesis_paths_for_split(self.output_dir, "validation")
        self.assertEqual(train.dataset.name, "train.jsonl")
        self.assertNotEqual(train.dataset, validation.dataset)
        self.assertNotEqual(train.progress, validation.progress)

    def test_run_keeps_input_order_and_excludes_reference_actions(self) -> None:
        blueprints = [generate_blueprint(seed=42, split="train", index=i) for i in range(3)]
        client = FakeClient(
            [{"user_request": item.canonical_request} for item in blueprints],
            [{"accepted": True, "reasons": []} for _ in blueprints],
        )
        paths = PipelinePaths.in_directory(self.output_dir)
        summary = TicketSynthesisPipeline(client).run(
            blueprints, paths=paths, concurrency=2, resume=False
        )
        rows = [json.loads(line) for line in paths.dataset.read_text(encoding="utf-8").splitlines()]
        self.assertEqual([row["episode_id"] for row in rows], [item.episode_id for item in blueprints])
        self.assertEqual(summary["accepted"], 3)
        for row in rows:
            self.assertNotIn("reference_actions", row)
            self.assertIn("initial_state", row)
            self.assertIn("goal_spec", row)

    def test_generated_dataset_is_reloaded_and_reference_validated(self) -> None:
        blueprint = generate_blueprint(seed=42, split="train", index=0)
        client = FakeClient(
            [{"user_request": blueprint.canonical_request}],
            [{"accepted": True, "reasons": []}],
        )
        paths = synthesis_paths_for_split(self.output_dir, "train")
        TicketSynthesisPipeline(client).run([blueprint], paths=paths)
        report = validate_synthesized_directory(self.output_dir)
        self.assertTrue(report["ok"], report)
        self.assertEqual(report["counts"], {"train": 1})
        self.assertEqual(report["reference_failures"], [])
        manifest = write_synthesis_manifest(
            self.output_dir, splits=["train"], summaries={"train": {"accepted": 1}}
        )
        self.assertEqual(manifest["counts"], {"train": 1})
        self.assertEqual(manifest["sha256"]["train"], report["sha256"]["train"])


if __name__ == "__main__":
    unittest.main()
