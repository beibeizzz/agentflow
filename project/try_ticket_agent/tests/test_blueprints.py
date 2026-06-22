from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import unittest

from try_ticket_agent.data_synthesis.blueprints import (
    execute_reference_actions,
    generate_blueprint,
    validate_blueprint_collection,
)
from try_ticket_agent.scripts.generate_blueprints import write_blueprint_dataset
from try_ticket_agent.scripts.validate_dataset import validate_blueprint_directory


class BlueprintTests(unittest.TestCase):
    def setUp(self) -> None:
        self.output_dir = Path(__file__).parent / "_blueprint_test_output"
        self.output_dir.mkdir(exist_ok=True)
        for path in self.output_dir.iterdir():
            path.unlink()

    def tearDown(self) -> None:
        for path in self.output_dir.iterdir():
            path.unlink()
        self.output_dir.rmdir()

    def test_generation_is_byte_deterministic(self) -> None:
        left = [generate_blueprint(seed=42, split="train", index=i).to_dict() for i in range(20)]
        right = [generate_blueprint(seed=42, split="train", index=i).to_dict() for i in range(20)]
        self.assertEqual(
            json.dumps(left, sort_keys=True, separators=(",", ":")),
            json.dumps(right, sort_keys=True, separators=(",", ":")),
        )

    def test_every_hundred_blueprints_has_eighty_twenty_curriculum(self) -> None:
        items = [generate_blueprint(seed=42, split="train", index=i) for i in range(100)]
        self.assertEqual(Counter(item.curriculum_mode for item in items), {"direct": 80, "indirect": 20})
        self.assertEqual(
            Counter(item.lookup_mode for item in items),
            {"ticket_id": 80, "customer_id": 10, "order_id": 10},
        )

    def test_episode_has_six_to_ten_tickets_and_unique_lookup(self) -> None:
        for index in range(25):
            item = generate_blueprint(seed=17, split="train", index=index)
            tickets = item.initial_state["tickets"]
            self.assertGreaterEqual(len(tickets), 6)
            self.assertLessEqual(len(tickets), 10)
            if item.curriculum_mode == "indirect":
                target = next(row for row in tickets if row["ticket_id"] == item.goal_spec["target_ticket_id"])
                value = target[item.lookup_mode]
                self.assertEqual(sum(row[item.lookup_mode] == value for row in tickets), 1)
                self.assertNotIn(item.goal_spec["target_ticket_id"], item.canonical_request)

    def test_reference_actions_pass_real_tools_and_verifier(self) -> None:
        for index in range(10):
            verification = execute_reference_actions(generate_blueprint(seed=5, split="train", index=index))
            self.assertTrue(verification.success, verification.failure_codes)

    def test_splits_have_disjoint_ids_states_and_signatures(self) -> None:
        items = [
            generate_blueprint(seed=42, split=split, index=index)
            for split in ("train", "validation", "test")
            for index in range(20)
        ]
        report = validate_blueprint_collection(items)
        self.assertTrue(report["ok"], report)
        self.assertEqual(report["duplicate_episode_ids"], [])
        self.assertEqual(report["duplicate_state_hashes"], [])
        self.assertEqual(report["duplicate_request_goal_signatures"], [])

    def test_dataset_writer_emits_jsonl_and_verified_manifest(self) -> None:
        manifest = write_blueprint_dataset(
            output_dir=self.output_dir,
            seed=42,
            counts={"smoke": 2, "train": 5, "validation": 3, "test": 4},
        )
        self.assertEqual(manifest["counts"], {"smoke": 2, "train": 5, "validation": 3, "test": 4})
        self.assertTrue(manifest["reference_validation"]["ok"])
        for split, expected_count in manifest["counts"].items():
            path = self.output_dir / f"{split}.jsonl"
            rows = path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(rows), expected_count)
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), manifest["sha256"][split])
        self.assertEqual(
            json.loads((self.output_dir / "manifest.json").read_text(encoding="utf-8")),
            manifest,
        )

    def test_directory_validator_rejects_cross_split_collision(self) -> None:
        write_blueprint_dataset(
            output_dir=self.output_dir,
            seed=42,
            counts={"smoke": 1, "train": 2, "validation": 2, "test": 2},
        )
        train_row = (self.output_dir / "train.jsonl").read_text(encoding="utf-8").splitlines()[0]
        with (self.output_dir / "validation.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(train_row + "\n")
        report = validate_blueprint_directory(self.output_dir)
        self.assertFalse(report["ok"])
        self.assertTrue(report["duplicate_episode_ids"])


if __name__ == "__main__":
    unittest.main()
