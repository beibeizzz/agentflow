from __future__ import annotations

import json
from pathlib import Path
import unittest

from try_ticket_agent.run_ticket_agentflow import atomic_write_json, load_rows, summarize_results


class TicketBaselineCliTests(unittest.TestCase):
    def test_load_rows_supports_json_and_jsonl(self) -> None:
        root = Path(__file__).parent
        rows = [{"episode_id": "E-1"}, {"episode_id": "E-2"}]
        json_path = root / "_runtime_rows.json"
        jsonl_path = root / "_runtime_rows.jsonl"
        self.addCleanup(json_path.unlink, missing_ok=True)
        self.addCleanup(jsonl_path.unlink, missing_ok=True)
        json_path.write_text(json.dumps(rows), encoding="utf-8")
        jsonl_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
        self.assertEqual(load_rows(json_path), rows)
        self.assertEqual(load_rows(jsonl_path), rows)

    def test_atomic_write_json_replaces_target(self) -> None:
        path = Path(__file__).parent / "_runtime_summary.json"
        temporary = path.with_suffix(".json.tmp")
        self.addCleanup(path.unlink, missing_ok=True)
        self.addCleanup(temporary.unlink, missing_ok=True)
        atomic_write_json(path, {"value": 1})
        atomic_write_json(path, {"value": 2})
        self.assertEqual(json.loads(path.read_text(encoding="utf-8")), {"value": 2})
        self.assertFalse(temporary.exists())

    def test_summary_reports_exact_binary_success(self) -> None:
        summary = summarize_results(
            [
                {"reward": 1.0, "step_count": 2, "verification": {"success": True}},
                {"reward": 0.0, "step_count": 3, "verification": {"success": False}},
            ]
        )
        self.assertEqual(summary["total"], 2)
        self.assertEqual(summary["success_count"], 1)
        self.assertEqual(summary["episode_success_rate"], 0.5)
        self.assertEqual(summary["average_steps"], 2.5)


if __name__ == "__main__":
    unittest.main()
