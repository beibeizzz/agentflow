from __future__ import annotations

from pathlib import Path
import re
import subprocess
import sys
import unittest

import yaml
from try_ticket_agent.flowgrpo_general_2x40g.train_ticket_gspo import parse_args as parse_train_args


ROOT = Path(__file__).resolve().parents[1]
TRAIN_CONFIG = ROOT / "flowgrpo_general_2x40g" / "config_train_general_2x40g.yaml"
EVAL_CONFIG = ROOT / "flowgrpo_general_2x40g" / "config_eval_learnable_general_2x40g.yaml"
BASELINE_CONFIG = ROOT / "baseline" / "config_agentflow_baseline.yaml"
TRAIN_SCRIPT = ROOT / "flowgrpo_general_2x40g" / "run_train_general_2x40g.sh"
BASELINE_SCRIPT = ROOT / "baseline" / "run_agentflow_baseline.sh"
README = ROOT / "README.md"
DATA_README = ROOT / "data" / "README.md"


def load_yaml(path: Path):
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def shell_default(text: str, name: str) -> str:
    match = re.search(rf'^{name}="\$\{{{name}:-([^}}]+)\}}"$', text, re.MULTILINE)
    if not match:
        raise AssertionError(f"Missing shell default for {name}")
    return match.group(1)


class RemoteConfigTests(unittest.TestCase):
    def test_readmes_document_reproducible_no_sft_workflow(self) -> None:
        text = README.read_text(encoding="utf-8") + "\n" + DATA_README.read_text(encoding="utf-8")
        commands = [
            "scripts/generate_blueprints.py",
            "scripts/synthesize_dataset.py",
            "scripts/validate_dataset.py --dataset",
            "baseline/run_agentflow_baseline.sh",
            "run_train_general_2x40g.sh",
            "EVAL_MODE=baseline",
            "EVAL_MODE=adapter",
        ]
        for command in commands:
            self.assertIn(command, text)
        for statement in (
            "No SFT",
            "binary reward",
            "per-turn GSPO ratio",
            "offline-only synthesis judge",
            "legacy defaults remain unchanged",
        ):
            self.assertIn(statement, text)
    def test_train_cli_accepts_every_shell_override(self) -> None:
        args = parse_train_args(
            [
                "--config", str(TRAIN_CONFIG), "--model-path", "model", "--train-file", "train.jsonl",
                "--output-dir", "out", "--frozen-base-url", "http://localhost/v1",
                "--frozen-model", "Qwen3-0.6B", "--question-batch-size", "4", "--group-size", "8",
                "--rollout-concurrency", "32", "--planner-batch-size", "32", "--max-steps", "3",
                "--clip-range-low", "0.001", "--clip-range-high", "0.003", "--policy-epochs", "2",
                "--max-train-items", "2", "--epochs", "1",
            ]
        )
        self.assertEqual(args.group_size, 8)
        self.assertEqual(args.clip_range_high, 0.003)

    def test_python_entrypoints_support_direct_file_execution(self) -> None:
        scripts = [
            ROOT / "run_ticket_agentflow.py",
            ROOT / "flowgrpo_general_2x40g" / "train_ticket_gspo.py",
            ROOT / "flowgrpo_general_2x40g" / "eval_ticket_agent.py",
            ROOT / "scripts" / "synthesize_dataset.py",
        ]
        for script in scripts:
            result = subprocess.run(
                [sys.executable, str(script), "--help"],
                cwd=ROOT.parent,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, f"{script}: {result.stderr}")

    def test_train_config_matches_required_remote_defaults(self) -> None:
        cfg = load_yaml(TRAIN_CONFIG)
        self.assertEqual((cfg["question_batch_size"], cfg["group_size"]), (4, 8))
        self.assertEqual((cfg["rollout_concurrency"], cfg["planner_batch_size"]), (32, 32))
        self.assertEqual(cfg["max_steps"], 3)
        self.assertEqual((cfg["clip_range_low"], cfg["clip_range_high"]), (0.001, 0.003))
        self.assertEqual(cfg["reward_mode"], "binary")
        self.assertEqual((cfg["lora_rank"], cfg["lora_alpha"]), (64, 128))
        self.assertEqual(cfg["policy_epochs"], 2)

    def test_shell_defaults_match_train_yaml_and_are_passed_to_cli(self) -> None:
        cfg = load_yaml(TRAIN_CONFIG)
        text = TRAIN_SCRIPT.read_text(encoding="utf-8")
        mapping = {
            "QUESTION_BATCH_SIZE": "question_batch_size",
            "GROUP_SIZE": "group_size",
            "ROLLOUT_CONCURRENCY": "rollout_concurrency",
            "PLANNER_BATCH_SIZE": "planner_batch_size",
            "MAX_STEPS": "max_steps",
            "CLIP_RANGE_LOW": "clip_range_low",
            "CLIP_RANGE_HIGH": "clip_range_high",
            "POLICY_EPOCHS": "policy_epochs",
        }
        for environment, key in mapping.items():
            self.assertEqual(float(shell_default(text, environment)), float(cfg[key]))
            self.assertIn(f'"${environment}"', text)

    def test_baseline_and_eval_use_same_task_contract(self) -> None:
        baseline = load_yaml(BASELINE_CONFIG)
        evaluation = load_yaml(EVAL_CONFIG)
        training = load_yaml(TRAIN_CONFIG)
        for key in ("model_path", "frozen_model", "eval_file", "max_steps", "think_mode"):
            expected = training["train_file"] if key == "eval_file" else training[key]
            if key == "eval_file":
                expected = evaluation[key]
            self.assertEqual(baseline[key], expected)
        self.assertEqual(baseline["planner_action_mode"], "structured")
        self.assertEqual(baseline["executor_mode"], "structured")
        self.assertEqual(baseline["output_types"], "workflow")
        self.assertEqual(evaluation["output_types"], "workflow")
        baseline_script = BASELINE_SCRIPT.read_text(encoding="utf-8")
        self.assertIn("config_agentflow_baseline.yaml", baseline_script)
        self.assertIn("--config", baseline_script)

    def test_eval_script_supports_explicit_output_dir_without_forcing_shared_default(self) -> None:
        eval_script = ROOT / "flowgrpo_general_2x40g" / "run_eval_learnable_general_2x40g.sh"
        text = eval_script.read_text(encoding="utf-8")
        self.assertIn("OUTPUT_DIR", text)
        self.assertIn("--output-dir", text)

    def test_ticket_experiment_has_no_sft_configuration(self) -> None:
        for path in ROOT.rglob("*"):
            if path.is_file() and "tests" not in path.parts and path.suffix in {".yaml", ".sh", ".py"}:
                self.assertNotRegex(path.read_text(encoding="utf-8"), r"(?i)\bsft\b")


if __name__ == "__main__":
    unittest.main()
