import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


PACKAGE_PARENT = Path(__file__).resolve().parents[2]
if str(PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PARENT))

from rewrite_calculator_prompts.rewrite_dataset import (
    build_client,
    load_config,
    parse_args,
)


class CliTests(unittest.TestCase):
    def test_defaults_target_learnable_dataset_and_three_attempts(self) -> None:
        args = parse_args([])

        self.assertEqual(args.input.name, "gsm8k_train_learnable.json")
        self.assertEqual(args.max_attempts, 3)
        self.assertEqual(args.rewrite_model, "deepseek-v4-flash")
        self.assertEqual(args.judge_model, "deepseek-v4-pro")
        self.assertEqual(args.concurrency, 2)
        self.assertTrue(args.resume)

    def test_cli_overrides_range_and_resume_settings(self) -> None:
        args = parse_args(
            [
                "--start",
                "10",
                "--limit",
                "25",
                "--max-attempts",
                "2",
                "--concurrency",
                "4",
                "--no-resume",
            ]
        )

        self.assertEqual(args.start, 10)
        self.assertEqual(args.limit, 25)
        self.assertEqual(args.max_attempts, 2)
        self.assertEqual(args.concurrency, 4)
        self.assertFalse(args.resume)

    def test_yaml_config_is_loaded_as_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "config.yaml"
            config_path.write_text(
                "rewrite_model: custom-flash\nmax_attempts: 2\nconcurrency: 1\n",
                encoding="utf-8",
            )

            config = load_config(config_path)

        self.assertEqual(config["rewrite_model"], "custom-flash")
        self.assertEqual(config["max_attempts"], 2)

    def test_config_relative_paths_resolve_from_config_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_dir = Path(tmp)
            config_path = config_dir / "config.yaml"
            config_path.write_text(
                "input: data/source.json\noutput_dir: generated\n",
                encoding="utf-8",
            )

            args = parse_args(["--config", str(config_path)])

        self.assertEqual(args.input, config_dir / "data" / "source.json")
        self.assertEqual(args.output_dir, config_dir / "generated")

    def test_build_client_requires_environment_api_key(self) -> None:
        args = parse_args([])
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "DEEPSEEK_API_KEY"):
                build_client(args)


if __name__ == "__main__":
    unittest.main()
