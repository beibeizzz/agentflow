from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rewrite_calculator_prompts.deepseek_client import DeepSeekClient
from rewrite_calculator_prompts.pipeline import DatasetRewriter, PipelinePaths


SCRIPT_DIR = Path(__file__).resolve().parent
WORK_DIR = SCRIPT_DIR.parent
DEFAULT_INPUT = WORK_DIR / "data" / "gsm8k_train_learnable.json"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "outputs"


def load_config(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required when --config is used") from exc
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    for key in ("input", "output_dir"):
        value = config.get(key)
        if value is not None:
            resolved = Path(value)
            if not resolved.is_absolute():
                resolved = path.parent / resolved
            config[key] = resolved
    return config


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=Path)
    pre_args, _ = pre_parser.parse_known_args(argv)
    config = load_config(pre_args.config)

    parser = argparse.ArgumentParser(
        description="Rewrite GSM8K prompts for calculator-only AgentFlow training with DeepSeek."
    )
    parser.add_argument("--config", type=Path, default=pre_args.config)
    parser.add_argument("--input", type=Path, default=Path(config.get("input", DEFAULT_INPUT)))
    parser.add_argument("--output-dir", type=Path, default=Path(config.get("output_dir", DEFAULT_OUTPUT_DIR)))
    parser.add_argument("--start", type=int, default=int(config.get("start", 0)))
    parser.add_argument("--limit", type=int, default=config.get("limit"))
    parser.add_argument("--max-attempts", type=int, default=int(config.get("max_attempts", 3)))
    parser.add_argument(
        "--transport-attempts",
        type=int,
        default=int(config.get("transport_attempts", 4)),
    )
    parser.add_argument("--concurrency", type=int, default=int(config.get("concurrency", 2)))
    parser.add_argument(
        "--rewrite-temperature",
        type=float,
        default=float(config.get("rewrite_temperature", 0.3)),
    )
    parser.add_argument(
        "--rewrite-model",
        default=str(config.get("rewrite_model", "deepseek-v4-flash")),
    )
    parser.add_argument(
        "--judge-model",
        default=str(config.get("judge_model", "deepseek-v4-pro")),
    )
    parser.add_argument(
        "--base-url",
        default=str(config.get("base_url", "https://api.deepseek.com")),
    )
    parser.set_defaults(resume=bool(config.get("resume", True)))
    parser.add_argument("--resume", dest="resume", action="store_true")
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    args = parser.parse_args(argv)
    if args.limit is not None:
        args.limit = int(args.limit)
    return args


def build_client(args: argparse.Namespace) -> DeepSeekClient:
    return DeepSeekClient(
        base_url=args.base_url,
        rewrite_model=args.rewrite_model,
        judge_model=args.judge_model,
        max_transport_attempts=args.transport_attempts,
    )


def load_records(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list) or not all(isinstance(item, dict) for item in payload):
        raise ValueError(f"Expected a JSON list of objects in {path}")
    return payload


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.start < 0:
        raise SystemExit("--start must be non-negative")
    if args.limit is not None and args.limit < 0:
        raise SystemExit("--limit must be non-negative")
    if args.max_attempts < 1:
        raise SystemExit("--max-attempts must be at least 1")
    if args.concurrency < 1:
        raise SystemExit("--concurrency must be at least 1")
    if not args.input.exists():
        raise SystemExit(f"Input dataset does not exist: {args.input}")

    records = load_records(args.input)
    client = build_client(args)
    rewriter = DatasetRewriter(
        client,
        max_attempts=args.max_attempts,
        rewrite_temperature=args.rewrite_temperature,
    )
    summary = rewriter.run(
        records,
        paths=PipelinePaths.in_directory(args.output_dir),
        start=args.start,
        limit=args.limit,
        resume=args.resume,
        concurrency=args.concurrency,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
