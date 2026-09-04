from __future__ import annotations

import argparse
from pathlib import Path

from agentflow_rl.synthesis.pipeline import TicketSynthesisPipeline, generate_blueprints


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate isolated synthetic Ticket episodes")
    parser.add_argument("--split", required=True)
    parser.add_argument("--count", required=True, type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--progress", required=True, type=Path)
    parser.add_argument(
        "--rewrite-model",
        help="Optionally rewrite requests through an OpenAI-compatible API model",
    )
    args = parser.parse_args()
    rewriter = None
    if args.rewrite_model:
        from openai import OpenAI

        from agentflow_rl.synthesis.client import OpenAIRequestRewriter

        rewriter = OpenAIRequestRewriter.from_environment(OpenAI, model=args.rewrite_model)
    blueprints = generate_blueprints(split=args.split, count=args.count, seed=args.seed)
    episodes = TicketSynthesisPipeline(rewriter=rewriter).run(
        blueprints, output_path=args.output, progress_path=args.progress
    )
    print(f"generated={len(episodes)} output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
