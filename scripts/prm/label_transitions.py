from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path

from openai import AsyncOpenAI

from agentflow_rl.prm import label_transitions, write_process_labels
from agentflow_rl.integrations.artifacts import load_process_transitions
from agentflow_rl.rewards.online_judge import JsonlScoreCache, OpenAICompatibleProcessJudge
from agentflow_rl.rewards.rubric import PROCESS_RUBRIC_REVISION
from agentflow_rl.rewards.transition_view import PROCESS_VIEW_REVISION
import json


async def run(args) -> None:
    from transformers import AutoTokenizer

    from agentflow_rl.rewards.transition_view import render_process_transition

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    client = AsyncOpenAI(
        base_url=args.base_url,
        api_key=os.environ[args.api_key_env],
        timeout=args.timeout,
    )
    judge = OpenAICompatibleProcessJudge(
        client,
        model=args.model,
        revision=args.revision,
        cache=JsonlScoreCache(args.cache),
        max_concurrency=args.concurrency,
        max_attempts=args.attempts,
        backoff_s=args.backoff,
        transition_renderer=lambda transition: render_process_transition(
            transition,
            tokenizer=tokenizer,
            max_length=args.max_length,
        ),
    )
    labels = await label_transitions(
        load_process_transitions(args.input), judge, max_concurrency=args.concurrency
    )
    write_process_labels(labels, args.output)
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    args.metrics.write_text(
        json.dumps(
            {
                **judge.metrics(),
                "rubric_revision": PROCESS_RUBRIC_REVISION,
                "process_view_revision": PROCESS_VIEW_REVISION,
                "tokenizer": args.tokenizer,
                "max_length": args.max_length,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Label AgentFlow transitions with an offline Judge")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", default="deepseek-v4-pro")
    parser.add_argument("--revision", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--api-key-env", default="DEEPSEEK_API_KEY")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--backoff", type=float, default=0.5)
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args()
    asyncio.run(run(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
