from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


DEFAULT_MODEL = "Qwen3-0.6B-Instruct"
DEFAULT_BASE_URL = "http://localhost:8000/v1"

SYSTEM_PROMPT = (
    "You are a careful grade-school math problem solver. "
    "Use only the information in the problem. "
    "Keep the reasoning concise and arithmetic-focused."
)


def build_prompt(question: str) -> str:
    return f"""Solve the following GSM8K math word problem.

Instructions:
- Reason step by step before the final answer.
- Use only facts stated in the problem.
- Do not introduce extra days, weeks, people, prices, or assumptions.
- The final answer must be a single numeric value, integer, decimal, or fraction.
- Do not include units, currency symbols, commas, or explanatory text inside the answer tag.
- End with exactly one final line in this format:
<answer>NUMBER</answer>
- No text after </answer>.

Problem:
{question}"""


def load_data(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise SystemExit(f"Expected a list in {path}")
    return data


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp_path, path)


def request_json(url: str, payload: dict[str, Any] | None = None, timeout: int = 120) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer dummy-token",
        },
        method="GET" if payload is None else "POST",
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def check_vllm_server(base_url: str, model: str, timeout: int) -> None:
    url = base_url.rstrip("/") + "/models"
    try:
        payload = request_json(url, timeout=timeout)
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Could not reach vLLM models endpoint at {url}: {exc}") from exc

    ids = {item.get("id") for item in payload.get("data", [])}
    if model not in ids:
        raise SystemExit(f"vLLM is reachable, but model {model!r} is not in /models. Found: {sorted(ids)}")
    print(f"Verified vLLM model {model!r} at {url}")


def generate_completion(
    base_url: str,
    model: str,
    question: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    timeout: int,
) -> str:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_prompt(question)},
        ],
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
    }
    response = request_json(base_url.rstrip("/") + "/chat/completions", payload=payload, timeout=timeout)
    try:
        return str(response["choices"][0]["message"]["content"])
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"Unexpected chat completion response: {response}") from exc


def run_examples(args: argparse.Namespace) -> None:
    data = load_data(args.data_file)
    if args.start:
        data = data[args.start :]
    if args.limit is not None:
        data = data[: args.limit]

    check_vllm_server(args.base_url, args.model, args.request_timeout)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    completed = 0
    skipped = 0
    failed = 0
    started_at = time.time()

    for position, row in enumerate(data, start=1):
        pid = row["pid"]
        output_path = args.output_dir / f"output_{pid}.json"
        if output_path.exists() and not args.overwrite:
            skipped += 1
            print(f"[{position}/{len(data)}] skip pid={pid}: {output_path} exists")
            continue

        print(f"[{position}/{len(data)}] run pid={pid}")
        item_started = time.time()
        prompt = build_prompt(row["question"])
        payload: dict[str, Any] = {
            "pid": pid,
            "question": row["question"],
            "query": prompt,
            "answer": row["answer"],
            "gold_answer": row["gold_answer"],
            "model": args.model,
            "base_url": args.base_url,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
        }
        try:
            direct_output = generate_completion(
                base_url=args.base_url,
                model=args.model,
                question=row["question"],
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                timeout=args.request_timeout,
            )
            payload["direct_output"] = direct_output
            payload["ok"] = True
            completed += 1
        except Exception as exc:
            payload["ok"] = False
            payload["error_type"] = type(exc).__name__
            payload["error"] = str(exc)
            failed += 1
            print(f"[{position}/{len(data)}] error pid={pid}: {type(exc).__name__}: {exc}")
            if args.stop_on_error:
                atomic_write_json(output_path, payload)
                raise
        finally:
            payload["wall_time"] = round(time.time() - item_started, 3)
            atomic_write_json(output_path, payload)

    elapsed = round(time.time() - started_at, 3)
    print(
        f"Finished direct baseline: completed={completed}, skipped={skipped}, failed={failed}, "
        f"elapsed={elapsed}s, output_dir={args.output_dir}"
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run direct local vLLM GSM8K baseline without AgentFlow.")
    parser.add_argument("--data-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--request-timeout", type=int, default=120)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    return parser.parse_args(argv)


def main() -> None:
    run_examples(parse_args())


if __name__ == "__main__":
    main()
