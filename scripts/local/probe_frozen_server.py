from __future__ import annotations

import argparse

from openai import OpenAI


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe the local frozen vLLM model")
    parser.add_argument("--url", default="http://127.0.0.1:8001/v1")
    parser.add_argument("--model", default="Qwen3-0.6B")
    args = parser.parse_args()
    client = OpenAI(base_url=args.url, api_key="not-required")
    response = client.chat.completions.create(
        model=args.model,
        messages=[{"role": "user", "content": "Return only the number: 2+3"}],
        max_tokens=32,
        temperature=0.0,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    content = response.choices[0].message.content
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("frozen model returned an empty response")
    print(content.strip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
