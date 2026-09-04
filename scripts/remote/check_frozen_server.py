from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request


def main() -> int:
    parser = argparse.ArgumentParser(description="Wait for the frozen vLLM OpenAI server")
    parser.add_argument("--url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", default="Qwen3-8B")
    parser.add_argument("--timeout", type=float, default=300.0)
    args = parser.parse_args()
    deadline = time.monotonic() + args.timeout
    last_error = ""
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"{args.url.rstrip('/')}/models", timeout=5) as response:
                payload = json.load(response)
            models = {str(item.get("id")) for item in payload.get("data", ())}
            if args.model in models:
                print(json.dumps({"ready": True, "model": args.model, "url": args.url}))
                return 0
            last_error = f"served models: {sorted(models)}"
        except (OSError, urllib.error.URLError, ValueError) as exc:
            last_error = str(exc)
        time.sleep(2)
    raise TimeoutError(f"frozen server readiness timeout: {last_error}")


if __name__ == "__main__":
    raise SystemExit(main())
