from __future__ import annotations

import json
import subprocess
from importlib import metadata
from pathlib import Path


EXPECTED_COMMIT = "7aed6b230776f963fa09509c10d9c3a767d1102c"


def git_commit_from_source(package_file: Path) -> str | None:
    for root in package_file.resolve().parents:
        if (root / ".git").exists():
            return subprocess.check_output(
                ["git", "-C", str(root), "rev-parse", "HEAD"],
                text=True,
            ).strip()
    return None


def commit_from_direct_url() -> str | None:
    payload = metadata.distribution("verl").read_text("direct_url.json")
    if not payload:
        return None
    return json.loads(payload).get("vcs_info", {}).get("commit_id")


def main() -> int:
    import verl

    actual = git_commit_from_source(Path(verl.__file__)) or commit_from_direct_url()
    if actual != EXPECTED_COMMIT:
        raise RuntimeError(
            f"veRL commit mismatch: installed={actual!r}, expected={EXPECTED_COMMIT}"
        )
    print(f"verl_commit={actual}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
