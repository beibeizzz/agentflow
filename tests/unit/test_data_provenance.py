from __future__ import annotations

import hashlib
from collections import Counter
from pathlib import Path

from agentflow_rl.tasks.ticket.dataset import load_ticket_rows


ROOT = Path(__file__).parents[2]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_formal_ticket_splits_are_valid_unique_and_exactly_balanced() -> None:
    for split in ("smoke", "train", "validation", "test"):
        rows = load_ticket_rows(ROOT / "data" / "ticket" / f"{split}.jsonl")
        modes = Counter(row.curriculum_mode for row in rows)
        assert rows
        assert modes["direct"] == modes["indirect"]
        assert len({row.episode_id for row in rows}) == len(rows)


def test_old_ticket_data_is_preserved_only_as_hash_identical_parity_data() -> None:
    expected = {
        "smoke.jsonl": "73a9708d80bcc6bb",
        "train.jsonl": "b2a4e7daf9d176bd",
        "validation.jsonl": "4d23b86c260c19e2",
        "test.jsonl": "dbc583f80ce2fca8",
    }
    for name, prefix in expected.items():
        path = ROOT / "data" / "ticket" / "parity_80_20" / name
        assert sha256(path).startswith(prefix)


def test_gsm8k_copies_load_and_readme_records_exact_sources() -> None:
    from agentflow_rl.tasks.gsm8k.dataset import load_gsm8k_rows

    train = load_gsm8k_rows(ROOT / "data" / "gsm8k" / "train.json")
    test = load_gsm8k_rows(ROOT / "data" / "gsm8k" / "test.json")
    smoke = load_gsm8k_rows(ROOT / "data" / "gsm8k" / "smoke.json")
    assert (len(train), len(test), len(smoke)) == (1327, 319, 20)
    readme = (ROOT / "data" / "README.md").read_text(encoding="utf-8")
    for text in (
        "gsm8k_train_calculator_structured.json",
        "gsm8k_train_learnable.json",
        "gsm8k_test_eval_rest.json",
        "parity_80_20",
        "SHA-256",
        "MIT",
    ):
        assert text in readme
