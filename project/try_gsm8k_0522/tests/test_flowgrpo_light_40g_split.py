from __future__ import annotations

import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from try_gsm8k_0522.flowgrpo_light_40g.prepare_split import split_rows


class FlowGrpoLight40GSplitTests(unittest.TestCase):
    def test_split_rows_uses_first_n_for_train_and_rest_for_eval(self) -> None:
        rows = [{"pid": idx} for idx in range(5)]

        train_rows, eval_rows = split_rows(rows, train_size=3)

        self.assertEqual([row["pid"] for row in train_rows], [0, 1, 2])
        self.assertEqual([row["pid"] for row in eval_rows], [3, 4])

    def test_split_rows_rejects_invalid_train_size(self) -> None:
        rows = [{"pid": idx} for idx in range(2)]

        with self.assertRaises(ValueError):
            split_rows(rows, train_size=2)


if __name__ == "__main__":
    unittest.main()
