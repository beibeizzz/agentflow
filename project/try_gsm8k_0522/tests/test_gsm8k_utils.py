import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gsm8k_utils import (
    answers_match,
    extract_gold_answer,
    extract_predicted_answer,
    normalize_numeric_answer,
)


class TestGSM8KUtils(unittest.TestCase):
    def test_extract_gold_answer_after_hash_marker(self):
        answer = (
            "Natalia sold 48/2 = <<48/2=24>>24 clips in May.\n"
            "Natalia sold 48+24 = <<48+24=72>>72 clips altogether.\n"
            "#### 72"
        )

        self.assertEqual(extract_gold_answer(answer), "72")

    def test_extract_gold_answer_removes_commas_and_spaces(self):
        self.assertEqual(extract_gold_answer("Reasoning here.\n#### 1,250 "), "1250")

    def test_extract_predicted_answer_prefers_boxed_value(self):
        response = "After solving, Therefore, the final answer is: $\\boxed{3.5}$."

        self.assertEqual(extract_predicted_answer(response), "3.5")

    def test_extract_predicted_answer_prefers_final_hash_marker_over_earlier_boxed_value(self):
        response = "Wrong intermediate boxed value: \\boxed{12}\nFinal answer:\n#### 1509"

        self.assertEqual(extract_predicted_answer(response), "1509")

    def test_extract_predicted_answer_prefers_generator_line(self):
        response = "Executor:\n1. <Calculator> 96 / 16 = 6 <Calculator>\nGenerator:\n6"

        self.assertEqual(extract_predicted_answer(response), "6")

    def test_extract_predicted_answer_prefers_answer_xml_tag(self):
        response = "Compute 16 - 7 = 9, then 9 * 2 = 18.\n<answer>18</answer>\nDebug id: 999"

        self.assertEqual(extract_predicted_answer(response), "18")

    def test_extract_predicted_answer_uses_last_number_without_boxed_value(self):
        response = "We compute 8 + 7 = 15. So the answer is 15 apples."

        self.assertEqual(extract_predicted_answer(response), "15")

    def test_extract_predicted_answer_handles_negative_and_comma_numbers(self):
        response = "The intermediate value is 20. Final answer: -1,234.50"

        self.assertEqual(extract_predicted_answer(response), "-1234.50")

    def test_normalize_numeric_answer_handles_integer_decimal_and_fraction(self):
        self.assertEqual(normalize_numeric_answer("1,000.0"), normalize_numeric_answer("1000"))
        self.assertEqual(normalize_numeric_answer("7/2"), normalize_numeric_answer("3.5"))

    def test_answers_match_accepts_equivalent_numeric_forms(self):
        self.assertTrue(answers_match("1,000", "1000.0"))
        self.assertTrue(answers_match("7/2", "3.5"))
        self.assertFalse(answers_match("71", "72"))

    def test_extract_predicted_answer_returns_none_without_number(self):
        self.assertIsNone(extract_predicted_answer("I cannot determine it."))


if __name__ == "__main__":
    unittest.main()
