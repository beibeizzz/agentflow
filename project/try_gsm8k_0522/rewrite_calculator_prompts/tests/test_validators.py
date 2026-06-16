import sys
import unittest
from pathlib import Path


PACKAGE_PARENT = Path(__file__).resolve().parents[2]
if str(PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PARENT))

from rewrite_calculator_prompts.validators import validate_rewrite


class RewriteValidatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.source = {
            "question": (
                "A store sells 3 pens for $2. Mia buys twice as many pens as "
                "Noah, who buys 6 pens. How much does Mia spend?"
            ),
            "answer": (
                "Mia buys 6*2=<<6*2=12>>12 pens. "
                "She buys 12/3=<<12/3=4>>4 groups and spends 4*2=<<4*2=8>>8. #### 8"
            ),
            "gold_answer": "8",
        }

    def test_accepts_natural_language_math_relationships(self) -> None:
        rewritten = """Known facts:
- A store sells every group of 3 pens for $2.
- Noah buys 6 pens.
- Mia buys twice the number of pens that Noah buys.

Question:
- How many dollars does Mia spend?"""

        result = validate_rewrite(self.source, rewritten)

        self.assertTrue(result.ok, result.messages)
        self.assertEqual(len(result.facts), 3)
        self.assertEqual(result.question, "How many dollars does Mia spend?")

    def test_rejects_extra_sections(self) -> None:
        rewritten = """Known facts:
- A store sells every group of 3 pens for $2.
- Noah buys 6 pens.
- Mia buys twice the number of pens that Noah buys.

Question:
- How many dollars does Mia spend?

Instructions:
- Solve step by step."""

        result = validate_rewrite(self.source, rewritten)

        self.assertFalse(result.ok)
        self.assertIn("invalid_format", result.codes)

    def test_rejects_explicit_arithmetic_expression(self) -> None:
        rewritten = """Known facts:
- Noah buys 6 pens.
- Mia buys 6 * 2 pens.

Question:
- How many dollars does Mia spend?"""

        result = validate_rewrite(self.source, rewritten)

        self.assertFalse(result.ok)
        self.assertIn("arithmetic_leak", result.codes)

    def test_rejects_intermediate_result_not_present_in_source_question(self) -> None:
        rewritten = """Known facts:
- A store sells every group of 3 pens for $2.
- Mia buys 12 pens.

Question:
- How many dollars does Mia spend?"""

        result = validate_rewrite(self.source, rewritten)

        self.assertFalse(result.ok)
        self.assertIn("new_number", result.codes)
        self.assertIn("solution_value_leak", result.codes)

    def test_rejects_missing_source_number(self) -> None:
        rewritten = """Known facts:
- A store sells pens for $2 per group.
- Mia buys twice the number of pens that Noah buys.

Question:
- How many dollars does Mia spend?"""

        result = validate_rewrite(self.source, rewritten)

        self.assertFalse(result.ok)
        self.assertIn("missing_number", result.codes)

    def test_rejects_answer_markers_and_equations(self) -> None:
        rewritten = """Known facts:
- A store sells every group of 3 pens for $2.
- Noah buys 6 pens.
- Mia buys twice the number of pens that Noah buys.
- The final answer = 8.

Question:
- How many dollars does Mia spend?"""

        result = validate_rewrite(self.source, rewritten)

        self.assertFalse(result.ok)
        self.assertIn("answer_leak", result.codes)
        self.assertIn("arithmetic_leak", result.codes)


if __name__ == "__main__":
    unittest.main()
