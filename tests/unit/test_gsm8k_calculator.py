from __future__ import annotations

from fractions import Fraction

import pytest


@pytest.mark.parametrize(
    ("expression", "expected"),
    [("2+3", Fraction(5)), ("0.5 * 4", Fraction(2)), ("(9-3)/4", Fraction(3, 2)), ("7%4", Fraction(3))],
)
def test_safe_calculator_exact_arithmetic(expression: str, expected: Fraction) -> None:
    from agentflow_rl.tasks.gsm8k.calculator import safe_eval_calculation

    assert safe_eval_calculation(expression) == expected


@pytest.mark.parametrize("expression", ["x+1", "f(1)", "2**8", "a[0]", "1=1", "1/0"])
def test_safe_calculator_rejects_unsafe_or_invalid_expression(expression: str) -> None:
    from agentflow_rl.tasks.gsm8k.calculator import safe_eval_calculation

    with pytest.raises((ValueError, ZeroDivisionError)):
        safe_eval_calculation(expression)
