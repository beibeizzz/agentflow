from __future__ import annotations

import ast
import operator
import re
import json
from decimal import Decimal, getcontext
from fractions import Fraction
from typing import Callable

from agentflow.tools.base import BaseTool


TOOL_NAME = "Calculator_Tool"


class Calculator_Tool(BaseTool):
    require_llm_engine = False

    _BIN_OPS: dict[type[ast.operator], Callable[[Fraction, Fraction], Fraction]] = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
    }
    _UNARY_OPS: dict[type[ast.unaryop], Callable[[Fraction], Fraction]] = {
        ast.UAdd: operator.pos,
        ast.USub: operator.neg,
    }

    def __init__(self, model_string: str | None = None):
        super().__init__(
            tool_name=TOOL_NAME,
            tool_description=(
                "A deterministic calculator supporting multi-step operations, percentages, and elementary arithmetic. "
            ),
            input_types={
                "expression": (
                    "str - Arithmetic expression using +, -, *, /, parentheses, decimals. "
                )
            },
            output_type="str - The numeric result of the evaluated expression.",
            demo_commands=[
            ],
            user_metadata={
                "limitations": (
                    'Only arithmetic expressions are allowed. Variables, functions, text, units, and "=" signs are not allowed.'
                ),
            },
            model_string=model_string,
        )

    def execute(self, expression: str | None = None, query: str | None = None) -> str:
        raw_expression = expression if expression is not None else query
        if raw_expression is None or not str(raw_expression).strip():
            return "Error: missing arithmetic expression."

        display_expression = str(raw_expression).strip()
        try:
            normalized = self._normalize_expression(display_expression)
            parsed = ast.parse(normalized, mode="eval")
            value = self._eval_node(parsed.body)
            return self._format_fraction(value)
        except SyntaxError:
            return f"invalid syntax: expression={json.dumps(display_expression)}"
        except Exception as exc:
            return f"Error: {exc}"

    def _normalize_expression(self, expression: str) -> str:
        normalized = expression.strip()
        normalized = normalized.replace("×", "*").replace("÷", "/")
        normalized = normalized.replace("$", "").replace(",", "")
        normalized = re.sub(r"(?<=\d)\s*[xX]\s*(?=\d)", "*", normalized)
        normalized = re.sub(r"^\s*\*\s*(?=\d|\()", "", normalized)
        if not re.fullmatch(r"[\d\s+\-*/().%]+", normalized):
            raise ValueError("expression contains unsupported characters.")
        normalized = re.sub(
            r"(?<![\w.])(\d+(?:\.\d+)?)\s*%",
            r"(\1/100)",
            normalized,
        )
        if "%" in normalized:
            raise ValueError("percent sign must follow a number.")
        return normalized

    def _eval_node(self, node: ast.AST) -> Fraction:
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return Fraction(str(node.value))
        if isinstance(node, ast.BinOp):
            op_type = type(node.op)
            if op_type not in self._BIN_OPS:
                raise ValueError(f"unsupported operator {op_type.__name__}.")
            left = self._eval_node(node.left)
            right = self._eval_node(node.right)
            if isinstance(node.op, ast.Div) and right == 0:
                raise ZeroDivisionError("division by zero.")
            return self._BIN_OPS[op_type](left, right)
        if isinstance(node, ast.UnaryOp):
            op_type = type(node.op)
            if op_type not in self._UNARY_OPS:
                raise ValueError(f"unsupported unary operator {op_type.__name__}.")
            return self._UNARY_OPS[op_type](self._eval_node(node.operand))
        raise ValueError(f"unsupported syntax {type(node).__name__}.")

    def _format_fraction(self, value: Fraction) -> str:
        if value.denominator == 1:
            return str(value.numerator)

        getcontext().prec = 28
        decimal_value = Decimal(value.numerator) / Decimal(value.denominator)
        formatted = format(decimal_value, "f").rstrip("0").rstrip(".")
        return formatted or "0"
