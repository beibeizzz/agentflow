from __future__ import annotations

import ast
import operator
import re
from fractions import Fraction
from typing import Any


CALCULATION_RE = re.compile(r"^[0-9+\-*/().% ]+$")

_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Mod: operator.mod,
}
_UNARY_OPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def safe_eval_calculation(expression: str) -> Fraction:
    expression = expression.strip()
    if not expression or not CALCULATION_RE.fullmatch(expression):
        raise ValueError(f"Invalid calculation expression: {expression!r}")
    parsed = ast.parse(expression, mode="eval")
    return _eval_node(parsed.body)


def _eval_node(node: ast.AST) -> Fraction:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return Fraction(str(node.value))
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        if isinstance(node.op, (ast.Div, ast.Mod)) and right == 0:
            raise ZeroDivisionError("division by zero")
        return _BIN_OPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_eval_node(node.operand))
    raise ValueError(f"Unsupported calculation node: {ast.dump(node)}")


def format_number(value: Fraction) -> str:
    if value.denominator == 1:
        return str(value.numerator)
    decimal = value.numerator / value.denominator
    text = f"{decimal:.10f}".rstrip("0").rstrip(".")
    return text or "0"


def memory_to_text(memory: dict[str, Any]) -> str:
    return str(memory)
