from __future__ import annotations

import ast
import operator
import re
from fractions import Fraction


_VALID = re.compile(r"^[0-9+\-*/().% ]+$")
_BINARY = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv, ast.Mod: operator.mod}
_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def safe_eval_calculation(expression: str) -> Fraction:
    value = expression.strip()
    if not value or not _VALID.fullmatch(value):
        raise ValueError("invalid calculation expression")
    return _evaluate(ast.parse(value, mode="eval").body)


def _evaluate(node: ast.AST) -> Fraction:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return Fraction(str(node.value))
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
        left, right = _evaluate(node.left), _evaluate(node.right)
        if isinstance(node.op, (ast.Div, ast.Mod)) and right == 0:
            raise ZeroDivisionError("division by zero")
        return _BINARY[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
        return _UNARY[type(node.op)](_evaluate(node.operand))
    raise ValueError("unsupported calculation node")


def format_number(value: Fraction) -> str:
    if value.denominator == 1:
        return str(value.numerator)
    return f"{value.numerator / value.denominator:.10f}".rstrip("0").rstrip(".")
