from __future__ import annotations

import gurobipy as gp
import numpy as np

from .linear_expression import LinearInequality
from .stl_base import StlFormula
from .trace import Trace


class Predicate:
    """Linear inequality predicate."""

    def __init__(
        self,
        inequality: LinearInequality,
        qualitative: bool = False,
    ) -> None:
        self.name = inequality.name
        self.inequality = inequality
        self.qualitative = qualitative

    def __hash__(self) -> int:
        return hash(self.name)

    def __repr__(self) -> str:
        return self.name

    def eval_depth(self, state: dict[str, np.ndarray]) -> np.ndarray:
        """Evaluate the (signed) depth at the given state."""
        value = self.inequality.expr.eval(state) * self.inequality.sign
        return value / self.inequality.expr.norm

    def expr_value(self, state: dict[str, gp.MVar]) -> gp.MLinExpr:
        """Return the value of the expression at the given state."""
        return self.inequality.expr.apply_to_variables(state) * self.inequality.sign


class Atomic(StlFormula):
    def __init__(self, symbol: LinearInequality, qualitative: bool = False):
        super().__init__(symbol.name)
        self.predicate = Predicate(symbol, qualitative=qualitative)

    def negation(self) -> StlFormula:
        if not self.predicate.qualitative:
            return Atomic(self.predicate.inequality.invert())
        else:
            # negation of x >= 0 is x <= -1
            expr = self.predicate.inequality.to_gt_zero().expr
            return Atomic(expr <= -1, qualitative=True)

    def robust_semantics(self, trace: Trace, memo=None) -> np.ndarray:
        """Return the robustness value of the formula at each step of the trace."""
        if memo is not None and self.name in memo:
            return memo[self.name]
        return self.predicate.eval_depth(trace.state)
