from __future__ import annotations

import gurobipy as gp
import numpy as np

from .constants import EPSILON, M


def if_all_then(
    conditions: list[gp.LinExpr | gp.Var], then: gp.LinExpr | gp.Var
) -> gp.TempLConstr:
    """Return a constraint that asserts if all conditions are true, then the expression is true."""
    return at_least_one(*(1 - cond for cond in conditions), then)


def at_least_one(*args: gp.LinExpr | gp.Var) -> gp.TempLConstr:
    """Return a constraint that asserts at least one of the arguments is 1."""
    if len(args) <= 1:
        raise ValueError("At least two arguments required")
    else:
        return gp.quicksum(args) >= 1  # type: ignore


def add_disjoint_enforcer(milp: gp.Model, z: gp.Var, a: tuple, b: tuple) -> None:
    """Add constraint enforcing that the intervals a and b are disjoint.
    Intervals are considered closed.
    """
    z1 = milp.addVar(vtype=gp.GRB.BINARY)
    z2 = milp.addVar(vtype=gp.GRB.BINARY)
    milp.addConstr(z1 + z2 >= 1)
    milp.addConstr(
        a[0] - b[1] >= EPSILON - M * (2 - z - z1),
    )
    milp.addConstr(
        b[0] - a[1] >= EPSILON - M * (2 - z - z2),
    )


def multiply_with_binary_variable(
    milp: gp.Model, x: gp.LinExpr | gp.Var, z: gp.Var
) -> gp.Var:
    """
    Return a variable that represents the product of continuous variable and binary variable.
    """
    y = milp.addVar(vtype=gp.GRB.CONTINUOUS, lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY)
    milp.addConstr((z == 0) >> (y == 0))
    milp.addConstr((z == 1) >> (y == x))
    return y


def add_and_aux_var(milp: gp.Model, z1: gp.Var, z2: gp.Var, name: str) -> gp.Var:
    """Add a binary variable that represents the conjunction of two binary variables."""
    z = milp.addVar(vtype=gp.GRB.BINARY, name=name)
    milp.addConstr(z == gp.and_(z1, z2), name=f'and_{name}')
    return z


def add_or_aux_var(milp: gp.Model, z1: gp.Var, z2: gp.Var, name: str) -> gp.Var:
    """Add a binary variable that represents the disjunction of two binary variables."""
    z = milp.addVar(vtype=gp.GRB.BINARY, name=name)
    milp.addConstr(z == gp.or_(z1, z2), name=f'or_{name}')
    return z


class BinaryExpressionValue:
    def __init__(self, milp: gp.Model, name: str, size: int, unit: float = 1.0) -> None:
        self.binary = milp.addMVar(size, vtype=gp.GRB.BINARY, name=name)
        self.powers: np.ndarray = unit * np.geomspace(1, 2 ** (size - 1), num=size)

    @property
    def value(self) -> gp.LinExpr:
        return (self.binary @ self.powers).item()

    @property
    def unit(self) -> float:
        return self.powers[0].item()

    def multiply(self, milp: gp.Model, expr: gp.LinExpr | gp.Var) -> gp.LinExpr:
        return gp.LinExpr(
            self.powers.tolist(),
            [
                multiply_with_binary_variable(milp, expr, z)
                for z in self.binary.tolist()
            ],
        )


def ensure_monotonicity_by_derivative(
    milp: gp.Model, derivative: list[gp.LinExpr | gp.Var], N: int
):
    # (To ensure monotonicity in interpolation)
    # All adjacent dx must have same sign (non-negative or non-positive)
    # z = 1 then non-negative, z = 0 then non-positive
    z = milp.addVars(N, vtype=gp.GRB.BINARY)
    for i in range(N):
        milp.addConstr(
            (z[i] == 1) >> (derivative[i] >= 0),
        )
        milp.addConstr(
            (z[i] == 1) >> (derivative[i + 1] >= 0),
        )
        milp.addConstr(
            (z[i] == 0) >> (derivative[i] <= 0),
        )
        milp.addConstr(
            (z[i] == 0) >> (derivative[i + 1] <= 0),
        )
