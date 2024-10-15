from __future__ import annotations

import dataclasses
from calendar import c
from typing import overload

import gurobipy as gp
import numpy as np

LE = gp.GRB.LESS_EQUAL
GE = gp.GRB.GREATER_EQUAL


@dataclasses.dataclass
class LinearExpression:
    variables: list[str]
    coefficients: list[float]
    constant: float = 0.0

    def __ge__(self, other: float) -> LinearInequality:
        x = dataclasses.replace(self, constant=self.constant - other)
        return LinearInequality(x, GE, name=f"{str(self).replace(' ', '')}_>=_{other}")

    def __le__(self, other: float) -> LinearInequality:
        x = dataclasses.replace(self, constant=self.constant - other)
        return LinearInequality(x, LE, name=f"{str(self).replace(' ', '')}_<=_{other}")

    def __repr__(self) -> str:
        combined = ' + '.join(
            f"{coeff} * {var}" for var, coeff in zip(self.variables, self.coefficients)
        )
        return f"{combined} + {self.constant}"

    @classmethod
    def f(cls, *args: str | float) -> LinearExpression:
        """Create a linear expression from variables and coefficients. No constant term is allowed.

        Example:
        >>> LinearExpression.f(1.0, 'x', 2.0, 'y')
        1.0 * x + 2.0 * y + 0.0
        """
        assert len(args) % 2 == 0
        coefficients = args[::2]
        variables = args[1::2]
        return cls(list(variables), list(coefficients))  # type: ignore

    @classmethod
    def unit(cls, var: str) -> LinearExpression:
        """Create a linear expression from a single variable with coefficient 1.0.

        Example:
        >>> LinearExpression.unit('x')
        1.0 * x + 0.0
        """
        return cls([var], [1.0])

    @overload
    def eval(self, state: dict[str, np.ndarray]) -> np.ndarray: ...

    @overload
    def eval(self, state: dict[str, float]) -> float: ...

    def eval(self, state: dict) -> np.ndarray | float:
        target_states = [state[var] for var in self.variables]
        if isinstance(target_states[0], np.ndarray):
            value = (
                np.vstack(target_states).T @ np.array(self.coefficients) + self.constant
            )
        elif isinstance(target_states[0], float):
            value = (
                np.array(target_states) @ np.array(self.coefficients) + self.constant
            )
        else:
            raise ValueError(f"Unknown type of state {target_states[0]}")
        return value

    def apply_to_variables(self, state: dict[str, gp.MVar]) -> gp.MLinExpr:
        return (
            gp.vstack([state[var] for var in self.variables]).T
            @ np.array(self.coefficients)
            + self.constant
        )

    @property
    def norm(self) -> float:
        return np.linalg.norm(self.coefficients)

    def __neg__(self) -> LinearExpression:
        return dataclasses.replace(
            self, coefficients=[-c for c in self.coefficients], constant=-self.constant
        )

    def __add__(self, other: LinearExpression | float) -> LinearExpression:
        if isinstance(other, float):
            return dataclasses.replace(self, constant=self.constant + other)
        elif isinstance(other, LinearExpression) and self.variables == other.variables:
            return dataclasses.replace(
                self,
                coefficients=[
                    c1 + c2 for c1, c2 in zip(self.coefficients, other.coefficients)
                ],
                constant=self.constant + other.constant,
            )
        elif isinstance(other, LinearExpression) and set(self.variables).isdisjoint(
            other.variables
        ):
            return LinearExpression(
                self.variables + other.variables,
                self.coefficients + other.coefficients,
                self.constant + other.constant,
            )
        else:
            return NotImplemented

    def __mul__(self, other: float) -> LinearExpression:
        return dataclasses.replace(
            self,
            coefficients=[c * other for c in self.coefficients],
            constant=self.constant * other,
        )

    def __rmul__(self, other: float) -> LinearExpression:
        return dataclasses.replace(
            self,
            coefficients=[c * other for c in self.coefficients],
            constant=self.constant * other,
        )


@dataclasses.dataclass
class LinearInequality:
    expr: LinearExpression
    sense: str
    name: str

    @property
    def sign(self) -> int:
        sign = 1 if self.sense == GE else -1
        return sign

    def invert(self) -> LinearInequality:
        return dataclasses.replace(
            self, sense=LE if self.sense == GE else GE, name=f"~{self.name}"
        )

    def to_gt_zero(self) -> LinearInequality:
        if self.sense == GE:
            return self
        else:
            return LinearInequality(-self.expr, GE, name=f"sf({self.name})")

    def __matmul__(self, name: str) -> LinearInequality:
        return dataclasses.replace(self, name=name)
