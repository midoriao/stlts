from __future__ import annotations

from typing import Callable

from .linear_expression import LinearInequality

EPSILON = 1e-3


class Approx:
    def __init__(self, value: float, epsilon: float = EPSILON):
        self.epsilon = epsilon
        self.value = value

    def __eq__(self, other):
        return abs(other - self.value) <= self.epsilon

    def __le__(self, other):
        return self.value - other <= self.epsilon

    def __ge__(self, other):
        return self.value - other >= -self.epsilon


class ApproxRange:
    def __init__(self, lb, ub, epsilon: float = EPSILON):
        self.epsilon = epsilon
        self.lb = lb
        self.ub = ub

    def __eq__(self, other):
        return self.lb - self.epsilon <= other <= self.ub + self.epsilon


class ValidatorRules:
    rules: list[tuple[LinearInequality, Callable | None, str | None]]

    def __init__(self, rules: list | None = None):
        if rules is None:
            rules = []
        self.rules = rules

    def append(
        self, inequality: LinearInequality, condition=None, tag: str | None = None
    ):
        self.rules.append((inequality, condition, tag))

    def filter_by_tag(self, tag: str) -> ValidatorRules:
        return ValidatorRules([x for x in self.rules if x[2] == tag])

    def validate(self, input_1, input_optional=None) -> None:
        for inequality, condition, _ in self.rules:
            if condition is None:
                triggered = True
            elif input_optional is None:
                triggered = condition(input_1)
            else:
                triggered = condition(input_1, input_optional)

            if triggered:
                value = inequality.to_gt_zero().expr.eval(input_1)
                assert value >= Approx(
                    0
                ), f"{inequality.name} violated  with value {value}, at {input_1}"
