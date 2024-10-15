from __future__ import annotations

import gurobipy as gp
import numpy as np

from .stl_atomic import Atomic
from .stl_base import StlFormula
from .trace import Trace


class Top(StlFormula):
    def __init__(self):
        super().__init__("True")

    def negation(self) -> StlFormula:
        return Bottom()

    def robust_semantics(self, trace: Trace, memo=None) -> np.ndarray:
        return np.ones(len(trace.time)) * np.inf


class Bottom(StlFormula):
    def __init__(self):
        super().__init__("False")

    def negation(self) -> StlFormula:
        return Top()

    def robust_semantics(self, trace: Trace, memo=None) -> np.ndarray:
        return -np.ones(len(trace.time)) * np.inf


class And(StlFormula):
    def __init__(self, *phis: StlFormula):
        super().__init__(f"({'&'.join(phi.name for phi in phis)})", children=list(phis))

    def negation(self) -> StlFormula:
        return Or(*[phi.negation() for phi in self.children])

    def robust_semantics(self, trace: Trace, memo=None) -> np.ndarray:
        if memo is not None and self.name in memo:
            return memo[self.name]

        robs = [child.robust_semantics(trace, memo) for child in self.children]
        return np.minimum.reduce(robs)


class Or(StlFormula):
    def __init__(self, *phis: StlFormula):
        super().__init__(f"({'|'.join(phi.name for phi in phis)})", children=list(phis))

    def negation(self) -> StlFormula:
        return And(*[phi.negation() for phi in self.children])

    def robust_semantics(self, trace: Trace, memo: None) -> np.ndarray:
        if memo is not None and self.name in memo:
            return memo[self.name]
        robs = [child.robust_semantics(trace, memo) for child in self.children]
        return np.maximum.reduce(robs)


class Until(StlFormula):
    def __init__(self, phi1: StlFormula, phi2: StlFormula):
        super().__init__(f"({phi1.name})Until({phi2.name})", children=[phi1, phi2])

    @property
    def phi1(self) -> StlFormula:
        return self.children[0]

    @property
    def phi2(self) -> StlFormula:
        return self.children[1]

    def negation(self) -> StlFormula:
        return Release(self.phi1.negation(), self.phi2.negation())

    def robust_semantics(self, trace: Trace, memo=None) -> np.ndarray:
        if memo is not None and self.name in memo:
            return memo[self.name]

        rho1 = self.phi1.robust_semantics(trace, memo)
        rho2 = self.phi2.robust_semantics(trace, memo)
        return np.array([
            np.max([
                np.minimum(rho2[j], rho1[i : j + 1].min()) for j in range(i, len(rho1))
            ])
            for i in range(len(rho1))
        ])


class Release(StlFormula):
    def __init__(self, phi1: StlFormula, phi2: StlFormula):
        super().__init__(f"({phi1.name})Release({phi2.name})", children=[phi1, phi2])

    @property
    def phi1(self) -> StlFormula:
        return self.children[0]

    @property
    def phi2(self) -> StlFormula:
        return self.children[1]

    def negation(self) -> StlFormula:
        return Until(self.phi1.negation(), self.phi2.negation())

    def robust_semantics(self, trace: Trace, memo=None) -> np.ndarray:
        if memo is not None and self.name in memo:
            return memo[self.name]

        rho1 = self.phi1.robust_semantics(trace, memo)
        rho2 = self.phi2.robust_semantics(trace, memo)
        return np.array([
            np.min([
                np.maximum(rho2[j], rho1[i : j + 1].max()) for j in range(i, len(rho1))
            ])
            for i in range(len(rho1))
        ])


class Ev(StlFormula):
    def __init__(self, phi: StlFormula):
        super().__init__(f"Ev({phi.name})", children=[phi])

    def negation(self) -> StlFormula:
        return Alw(self.child.negation())

    def robust_semantics(self, trace: Trace, memo=None) -> np.ndarray:
        if memo is not None and self.name in memo:
            return memo[self.name]

        rho = self.child.robust_semantics(trace, memo)
        return np.array([np.max(rho[i:]) for i in range(len(rho))])


class Alw(StlFormula):
    def __init__(self, phi: StlFormula):
        super().__init__(f"Alw({phi.name})", children=[phi])

    def negation(self) -> StlFormula:
        return Ev(self.child.negation())

    def robust_semantics(self, trace: Trace, memo=None) -> np.ndarray:
        if memo is not None and self.name in memo:
            return memo[self.name]

        rho = self.child.robust_semantics(trace, memo)
        return np.array([np.min(rho[i:]) for i in range(len(rho))])


class BoundedAlw(StlFormula):
    interval: tuple[float, float]

    def __init__(self, interval: tuple[float, float], phi: StlFormula):
        super().__init__(
            f"Alw_[{interval[0]},{interval[1]}]({phi.name})", children=[phi]
        )
        self.bounded = True
        self.interval = interval

    def negation(self) -> StlFormula:
        return BoundedEv(self.interval, self.child.negation())

    def robust_semantics(self, trace: Trace, memo=None) -> np.ndarray:
        if memo is not None and self.name in memo:
            return memo[self.name]

        rho = self.child.robust_semantics(trace, memo)

        interval = self.interval
        if isinstance(interval[0], gp.Var):
            interval = (interval[0].X, interval[1])
        if isinstance(interval[1], gp.Var):
            interval = (interval[0], interval[1].X)

        def intersecting_index(i: int):
            if trace.time[i] + interval[0] > trace.time[-1]:
                return trace.time == trace.time[-1]
            return (trace.time >= trace.time[i] + interval[0]) & (
                trace.time <= trace.time[i] + interval[1]
            )

        return np.array([rho[intersecting_index(i)].min() for i in range(len(rho))])


class BoundedEv(StlFormula):
    interval: tuple[float, float]

    def __init__(self, interval: tuple[float, float], phi: StlFormula):
        super().__init__(
            f"Ev_[{interval[0]},{interval[1]}]({phi.name})", children=[phi]
        )
        self.bounded = True
        self.interval = interval

    def negation(self) -> StlFormula:
        return BoundedAlw(self.interval, self.child.negation())

    def robust_semantics(self, trace: Trace, memo=None) -> np.ndarray:
        if memo is not None and self.name in memo:
            return memo[self.name]

        rho = self.child.robust_semantics(trace, memo)

        interval = self.interval
        if isinstance(interval[0], gp.Var):
            interval = (interval[0].X, interval[1])
        if isinstance(interval[1], gp.Var):
            interval = (interval[0], interval[1].X)

        def intersecting_index(i: int):
            if trace.time[i] + interval[0] > trace.time[-1]:
                return trace.time == trace.time[-1]
            return (trace.time >= trace.time[i] + interval[0]) & (
                trace.time <= trace.time[i] + interval[1]
            )

        return np.array([rho[intersecting_index(i)].max() for i in range(len(rho))])


def BoundedUntil(
    interval: tuple[float, float], phi1: StlFormula, phi2: StlFormula
) -> StlFormula:
    if interval[0] == 0:
        until = Until(phi1, phi2)
    else:
        until = BoundedAlw((0, interval[0]), Until(phi1, phi2))
    return And(BoundedEv(interval, phi2), until)


def BoundedRelease(
    interval: tuple[float, float], phi1: StlFormula, phi2: StlFormula
) -> StlFormula:
    if interval[0] == 0:
        release = Release(phi1, phi2)
    else:
        release = BoundedEv((0, interval[0]), Release(phi1, phi2))
    return Or(BoundedAlw(interval, phi2), release)


def Implies(phi1: StlFormula, phi2: StlFormula) -> StlFormula:
    return Or(phi1.negation(), phi2)


def atomic_predicates(phi: StlFormula) -> list[Atomic]:
    """Return a list of atomic predicates in the formula."""
    if isinstance(phi, Atomic):
        return [phi]
    elif isinstance(phi, BoundedEv) and isinstance(phi.phantom_child, Atomic):
        return [phi.phantom_child]
    else:
        result = []
        for child in phi.children:
            candidates = atomic_predicates(child)
            for new in candidates:
                if new.name not in [p.name for p in result]:
                    result.append(new)
        return result


def make_unique(root: StlFormula):
    formula_visited = {}

    for phi in root.get_subformulas():
        if phi.name not in formula_visited:
            formula_visited[phi.name] = phi
        for i, child in enumerate(phi.children):
            if child.name in formula_visited:
                phi.children[i] = formula_visited[child.name]

    return formula_visited[root.name]
