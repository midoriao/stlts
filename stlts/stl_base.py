from __future__ import annotations

import numpy as np

from .trace import Trace


class StlFormula:
    """STL formula."""

    def __init__(self, name: str, children: list[StlFormula] | None = None):
        self.name = name
        self.bounded = False
        self.interval = None
        self._timing = None

        if children is None:
            self.children = []
        else:
            self.children = children

    def __hash__(self) -> int:
        return hash(self.name)

    @property
    def shortname(self) -> str:
        return self.name[:200]

    @property
    def child(self) -> StlFormula:
        assert len(self.children) == 1
        return self.children[0]

    def get_subformula(self, name: str) -> StlFormula | None:
        for phi in self.get_subformulas():
            if phi.name == name:
                return phi
        return None

    def get_subformulas(self, depth: int | None = None) -> list[StlFormula]:
        if depth is not None and depth == 0:
            return [self]
        if len(self.children) == 0:
            return [self]
        else:
            subformulas = []
            for child in self.children:
                candidates = child.get_subformulas(
                    depth=depth - 1 if depth is not None else None
                )
                for p in candidates:
                    if p.name not in [q.name for q in subformulas]:
                        subformulas.append(p)
            return [*subformulas, self]

    def __repr__(self) -> str:
        return self.name

    def robust_semantics(self, trace: Trace, memo=None) -> np.ndarray:
        """Return the robustness value of the formula at each step of the trace."""
        raise NotImplementedError

    def negation(self) -> StlFormula:
        raise NotImplementedError
