from __future__ import annotations

from .node_base import StlFormula


def make_unique(root: StlFormula):
    formula_visited = {}

    for phi in root.get_subformulas():
        if phi.name not in formula_visited:
            formula_visited[phi.name] = phi
        for i, child in enumerate(phi.children):
            if child.name in formula_visited:
                phi.children[i] = formula_visited[child.name]

    return formula_visited[root.name]
