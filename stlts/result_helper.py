from __future__ import annotations

import numpy as np
import pandas as pd

from stlts.stl_base import StlFormula

from .problem import StlMilpProblem
from .validator import Approx


def get_robust_semantics_result(
    prob: StlMilpProblem, depth: int | None = None, stl_spec: StlFormula | None = None
) -> pd.DataFrame:
    if stl_spec is None:
        stl_spec = prob.stl_spec
    trace = prob.get_trace_result(interpolation=True)

    memo = {}
    for phi in stl_spec.get_subformulas(depth=depth):
        rob = phi.robust_semantics(trace, memo)
        memo[phi.name] = rob

    df = pd.DataFrame(
        memo,
        index=trace.time,
    )
    df.index.name = 't'
    return df


def validate_robust_semantics(prob: StlMilpProblem, depth=None) -> bool:
    trace = prob.get_trace_result(interpolation=True)
    stl_spec = prob.stl_spec
    gamma = prob.get_gamma_result()
    delta = prob.delta
    stl_sat = prob.milp_variables.stl_sat

    memo = {}
    for phi in stl_spec.get_subformulas(depth=depth):
        robustness = phi.robust_semantics(trace, memo)
        memo[phi.name] = robustness

        for k, Gamma in enumerate(zip(gamma[:-1], gamma[1:]), 1):
            target = (
                np.searchsorted(trace.time, Gamma[0]),
                np.searchsorted(trace.time, Gamma[1]),
            )
            if target[0] == target[1]:
                continue  # skip empty interval

            if stl_sat[phi][k].X >= 0.5:
                assert robustness[target[0] : target[1]].min() >= Approx(0), f'min robustness of `{phi.name}` in Gamma_{k}=[{Gamma[0]}, {Gamma[1]}] was {robustness[target[0]:target[1]].min()}'  # fmt: skip
            else:
                assert robustness[target[0] : target[1]].max() <= Approx(delta), f'max robustness of `{phi.name}` in Gamma_{k}=[{Gamma[0]}, {Gamma[1]}] was {robustness[target[0]:target[1]].max()}'  # fmt: skip

    return True
