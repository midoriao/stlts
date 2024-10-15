from __future__ import annotations

import dataclasses
import functools
from typing import TYPE_CHECKING

import gurobipy as gp
import numpy as np

from stlts import stl
from stlts.milp_helper import add_and_aux_var, add_or_aux_var, at_least_one
from stlts.stl_atomic import Predicate

from .constants import EPSILON

if TYPE_CHECKING:
    Vars = gp.tupledict[int, gp.Var]


@dataclasses.dataclass
class MilpVariables:
    N: int
    gamma: Vars
    stl_sat: dict[stl.StlFormula, Vars]
    pred_sat: dict[Predicate, Vars]
    pred_strong_sat: dict[Predicate, Vars]
    sat_durations: dict[stl.BoundedAlw, Vars]
    unsat_durations: dict[stl.BoundedEv, Vars]

    def get_values(self) -> dict:
        def get_values_from_vars(vars: Vars) -> np.ndarray:
            return np.array([vars[i].X for i in vars])

        def get_bool_values_from_vars(vars: Vars) -> np.ndarray:
            return np.array([True if vars[i].X >= 0.5 else False for i in vars])

        return dict(
            gamma=get_values_from_vars(self.gamma),
            stl_sat={
                phi: get_bool_values_from_vars(vars)
                for phi, vars in self.stl_sat.items()
            },
            pred_sat={
                p: get_bool_values_from_vars(self.pred_sat[p]) for p in self.pred_sat
            },
            sat_durations={
                phi: get_values_from_vars(dur)
                for phi, dur in self.sat_durations.items()
            },
            unsat_durations={
                phi: get_values_from_vars(dur)
                for phi, dur in self.unsat_durations.items()
            },
        )


def make_milp_variables(
    milp: gp.Model,
    spec: stl.StlFormula,
    N: int,
    gamma_0: float,
    gamma_N: float,
    gamma_unit_length: float = 0.01,
) -> MilpVariables:
    subformulas = spec.get_subformulas()
    predicates = [phi.predicate for phi in subformulas if isinstance(phi, stl.Atomic)]

    gamma = milp.addVars(N + 1, lb=gamma_0, ub=gamma_N, name='gamma')

    stl_sat = {
        phi: milp.addVars(
            list(range(1, N + 1)), vtype=gp.GRB.BINARY, name=f'stl_{phi.shortname}'
        )
        for phi in subformulas
    }
    pred_sat = {
        p: milp.addVars(
            list(range(0, N + 1)), vtype=gp.GRB.BINARY, name=f'pred_sat_{p.name}'
        )
        for p in predicates
    }
    pred_strong_sat = {
        p: milp.addVars(
            list(range(0, N + 1)), vtype=gp.GRB.BINARY, name=f'pred_strong_sat_{p.name}'
        )
        for p in predicates
    }
    sat_durations = {
        phi: milp.addVars(
            list(range(1, N + 1)),
            lb=0,
            ub=gamma_N - gamma_0,
            name=f'sat_dur_{phi.shortname}',
        )
        for phi in subformulas
        if isinstance(phi, stl.BoundedAlw)
    }
    unsat_durations = {
        phi: milp.addVars(
            list(range(1, N + 1)),
            lb=0,
            ub=gamma_N - gamma_0,
            name=f'unsat_dur_{phi.shortname}',
        )
        for phi in subformulas
        if isinstance(phi, stl.BoundedEv)
    }

    for phi in subformulas:
        if isinstance(phi, stl.BoundedAlw) or isinstance(phi, stl.BoundedEv):
            assert phi.interval[0] < phi.interval[1]
            assert 0 <= phi.interval[0]
            assert phi.interval[1] <= gamma_N - gamma_0 - EPSILON

    return MilpVariables(
        N=N,
        gamma=gamma,
        stl_sat=stl_sat,
        pred_sat=pred_sat,
        pred_strong_sat=pred_strong_sat,
        sat_durations=sat_durations,
        unsat_durations=unsat_durations,
    )


def add_timeline_constraints(
    milp: gp.Model,
    v: MilpVariables,
    gamma_0: float,
    gamma_N: float,
    gamma_unit_length: float = 0.01,
) -> None:
    N = v.N
    gamma = v.gamma

    gamma_length = {i: gamma[i] - gamma[i - 1] for i in range(1, N + 1)}

    milp.addConstr(gamma[0] == gamma_0, name='gamma_0')
    milp.addConstr(gamma[N] == gamma_N, name='gamma_N')

    milp.addConstrs(
        (gamma_length[i] >= EPSILON for i in range(1, N + 1)),
        name='gamma_monotonicity',
    )


def add_predicate_constraints(
    milp: gp.Model,
    v: MilpVariables,
    system_states: dict[str, gp.MVar],
    delta: float = 0.1,
) -> None:
    N = v.N
    pred_sat = v.pred_sat
    pred_strong_sat = v.pred_strong_sat
    predicates = list(pred_sat.keys())

    for p in predicates:
        postfix = f'{p.name}'
        expr_value = p.expr_value(system_states)
        depth = expr_value / p.inequality.expr.norm

        milp.addConstrs(
            (
                pred_sat[p][i - 1] >= pred_strong_sat[p][i] - pred_strong_sat[p][i - 1]
                for i in range(1, N + 1)
            ),
            name=f'stationary_rise_{postfix}',
        )
        milp.addConstrs(
            (
                pred_sat[p][i + 1] >= pred_strong_sat[p][i] - pred_strong_sat[p][i + 1]
                for i in range(N)
            ),
            name=f'stationary_fall_{postfix}',
        )
        milp.addConstrs(
            ((pred_sat[p][i] == 1) >> (depth[i].item() >= 0) for i in range(N + 1)),
            name=f'pred_sat_geq_{postfix}',
        )
        milp.addConstrs(
            (
                (pred_sat[p][i] == 0) >> (depth[i].item() <= -EPSILON)
                for i in range(N + 1)
            ),
            name=f'pred_sat_lt_{postfix}',
        )
        milp.addConstrs(
            (
                (pred_strong_sat[p][i] == 1) >> (depth[i].item() >= delta)
                for i in range(N + 1)
            ),
            name=f'pred_strong_sat_geq_{postfix}',
        )
        milp.addConstrs(
            (
                (pred_strong_sat[p][i] == 0) >> (depth[i].item() <= delta - EPSILON)
                for i in range(N + 1)
            ),
            name=f'pred_strong_sat_lt_{postfix}',
        )
        milp.addConstrs(
            (pred_sat[p][i] >= pred_strong_sat[p][i] for i in range(N + 1)),
            name=f'strong_sat_then_sat_{postfix}',
        )  # redundant


### STL constraints


def add_full_stl_constraints(
    milp: gp.Model,
    v: MilpVariables,
    root: stl.StlFormula,
    require_satisfaction=True,
) -> None:
    root_subformulas = root.get_subformulas()
    known_formulas = list(v.stl_sat.keys())

    for phi in root_subformulas:
        if phi not in known_formulas:
            raise ValueError(f"Unknown subformula {phi} in the STL formula")

    subformulas = [
        phi for phi in known_formulas if phi in root_subformulas
    ]  # only consider known formulas that are in the root formula

    for phi in subformulas:
        add_stl_constraints(phi, milp, v)
    if require_satisfaction:
        milp.addConstr(v.stl_sat[root][1] == True, name='root_satisfaction')


@functools.singledispatch
def add_stl_constraints(
    phi: stl.StlFormula,
    milp: gp.Model,
    v: MilpVariables,
) -> None:
    raise NotImplementedError(f"Unsupported STL formula {phi}")


@add_stl_constraints.register
def _(phi: stl.Top, milp: gp.Model, v: MilpVariables):
    phi_sat = v.stl_sat[phi]
    milp.addConstrs((phi_sat[i] == True for i in phi_sat), name=f'stl[top]')


@add_stl_constraints.register
def _(phi: stl.Bottom, milp: gp.Model, v: MilpVariables):
    phi_sat = v.stl_sat[phi]
    milp.addConstrs((phi_sat[i] == False for i in phi_sat), name=f'stl[bottom]')


@add_stl_constraints.register
def _(phi: stl.And, milp: gp.Model, v: MilpVariables):
    phi_sat = v.stl_sat[phi]
    children_sat = [v.stl_sat[child] for child in phi.children]
    def children_sat_and(i):
        return gp.and_(*[child[i] for child in children_sat])
    milp.addConstrs(
        (phi_sat[i] == children_sat_and(i) for i in phi_sat),
        name=f'stl[and]_{phi.shortname}',
    )


@add_stl_constraints.register
def _(phi: stl.Or, milp: gp.Model, v: MilpVariables):
    phi_sat = v.stl_sat[phi]
    children_sat = [v.stl_sat[child] for child in phi.children]
    def children_sat_or(i):
        return gp.or_(*[child[i] for child in children_sat])
    milp.addConstrs(
        (phi_sat[i] == children_sat_or(i) for i in phi_sat),
        name=f'stl[or]_{phi.shortname}',
    )


@add_stl_constraints.register
def _(phi: stl.Until, milp: gp.Model, v: MilpVariables):
    postfix = f'{phi.shortname}'
    N = v.N
    z1 = milp.addVars(list(range(1, N)), vtype=gp.GRB.BINARY, name=f'z1_{postfix}')
    z2 = milp.addVars(list(range(1, N)), vtype=gp.GRB.BINARY, name=f'z2_{postfix}')
    milp.addConstr(
        v.stl_sat[phi][N] == gp.and_(v.stl_sat[phi.phi1][N], v.stl_sat[phi.phi2][N]),
        name=f'stl[until]_N_{postfix}',
    )
    for i in range(1, N):
        z1[i] = add_and_aux_var(
            milp,
            v.stl_sat[phi][i + 1],
            v.stl_sat[phi.phi1][i],
            name=f'aux1_{postfix}[{i}]',
        )
        z2[i] = add_and_aux_var(
            milp,
            v.stl_sat[phi.phi1][i],
            v.stl_sat[phi.phi2][i],
            name=f'aux2_{postfix}[{i}]',
        )
        milp.addConstr(
            v.stl_sat[phi][i] == gp.or_(z1[i], z2[i]), name=f'stl_until_{postfix}[{i}]'
        )


@add_stl_constraints.register
def _(phi: stl.Release, milp: gp.Model, v: MilpVariables):
    postfix = f'{phi.shortname}'
    N = v.N
    z1 = milp.addVars(list(range(1, N)), vtype=gp.GRB.BINARY, name=f'z1_{postfix}')
    z2 = milp.addVars(list(range(1, N)), vtype=gp.GRB.BINARY, name=f'z2_{postfix}')
    milp.addConstr(
        v.stl_sat[phi][N] == gp.or_(v.stl_sat[phi.phi1][N], v.stl_sat[phi.phi2][N]),
        name=f'stl[release]_N_{postfix}',
    )
    for i in range(1, N):
        z1[i] = add_or_aux_var(
            milp,
            v.stl_sat[phi][i + 1],
            v.stl_sat[phi.phi1][i],
            name=f'aux1_{postfix}[{i}]',
        )
        z2[i] = add_or_aux_var(
            milp,
            v.stl_sat[phi.phi1][i],
            v.stl_sat[phi.phi2][i],
            name=f'aux2_{postfix}[{i}]',
        )
        milp.addConstr(
            v.stl_sat[phi][i] == gp.and_(z1[i], z2[i]),
            name=f'stl[release]_{postfix}[{i}]',
        )


@add_stl_constraints.register
def _(phi: stl.Alw, milp: gp.Model, v: MilpVariables):
    phi_sat = v.stl_sat[phi]
    postfix = f'{phi.shortname}'
    N = v.N
    milp.addConstr(phi_sat[N] == v.stl_sat[phi.child][N], name=f'stl[alw]_N_{postfix}')
    milp.addConstrs(
        (
            phi_sat[i] == gp.and_(phi_sat[i + 1], v.stl_sat[phi.child][i])
            for i in range(1, N)
        ),
        name=f'stl[alw]_{postfix}',
    )


@add_stl_constraints.register
def _(phi: stl.Ev, milp: gp.Model, v: MilpVariables):
    phi_sat = v.stl_sat[phi]
    postfix = f'{phi.shortname}'
    N = v.N
    milp.addConstr(phi_sat[N] == v.stl_sat[phi.child][N], name=f'stl[ev]_N_{postfix}')
    milp.addConstrs(
        (
            phi_sat[i] == gp.or_(phi_sat[i + 1], v.stl_sat[phi.child][i])
            for i in range(1, N)
        ),
        name=f'stl[ev]_{postfix}',
    )


@add_stl_constraints.register
def _(phi: stl.Atomic, milp: gp.Model, v: MilpVariables):
    phi_sat = v.stl_sat[phi]
    pred_strong_sat = v.pred_strong_sat[phi.predicate]
    milp.addConstrs(
        (
            phi_sat[i] == gp.or_(pred_strong_sat[i - 1], pred_strong_sat[i])
            for i in phi_sat
        ),
        name=f'stl[atomic]_{phi.shortname}',
    )


@add_stl_constraints.register
def _(phi: stl.BoundedAlw, milp: gp.Model, v: MilpVariables):
    _bounded_alw_constraints(
        milp=milp,
        phi_sat=v.stl_sat[phi],
        child_sat=v.stl_sat[phi.child],
        milp_prefix=f'stl[alwI]_{phi.shortname}',
        sat_durations=v.sat_durations[phi],
        interval=phi.interval,
        gamma=v.gamma,
    )


@add_stl_constraints.register
def _(phi: stl.BoundedEv, milp: gp.Model, v: MilpVariables):
    phi_sat = v.stl_sat[phi]
    child_sat = v.stl_sat[phi.child]

    phi_unsat = milp.addVars(
        phi_sat.keys(), vtype=gp.GRB.BINARY, name=f'unsat_{phi.shortname}'
    )
    milp.addConstrs(
        (phi_unsat[i] == 1 - phi_sat[i] for i in phi_sat), name=f'unsat_{phi.shortname}'
    )
    child_unsat = milp.addVars(
        child_sat.keys(), vtype=gp.GRB.BINARY, name=f'unsat_child_{phi.shortname}'
    )
    milp.addConstrs(
        (child_unsat[i] == 1 - child_sat[i] for i in child_sat),
        name=f'unsat_child_{phi.shortname}',
    )

    _bounded_alw_constraints(
        milp=milp,
        phi_sat=phi_unsat,
        child_sat=child_unsat,
        milp_prefix=f'stl[evI]_{phi.shortname}',
        sat_durations=v.unsat_durations[phi],
        interval=phi.interval,
        gamma=v.gamma,
    )


def _bounded_alw_constraints(
    milp: gp.Model,
    phi_sat: Vars,
    child_sat: Vars,
    milp_prefix: str,
    sat_durations: Vars,
    interval: tuple[float, float],
    gamma: Vars,
) -> None:
    """
    Suppose Alw_I(phi) is strongly satisfactory in Gamma[i].
    For any interval J that intersects with Gamma[i] + I, phi must strongly satisfactory in J.
    """
    ## Prepare variables
    N = len(phi_sat)
    a, b = interval
    gamma_length = {i: gamma[i] - gamma[i - 1] for i in range(1, N + 1)}

    _indicator: dict[tuple[str, int, str, int], gp.Var] = {}

    def _indicate(kind, i, op, j) -> gp.Var:
        if (kind, i, op, j) in _indicator:
            return _indicator[kind, i, op, j]
        assert kind in ['inf', 'sup']
        assert op in ['gt', 'lt', 'geq', 'leq']
        assert 0 <= i <= N or (i == N + 1 and kind == 'inf')
        assert 0 <= j <= N

        name = f'{milp_prefix}_indicate_[{kind}_{i}_{op}_{j}]'

        z = _indicator[kind, i, op, j] = milp.addVar(vtype=gp.GRB.BINARY, name=name)

        if kind == 'inf':
            lhs = gamma[i - 1] + interval[0]
        elif kind == 'sup':
            lhs = gamma[i] + interval[1]

        if op == 'gt':
            milp.addConstr(
                (z == 1) >> (lhs >= gamma[j] + EPSILON),
            )
        elif op == 'lt':
            milp.addConstr(
                (z == 1) >> (lhs <= gamma[j] - EPSILON),
            )
        elif op == 'geq':
            milp.addConstr(
                (z == 1) >> (lhs >= gamma[j]),
            )
        elif op == 'leq':
            milp.addConstr(
                (z == 1) >> (lhs <= gamma[j]),
            )
        return z

    ## Constraints for |R_i|
    # R_1 = 0 if <phi>_1 = 0 else Gamma[1]
    # R_i = 0 if <phi>_i = 0 else Gamma[i] + R_{i-1} for i = [2, N]
    milp.addConstrs(
        ((child_sat[i] == 0) >> (sat_durations[i] == 0) for i in range(1, N + 1)),
        name=f'{milp_prefix}_sat_dur_unsat',
    )

    _length = lambda i: (0 if i == 1 else sat_durations[i - 1]) + gamma_length[i]  # type: ignore
    milp.addConstrs(
        (
            (child_sat[i] == 1) >> (sat_durations[i] == _length(i))
            for i in range(1, N + 1)
        ),
        name=f'{milp_prefix}_sat_dur_sat',
    )

    ## Constraints for <Alw_I psi>_i

    for i in range(1, N + 1):
        milp_prefix_i = f"{milp_prefix}[{i}]"
        # Rule `Alw_I_pos_1`:
        #   <Alw_I psi>_i = 1, intersect(Gamma[i] + I, Gamma[j]) ==> <psi>_j = 1
        #   for j = [i, N]
        milp.addConstrs(
            (
                at_least_one(
                    1 - phi_sat[i],
                    _indicate('sup', i, 'leq', j - 1),
                    _indicate('inf', i, 'gt', j),
                    child_sat[j],
                )
                for j in range(i, N + 1)
            ),
            name=f"{milp_prefix_i}_pos_1",
        )
        # Rule `Alw_I_pos_1N`:
        #   <Alw_I psi>_i = 1, intersect(Gamma[i] + I, (gamma[N], +inf)) ==> <psi>_N = 1
        milp.addConstr(
            at_least_one(1 - phi_sat[i], _indicate('sup', i, 'lt', N), child_sat[N]),
            name=f"{milp_prefix_i}_pos_1N",
        )

        # Rule `Alw_I_neg_1`:
        #   <Alw_I psi>_i = 0, gamma[i - 1] + b <= gamma[j] <= gamma[i] + b ==> |R_j| < |I|
        #   for j = [i, N]
        z = milp.addVars(
            ['neg_1', 'neg_2', 'neg_3'],
            range(i, N + 1),
            vtype=gp.GRB.BINARY,
            name=f'{milp_prefix_i}_z',
        )
        milp.addConstrs(
            (
                at_least_one(
                    phi_sat[i],
                    _indicate('sup', i - 1, 'geq', j),
                    _indicate('sup', i, 'leq', j),
                    z['neg_1', j],
                )
                for j in range(i, N + 1)
            ),
            name=f"{milp_prefix_i}_neg_1",
        )
        milp.addConstrs((
            (z['neg_1', j] == 1) >> (sat_durations[j] <= (b - a))
            for j in range(i, N + 1)
        ))

        # Rule `Alw_I_neg_2`:
        #   <Alw_I psi>_i = 0, gamma[j - 1] <= gamma[i] + b <= gamma[j]  ==> |R_j| < gamma[j] - gamma[i] - a
        #   for j = [i + 1, N]
        milp.addConstrs(
            (
                at_least_one(
                    phi_sat[i],
                    _indicate('sup', i, 'lt', j - 1),
                    _indicate('sup', i, 'gt', j),
                    z['neg_2', j],
                )
                for j in range(i + 1, N + 1)
            ),
            name=f"{milp_prefix_i}_neg_2",
        )

        milp.addConstrs((
            (z['neg_2', j] == 1) >> (sat_durations[j] <= gamma[j] - gamma[i] - a)
            for j in range(i + 1, N + 1)
        ))

        # Rule `Alw_I_neg_3`:
        #   <Alw_I psi>_i = 0, gamma[i] + a <= gamma[N] <= gamma[i] + b ==> |R_N| < gamma[N] - gamma[i] - a
        milp.addConstr(
            at_least_one(
                phi_sat[i],
                _indicate('inf', i + 1, 'gt', N),
                _indicate('sup', i, 'lt', N),
                z['neg_3', N],
            ),
            name=f"{milp_prefix_i}_neg_3",
        )
        milp.addConstr(
            (z['neg_3', N] == 1) >> (sat_durations[N] <= (gamma[N] - gamma[i] - a))
        )

        # Rule `Alw_I_neg_4`:
        #   <Alw_I psi>_i = 0, gamma[N] <= gamma[i] + a  ==> <psi>_N = 0
        milp.addConstr(
            at_least_one(
                phi_sat[i], _indicate('inf', i + 1, 'leq', N), 1 - child_sat[N]
            ),
            name=f"{milp_prefix_i}_neg_4",
        )
