import gurobipy as gp
import pytest

from stlts import benchmarks
from stlts.result_helper import validate_robust_semantics

RESTRICTED_LICENSE = ((m := gp.Model()).Params.LicenseID == 0)

@pytest.fixture
def milp():
    milp = gp.Model()
    milp.setParam('MIPFocus', 1)
    milp.setParam('FeasibilityTol', 1e-7)
    milp.setParam('IntFeasTol', 1e-7)
    return milp


@pytest.mark.parametrize('spec', ['rnc1'])
def test_chasing_car(milp, spec):
    prob = benchmarks.get_chasing_car(
        milp,
        N=6,
        spec=spec,
        optimize=False,
        delta=0.1,
    )
    prob.search_satisfaction()
    assert prob.has_solution
    assert prob.system_model.validate(prob.get_trace_result(interpolation=False))
    assert prob.system_model.validate(prob.get_trace_result(interpolation=True))
    assert validate_robust_semantics(prob)


@pytest.mark.skipif(RESTRICTED_LICENSE, reason='Free Gurobi license does not support large MILP')
@pytest.mark.parametrize('spec', ['nav1'])
def test_robot_navigation(milp, spec):
    prob = benchmarks.get_robot_navigation(
        milp,
        N=19,
        spec=spec,
        optimize=False,
        delta=0.1,
    )
    prob.search_satisfaction()
    assert prob.has_solution
    assert prob.system_model.validate(prob.get_trace_result(interpolation=False))
    assert prob.system_model.validate(prob.get_trace_result(interpolation=True))
    assert validate_robust_semantics(prob)


@pytest.mark.skipif(RESTRICTED_LICENSE, reason='Free Gurobi license does not support large MILP')
@pytest.mark.parametrize('spec', ['iso1'])
def test_iso_rss(milp, spec):
    prob = benchmarks.get_iso_rss(
        milp,
        N=6,
        spec=spec,
        optimize=False,
        delta=0.1,
    )
    prob.search_satisfaction()
    assert prob.has_solution
    assert prob.system_model.validate(prob.get_trace_result(interpolation=False))
    assert prob.system_model.validate(prob.get_trace_result(interpolation=True))
    assert validate_robust_semantics(prob, depth=1)
