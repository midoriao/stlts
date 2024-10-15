from __future__ import annotations

import gurobipy as gp

from stlts import StlMilpProblem, create_stl_milp_problem
from stlts.linear_expression import LinearExpression as L
from stlts.milp_helper import ensure_monotonicity_by_derivative
from stlts.stl import (
    Alw,
    And,
    Atomic,
    BoundedAlw,
    BoundedEv,
    BoundedUntil,
    Ev,
    Implies,
    Or,
    Until,
    make_unique,
)


def get_benchmark(milp: gp.Model, name: str, **kwargs) -> StlMilpProblem:
    if name.startswith('rnc'):
        return get_chasing_car(milp, spec=name, **kwargs)
    elif name.startswith('nav'):
        return get_robot_navigation(milp, spec=name, **kwargs)
    elif name.startswith('iso'):
        return get_iso_rss(milp, spec=name, **kwargs)
    else:
        raise ValueError(f'Unknown benchmark: {name}')


def get_chasing_car(
    milp: gp.Model,
    N: int = 10,
    spec: str = 'rnc1',
    optimize=False,
    delta=0.1,
) -> StlMilpProblem:
    prob = create_stl_milp_problem(
        milp,
        N=N,
        delta=delta,
        gamma_N=20.0,
        gamma_unit_length=0.005,
        use_binary_expansion=True,
    )

    # System Dynamics
    car = prob.create_system_model()

    x1 = car.add_state('x1', 0, 500)
    v1 = car.add_state('v1', 2, 27)
    a1 = car.add_state('a1', -3, 3)

    x2 = car.add_state('x2', 0, 500)
    v2 = car.add_state('v2', 2, 27)
    a2 = car.add_state('a2', -3, 3)

    car.set_initial_state(
        x1=(45.0, 100.0),
        x2=(0.0, 5.0),
    )

    # Dynamics
    car.add_dynamics('a1', -3, 3, constant=True)
    car.add_dynamics('a2', -3, 3, constant=True)

    car.add_double_integrator_dynamics('x1', 'v1', 'a1')
    car.add_double_integrator_dynamics('x2', 'v2', 'a2')

    # Predicates in linear inequality form
    distance = L.f(1.0, 'x1', -1.0, 'x2')

    danger = (distance <= 10) @ 'danger'
    precedence = (distance >= 0.0) @ 'precedence'
    accelerating = (L.unit('a2') >= 1) @ 'accelerating'

    ensure_monotonicity_by_derivative(
        milp, [v2[i].item() - v1[i].item() for i in range(prob.N + 1)], prob.N
    )

    # Stl Spec

    if spec == 'rnc1' or spec == 'rnc3':
        if spec == 'rnc1':
            safe_t = 1
        elif spec == 'rnc3':
            safe_t = 0.2
        sticky_accel = BoundedAlw((0, safe_t), Atomic(accelerating))
        control_spec = (
            Atomic(precedence),
            Implies(
                Ev(Atomic(danger)),
                Until(sticky_accel, Atomic(danger)),
            ),
        )

        scenario_spec = (BoundedEv((0, 9), BoundedAlw((0, 1), Atomic(danger))),)
        stl_spec = make_unique(
            And(
                Alw(And(*control_spec)),
                *scenario_spec,
            )
        )
    elif spec == 'rnc2':
        control_spec = (Atomic(precedence),)

        scenario_spec = (
            BoundedEv(
                (0, 9),
                And(
                    BoundedAlw((0, 1), Atomic(accelerating)),
                    BoundedAlw((0, 1), Atomic(danger)),
                    BoundedEv((1, 5), Atomic(danger).negation()),
                ),
            ),
        )

        stl_spec = make_unique(
            And(
                Alw(And(*control_spec)),
                *scenario_spec,
            )
        )
    else:
        raise ValueError(f'Unknown spec: {spec}')

    prob.initialize_milp_formulation(stl_spec)

    return prob


def get_robot_navigation(
    milp: gp.Model,
    N: int = 10,
    spec: str = 'nav1',
    optimize=False,
    delta=0.01,
) -> StlMilpProblem:
    horizon = 40.0

    prob = create_stl_milp_problem(
        milp,
        N=N,
        delta=delta,
        gamma_N=horizon,
        gamma_unit_length=0.01,
        use_binary_expansion=False,
    )

    car = prob.create_system_model()

    x = car.add_state('x', 0, 10)
    y = car.add_state('y', 0, 10)

    modes = [
        LOC1 := 'loc1',
        LOC2 := 'loc2',
        LOC3 := 'loc3',
        LOC4 := 'loc4',
    ]

    for mode in modes:
        car.add_mode(mode)

    car.set_initial_state(
        x=(0, 3),
        y=0.0,
    )

    # Mode constraints should be manually maintained
    car.milp.addConstr(
        sum(car.mode[var] for var in modes) == 1,
        name='mode_exact_one',
    )

    # Dynamics
    car.add_dynamics('x', 1, mode=LOC1)
    car.add_dynamics('y', 0.1, 2, mode=LOC1)
    car.add_invariants(
        LOC1,
        L.unit('x') >= 0,
        L.unit('x') <= 5,
        L.unit('y') >= 5,
        L.unit('y') <= 10,
    )

    car.add_dynamics('x', 0.1, 2, mode=LOC2)
    car.add_dynamics('y', -1, mode=LOC2)
    car.add_invariants(
        LOC2,
        L.unit('x') >= 5,
        L.unit('x') <= 10,
        L.unit('y') >= 4,
        L.unit('y') <= 10,
    )

    car.add_dynamics('x', -1, mode=LOC3)
    car.add_dynamics('y', -2, -0.1, mode=LOC3)
    car.add_invariants(
        LOC3,
        L.unit('x') >= 5,
        L.unit('x') <= 10,
        L.unit('y') >= 0,
        L.unit('y') <= 4,
    )

    car.add_dynamics('x', -2, -0.1, mode=LOC4)
    car.add_dynamics('y', 1, mode=LOC4)
    car.add_invariants(
        LOC4,
        L.unit('x') >= 0,
        L.unit('x') <= 5,
        L.unit('y') >= 0,
        L.unit('y') <= 5,
    )

    goal_region = And(
        Atomic(L.unit('x') >= 4.0),
        Atomic(L.unit('x') <= 6.0),
        Atomic(L.unit('y') >= 2.0),
        Atomic(L.unit('y') <= 5.0),
    )

    unsafe_region = And(Atomic(L.unit('x') >= 9.0))

    in_l1 = And(
        Atomic(L.unit('x') <= 5.0),
        Atomic(L.unit('y') >= 5.0),
    )
    in_l3 = And(
        Atomic(L.unit('x') >= 5.0),
        Atomic(L.unit('y') <= 4.0),
    )

    in_l4 = And(
        Atomic(L.unit('x') <= 5.0),
        Atomic(L.unit('y') <= 5.0),
    )

    if spec == 'nav1':
        scenario_spec = [
            Ev(BoundedAlw((0, 3), goal_region)),
            Ev(unsafe_region).negation(),
        ]
    elif spec == 'nav2':
        p = Atomic(L.unit('x') <= 5.0)
        scenario_spec = [
            Ev(in_l3),
            Alw(Implies(in_l3, BoundedEv((0, 3), p))),
        ]
    else:
        raise ValueError(f'Unknown spec: {spec}')

    stl_spec = make_unique(
        And(
            #  Alw(And(*control_spec)),
            *scenario_spec,
        )
    )

    prob.initialize_milp_formulation(stl_spec)

    return prob


def get_iso_rss(
    milp: gp.Model,
    N: int = 10,
    spec: str = 'iso1',
    optimize=False,
    delta=0.1,
) -> StlMilpProblem:
    prob = create_stl_milp_problem(
        milp,
        N=N,
        delta=delta,
        gamma_0=0.0,
        gamma_N=10.0,
        gamma_unit_length=0.005,
        use_binary_expansion=True,
        n_gamma_digits=10,  # 0.005 * 2^10 = 5.12
    )

    # `iso1a` represents a modified version of `iso1` with different parameter set
    if spec[-1] == 'a':
        parameter_set = 'a'
        scenario_id = int(spec[-2])
    else:
        parameter_set = 'normal'
        scenario_id = int(spec[-1])

    car = prob.create_system_model()

    ax = car.add_state('ax', 0, 1000)
    bx = car.add_state('bx', 0, 1000)
    ax_vel = car.add_state('ax_vel', 10, 25)
    bx_vel = car.add_state('bx_vel', 10, 25)
    ax_acc = car.add_state('ax_acc', -10, 5)
    bx_acc = car.add_state('bx_acc', -10, 5)
    ay = car.add_state('ay', 0, 10.5)
    by = car.add_state('by', 0, 10.5)
    ay_vel = car.add_state('ay_vel', -10, 10)
    by_vel = car.add_state('by_vel', -10, 10)
    ay_acc = car.add_state('ay_acc', -3, 3)
    by_acc = car.add_state('by_acc', -3, 3)
    rssDistance_a = car.add_state('rssDistance_a', 0, 1000)
    rssDistance_b = car.add_state('rssDistance_b', 0, 1000)
    lateralRssDistance_a = car.add_state('lateralRssDistance_a', 0, 1000)
    lateralRssDistance_b = car.add_state('lateralRssDistance_b', 0, 1000)

    car.set_initial_state(
        ax=50.0,
        bx=(0, 100),
    )

    # Dynamics
    car.add_dynamics('ax_acc', -3, 1, constant=True)
    car.add_dynamics('ay_acc', -1, 1, constant=True)
    car.add_dynamics('bx_acc', -3, 1, constant=True)
    car.add_dynamics('by_acc', -1, 1, constant=True)

    car.add_double_integrator_dynamics('ax', 'ax_vel', 'ax_acc')
    car.add_double_integrator_dynamics('ay', 'ay_vel', 'ay_acc')
    car.add_double_integrator_dynamics('bx', 'bx_vel', 'bx_acc')
    car.add_double_integrator_dynamics('by', 'by_vel', 'by_acc')

    # extra constraints to define rss distances
    from stlts import rss_distance as rss

    rss.setRssDistance(milp, rssDistance_a, rssDistance_b, ax_vel, bx_vel)
    rss.setLateralRssDistance(
        milp, lateralRssDistance_a, lateralRssDistance_b, ay_vel, by_vel
    )

    from stlts.rss_scenario import car_length, car_width, get_danger, get_scenario

    def no_collision(margin=0.0):
        overlap_x = And(
            Atomic(L.f(1.0, 'ax', -1.0, 'bx') <= car_length + margin),
            Atomic(L.f(1.0, 'bx', -1.0, 'ax') <= car_length + margin),
        )
        overlap_y = And(
            Atomic(L.f(1.0, 'ay', -1.0, 'by') <= car_width + margin),
            Atomic(L.f(1.0, 'by', -1.0, 'ay') <= car_width + margin),
        )
        return Alw(And(overlap_x, overlap_y).negation())

    def no_crossing(car: str, margin=0.1):
        return Or(
            Atomic(L.unit(f'{car}y') <= 3.5 - margin),
            And(
                Atomic(L.unit(f'{car}y') <= 7.0 - margin),
                Atomic(L.unit(f'{car}y') >= 3.5 + car_width + margin),
            ),
            Atomic(L.unit(f'{car}y') >= 7.0 + car_width + margin),
        )

    ensure_monotonicity_by_derivative(milp, ax_vel.tolist(), prob.N)
    ensure_monotonicity_by_derivative(milp, bx_vel.tolist(), prob.N)
    ensure_monotonicity_by_derivative(milp, ay_vel.tolist(), prob.N)
    ensure_monotonicity_by_derivative(milp, by_vel.tolist(), prob.N)

    if optimize:
        alpha = milp.addMVar(1, lb=0.0, ub=20.0, name='alpha').item()
        milp.setObjective(alpha, gp.GRB.MAXIMIZE)
    else:
        alpha = 0.1

    if parameter_set == 'normal':
        safe_distance_margin = 1.0
    elif parameter_set == 'a':
        safe_distance_margin = 5.0
    else:
        raise ValueError(f'Unknown parameter set: {parameter_set}')

    stl_spec = make_unique(
        And(
            get_danger(margin=1.0).negation(),
            get_scenario(scenario_id, alpha=alpha),
            no_collision(margin=safe_distance_margin),
            no_crossing('a'),
            no_crossing('b'),
        )
    )

    prob.initialize_milp_formulation(stl_spec)

    return prob
