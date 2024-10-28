from __future__ import annotations

import gurobipy as gp


def reactionTime():
    return 0.6


def amax():
    return 5


def bmin():
    return 6


def bmax():
    return 10


def amax_y():
    return 1.5


def bmin_y():
    return 1.5


def add_square(milp: gp.Model, v: gp.MVar):
    assert v.ndim == 1
    sq = milp.addMVar(shape=v.shape)
    for i in range(v.size):
        milp.addGenConstrPoly(
            v[i].item(),
            sq[i].item(),
            (1, 0, 0),
            options='FuncPieces=-1 FuncPieceError=0.01',
        )
    return sq


def preRssDistance(
    milp: gp.Model,
    rear_vel: gp.MVar,
    rear_vel_sq: gp.MVar,
    front_vel_sq: gp.MVar,
    name: str,
):
    preRssDistance = milp.addMVar(
        shape=rear_vel.shape, lb=-gp.GRB.INFINITY, name=f'preRssDistance_{name}'
    )
    milp.addConstr(
        preRssDistance
        == (
            reactionTime() * rear_vel
            + 0.5 * amax() * reactionTime() ** 2
            + (
                rear_vel_sq
                + 2 * amax() * reactionTime() * rear_vel
                + (amax() * reactionTime()) ** 2
            )
            / (2 * bmin())
            - (front_vel_sq) / (2 * bmax())
        )
    )
    return preRssDistance


def preLateralRssDistance(
    milp: gp.Model,
    left_vel: gp.MVar,
    left_vel_sq: gp.MVar,
    right_vel: gp.MVar,
    right_vel_sq: gp.MVar,
    name: str,
):
    # left is bigger
    preLateralRssDistance = milp.addMVar(
        shape=left_vel.shape, lb=-gp.GRB.INFINITY, name=f'preLateralRssDistance_{name}'
    )

    milp.addConstr(
        preLateralRssDistance
        == (
            (left_vel - right_vel) * reactionTime()
            + amax_y() * reactionTime() ** 2
            + (
                left_vel_sq
                + 2 * reactionTime() * amax_y() * left_vel
                + (reactionTime() * amax_y()) ** 2
                + right_vel_sq
                - 2 * reactionTime() * amax_y() * right_vel
                + (reactionTime() * amax_y()) ** 2
            )
            / (2 * bmin_y())
        )
    )
    return preLateralRssDistance


def setRssDistance(
    milp: gp.Model,
    a_var: gp.MVar,
    b_var: gp.MVar,
    a_vel: gp.MVar,
    b_vel: gp.MVar,
):
    # Precompute squares
    a_vel_sq = add_square(milp, a_vel)
    b_vel_sq = add_square(milp, b_vel)

    # 1) car a is rear
    preRssDistance_a = preRssDistance(
        milp,
        rear_vel=a_vel,
        rear_vel_sq=a_vel_sq,
        front_vel_sq=b_vel_sq,
        name='car_a',
    )
    milp.addConstrs(
        a_var[i].item() == gp.max_(preRssDistance_a[i].item(), constant=0.0)
        for i in range(a_var.shape[0])
    )

    # 2) car b is rear
    preRssDistance_b = preRssDistance(
        milp,
        rear_vel=b_vel,
        rear_vel_sq=b_vel_sq,
        front_vel_sq=a_vel_sq,
        name='car_b',
    )
    milp.addConstrs(
        b_var[i].item() == gp.max_(preRssDistance_b[i].item(), constant=0.0)
        for i in range(b_var.shape[0])
    )


def setLateralRssDistance(
    milp: gp.Model,
    a_var: gp.MVar,
    b_var: gp.MVar,
    a_yvel: gp.MVar,
    b_yvel: gp.MVar,
):
    # Precompute squares
    a_vel_sq = add_square(milp, a_yvel)
    b_vel_sq = add_square(milp, b_yvel)

    # 1) car a is right
    preLateralRssDistance_a = preLateralRssDistance(
        milp,
        right_vel=a_yvel,
        left_vel=b_yvel,
        right_vel_sq=a_vel_sq,
        left_vel_sq=b_vel_sq,
        name='car_a',
    )
    milp.addConstrs(
        a_var[i].item() == gp.max_(preLateralRssDistance_a[i].item(), constant=0.0)
        for i in range(a_var.shape[0])
    )

    # 1) car b is right
    preLateralRssDistance_b = preLateralRssDistance(
        milp,
        right_vel=b_yvel,
        left_vel=a_yvel,
        right_vel_sq=b_vel_sq,
        left_vel_sq=a_vel_sq,
        name='car_b',
    )
    milp.addConstrs(
        b_var[i].item() == gp.max_(preLateralRssDistance_b[i].item(), constant=0.0)
        for i in range(b_var.shape[0])
    )
