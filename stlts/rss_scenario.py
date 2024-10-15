from __future__ import annotations

import typing

from .linear_expression import LinearExpression as L
from .stl import Alw, And, Atomic, Bottom, BoundedAlw, BoundedEv, Ev, Or, Top, Until

Real = typing.Any

car_width = 1.7
car_length = 4.7

extension = True


# longitudinal danger: long. car distance <= car length + RSS distance,


def danger_x(
    ax: Real, ax_vel: Real, bx: Real, bx_vel: Real, length_a: Real, length_b: Real
):
    formulas = [
        ((bx + -ax) + -(L.unit('rssDistance_a') + length_b) <= 0) @ 'danger_car_a_rear',
        ((ax + -bx) + -(L.unit('rssDistance_b') + length_a) <= 0) @ 'danger_car_b_rear',
    ]
    return And(*[Atomic(f) for f in formulas])


# //lateral danger: lat. car distance <= car width + RSS distance
# //1.6m = width of smart. Numbers taken from GA-RSS paper,
# //(except for ay_max (see paper in notes),
# //by_min, by_max (derived by trigonometry))
def danger_y(
    ay: Real, ay_vel: Real, by: Real, by_vel: Real, width_a: Real, width_b: Real
):
    formulas = [
        ((by + -ay) + -(L.unit('lateralRssDistance_a') + width_b) <= 0)
        @ 'danger_car_a_right',
        ((ay + -by) + -(L.unit('lateralRssDistance_b') + width_a) <= 0)
        @ 'danger_car_b_right',
    ]
    return And(*[Atomic(f) for f in formulas])


# //danger : RSS distance violated in both lat. and long. direction
# //e.g. if cars are driving next to eachother in different lanes,
# //there is no danger even when ax=bx if lateral safety distance is kept
def danger(
    ax: Real,
    ax_vel: Real,
    ay: Real,
    ay_vel: Real,
    bx: Real,
    bx_vel: Real,
    by: Real,
    by_vel: Real,
    length_a: Real,
    width_a: Real,
    length_b: Real,
    width_b: Real,
):
    return And(
        danger_x(ax, ax_vel, bx, bx_vel, length_a, length_b),
        danger_y(ay, ay_vel, by, by_vel, width_a, width_b),
    )


# //safety = not danger
def safe(
    ax: Real,
    ax_vel: Real,
    ay: Real,
    ay_vel: Real,
    bx: Real,
    bx_vel: Real,
    by: Real,
    by_vel: Real,
    length_a: Real,
    width_a: Real,
    length_b: Real,
    width_b: Real,
):
    return (
        danger(
            ax,
            ax_vel,
            ay,
            ay_vel,
            bx,
            bx_vel,
            by,
            by_vel,
            length_a,
            width_a,
            length_b,
            width_b,
        )
    ).negation()


# //----defining road structure----
#
# // i = 0, 1, 2 is a lane number
# // lane centers are at 1.75, 5.25, 8.75
def atLane(laneID: Real, i: int):
    if laneID.variables[0] == 'lane_a':
        car = 'a'
    elif laneID.variables[0] == 'lane_b':
        car = 'b'
    else:
        raise ValueError(f"Unknown laneID {laneID}")

    position = L.unit(f'{car}y')

    # 0--3.5, 3.5--7, 7--10.5
    return And(
        Atomic(position <= 3.5 * (i + 1) + car_width),
        Atomic(position >= 3.5 * i),
    )


# //OLD: the point masses ax and bx are at least one car length apart
# //NEW: b is slightly ahead of a (for use in cut-in/out scenario)
def aheadOf(ax: Real, bx: Real):
    return Atomic(ax + -bx <= 0)


def strictAheadOf(ax: Real, bx: Real, b_length: Real):
    return Atomic(ax + b_length + -bx <= 0)


def safe_dur(
    ax: Real,
    ax_vel: Real,
    ay: Real,
    ay_vel: Real,
    bx: Real,
    bx_vel: Real,
    by: Real,
    by_vel: Real,
    length_a: Real,
    width_a: Real,
    length_b: Real,
    width_b: Real,
    durness: float = 0.6,
):
    safety_formula = safe(
        ax,
        ax_vel,
        ay,
        ay_vel,
        bx,
        bx_vel,
        by,
        by_vel,
        length_a,
        width_a,
        length_b,
        width_b,
    )
    if durness == 0.0:
        return safety_formula
    else:
        return BoundedAlw(
            (0.0, durness),
            safety_formula,
        )


def danger_dur(
    ax: Real,
    ax_vel: Real,
    ay: Real,
    ay_vel: Real,
    bx: Real,
    bx_vel: Real,
    by: Real,
    by_vel: Real,
    length_a: Real,
    width_a: Real,
    length_b: Real,
    width_b: Real,
    durness: float = 0.6,
):
    danger_formula = danger(
        ax,
        ax_vel,
        ay,
        ay_vel,
        bx,
        bx_vel,
        by,
        by_vel,
        length_a,
        width_a,
        length_b,
        width_b,
    )
    if durness == 0.0:
        return danger_formula
    else:
        return BoundedAlw(
            (0.0, durness),
            danger_formula,
        )


def adjacentLanes(lane1: Real, lane2: Real):
    if isinstance(lane1, float) and isinstance(lane2, float):
        if abs(lane1 - lane2) == 1:
            return Top()
        else:
            return Bottom()
    else:
        return NotImplementedError()
    return Or(
        And(
            Atomic(lane1 + -lane2 >= -1, qualitative=True),
            Atomic(lane1 + -lane2 <= -1, qualitative=True),
        ),
        And(
            Atomic(lane1 + -lane2 >= 1, qualitative=True),
            Atomic(lane1 + -lane2 <= 1, qualitative=True),
        ),
    )


def sameLane(lane1: Real, lane2: Real):
    if isinstance(lane1, float) and isinstance(lane2, float):
        if lane1 - lane2 == 0:
            return Top()
        else:
            return Bottom()
    else:
        return NotImplementedError()
    return And(
        Atomic(lane1 + -lane2 >= 0, qualitative=True),
        Atomic(lane1 + -lane2 <= 0, qualitative=True),
    )


def enteringLane(lane_a: Real, targetLane: Real):
    return And((atLane(lane_a, targetLane)).negation(), Ev(atLane(lane_a, targetLane)))


def leavingLane(lane_a: Real, startLane: Real):
    return And(atLane(lane_a, startLane), Ev(atLane(lane_a, startLane).negation()))


def accel(
    x: Real,
    x_vel: Real,
    x_acc: Real,
    y: Real,
    refX: Real,
    refX_vel: Real,
    laneID: Real,
    startLane: Real,
    rate: float = 0.1,
):
    if extension:
        f1 = Or(Atomic(x_acc >= rate), Atomic(x_vel + -refX_vel >= rate))
    else:
        f1 = Atomic(x_vel + -refX_vel >= rate)
    f2 = atLane(laneID, startLane)
    return And(f1, f2)


def decel(
    x: Real,
    x_vel: Real,
    x_acc: Real,
    y: Real,
    refX: Real,
    refX_vel: Real,
    laneID: Real,
    startLane: Real,
    rate: float = 0.1,
):
    if extension:
        f1 = Or(Atomic(x_acc <= -rate), Atomic(x_vel + -refX_vel <= -rate))
    else:
        f1 = Atomic(x_vel + -refX_vel <= -rate)
    f2 = atLane(laneID, startLane)
    return And(f1, f2)


# // SV and POV b:  b is behind a and they are in the same lane until there is danger (because b is accelerating)
def scenario3(
    ax: Real,
    ax_vel: Real,
    ax_acc: Real,
    ay: Real,
    ay_vel: Real,
    bx: Real,
    bx_vel: Real,
    bx_acc: Real,
    by: Real,
    by_vel: Real,
    lane_a: Real,
    initialLane_a: Real,
    lane_b: Real,
    initialLane_b: Real,
    length_a: Real,
    width_a: Real,
    length_b: Real,
    width_b: Real,
):
    if extension:
        f1 = Or(aheadOf(bx, ax), strictAheadOf(bx, ax, length_a))
    else:
        f1 = strictAheadOf(bx, ax, length_a)
    # // probably needed without restrictions on ax
    f2 = safe_dur(
        bx,
        bx_vel,
        by,
        by_vel,
        ax,
        ax_vel,
        ay,
        ay_vel,
        length_b,
        width_b,
        length_a,
        width_a,
    )
    f3 = Or(
        adjacentLanes(initialLane_a, initialLane_b),
        sameLane(initialLane_a, initialLane_b),
    )
    f4 = Until(
        And(
            atLane(lane_a, initialLane_a),
            accel(bx, bx_vel, bx_acc, by, ax, ax_vel, lane_b, initialLane_b),
        ),
        (
            danger_dur(
                ax,
                ax_vel,
                ay,
                ay_vel,
                bx,
                bx_vel,
                by,
                by_vel,
                length_a,
                width_a,
                length_b,
                width_b,
            )
        ),
    )  # // implicitly because of danger: their distance will decrease
    return And(f1, f2, f3, f4)


# // SV and POV b:  a is behind b and they are in the same lane until there is danger (because b is braking)
def scenario4(
    ax: Real,
    ax_vel: Real,
    ax_acc: Real,
    ay: Real,
    ay_vel: Real,
    bx: Real,
    bx_vel: Real,
    bx_acc: Real,
    by: Real,
    by_vel: Real,
    lane_a: Real,
    initialLane_a: Real,
    lane_b: Real,
    initialLane_b: Real,
    length_a: Real,
    width_a: Real,
    length_b: Real,
    width_b: Real,
):
    if extension:
        f1 = Or(aheadOf(ax, bx), strictAheadOf(ax, bx, length_b))
    else:
        f1 = strictAheadOf(ax, bx, length_b)
    f2 = safe_dur(
        ax,
        ax_vel,
        ay,
        ay_vel,
        bx,
        bx_vel,
        by,
        by_vel,
        length_a,
        width_a,
        length_b,
        width_b,
    )
    f3 = Or(
        adjacentLanes(initialLane_a, initialLane_b),
        sameLane(initialLane_a, initialLane_b),
    )
    f4 = Until(
        And(
            atLane(lane_a, initialLane_a),
            decel(bx, bx_vel, bx_acc, by, ax, ax_vel, lane_b, initialLane_b),
        ),
        (
            danger_dur(
                ax,
                ax_vel,
                ay,
                ay_vel,
                bx,
                bx_vel,
                by,
                by_vel,
                length_a,
                width_a,
                length_b,
                width_b,
            )
        ),
    )  # // implicitly because of danger: their distance will decrease
    return And(f1, f2, f3, f4)


# // SV a and POV b: a is in  lane j, b in lane i, and a wants to merge in front of b, but they get too close to each other
# // because of the arrow in the table: set b to acceleration?
def scenario7(
    ax: Real,
    ax_vel: Real,
    ax_acc: Real,
    ay: Real,
    ay_vel: Real,
    bx: Real,
    bx_vel: Real,
    bx_acc: Real,
    by: Real,
    by_vel: Real,
    lane_a: Real,
    initialLane_a: Real,
    lane_b: Real,
    initialLane_b: Real,
    length_a: Real,
    width_a: Real,
    length_b: Real,
    width_b: Real,
):
    if extension:
        f1 = Or(aheadOf(bx, ax), strictAheadOf(bx, ax, length_a))
    else:
        f1 = strictAheadOf(bx, ax, length_a)
    # // probably needed without restrictions on ax

    f2 = safe_dur(
        ax,
        ax_vel,
        ay,
        ay_vel,
        bx,
        bx_vel,
        by,
        by_vel,
        length_a,
        width_a,
        length_b,
        width_b,
    )
    f3 = enteringLane(lane_a, initialLane_b)
    f4 = Until(
        (accel(bx, bx_vel, bx_acc, by, ax, ax_vel, lane_b, initialLane_b)),
        danger_dur(
            ax,
            ax_vel,
            ay,
            ay_vel,
            bx,
            bx_vel,
            by,
            by_vel,
            length_a,
            width_a,
            length_b,
            width_b,
        ),
    )  # // implicitly because of danger: their distance will decrease
    return And(f1, f2, f3, f4)


# // SV a and POV b: a is behind b, b is decelerating while a is changing lanes
def scenario8(
    ax: Real,
    ax_vel: Real,
    ax_acc: Real,
    ay: Real,
    ay_vel: Real,
    bx: Real,
    bx_vel: Real,
    bx_acc: Real,
    by: Real,
    by_vel: Real,
    lane_a: Real,
    initialLane_a: Real,
    lane_b: Real,
    initialLane_b: Real,
    length_a: Real,
    width_a: Real,
    length_b: Real,
    width_b: Real,
    alpha=1.0,
):
    if extension:
        f1 = Or(aheadOf(ax, bx), strictAheadOf(ax, bx, length_b))
    else:
        f1 = strictAheadOf(ax, bx, length_b)

    f2 = safe_dur(
        ax,
        ax_vel,
        ay,
        ay_vel,
        bx,
        bx_vel,
        by,
        by_vel,
        length_a,
        width_a,
        length_b,
        width_b,
    )
    if initialLane_a != initialLane_b:
        raise ValueError("initialLane_a != initialLane_b")
    # f3 = sameLane(initialLane_a, initialLane_b)
    f4 = atLane(lane_a, initialLane_a)
    f5 = leavingLane(lane_a, initialLane_a)
    f6 = Until(
        decel(
            bx,
            bx_vel,
            bx_acc,
            by,
            ax,
            ax_vel,
            lane_b,
            initialLane_b,
            rate=alpha,
        ),
        (
            danger_dur(
                ax,
                ax_vel,
                ay,
                ay_vel,
                bx,
                bx_vel,
                by,
                by_vel,
                length_a,
                width_a,
                length_b,
                width_b,
            )
        ),
    )  #    // implicitly because of danger: their distance will decrease
    return And(f1, f2, f4, f5, f6)


# //Predicates for cut-in/out


# //cutIn - TODO: make durness-friendly
def cutIn(
    ax: Real,
    ax_vel: Real,
    ay: Real,
    ay_vel: Real,
    bx: Real,
    bx_vel: Real,
    by: Real,
    by_vel: Real,
    lane_a: Real,
    initialLane_a: Real,
    lane_b: Real,
    length_a: Real,
    width_a: Real,
    length_b: Real,
    width_b: Real,
):
    if extension:
        f = []
    else:
        f = [strictAheadOf(ax, bx, length_b)]
    ev_content = And(
        danger_dur(
            ax,
            ax_vel,
            ay,
            ay_vel,
            bx,
            bx_vel,
            by,
            by_vel,
            length_a,
            width_a,
            length_b,
            width_b,
        ),
        BoundedEv(
            (0.0, 0.6),
            And(
                *f,
                atLane(lane_b, initialLane_a),
            ),
        ),
    )
    return And(
        atLane(lane_a, initialLane_a),
        (atLane(lane_b, initialLane_a)).negation(),
        Ev(ev_content),
    )


# //three vehicle cut-out
# //def cutOut_sameLane(ax: Real,ay: Real,bx: Real,by: Real,cx: Real,cy: Real, lane_a: Real, lane_b: Real, lane_c: Real, length_a: Real, width_a: Real, length_b: Real, width_b: Real, length_c: Real, width_c: Real, initialLane_a: Real) : Bool =
# //  atLane(lane_a, lane_c) /\  atLane(lane_b,lane_c) /\ atLane(lane_c,initialLane_a)
# //  /\ aheadOf(ax,bx) /\ aheadOf(bx,cx)
# //  /\ ((atLane(lane_a,initialLane_a) /\ atLane(lane_c,initialLane_a)) U (~(atLane(lane_b,initialLane_a)) /\ danger(ax,ay,cx,cy, length_a, width_a, length_c, width_c))) // a and c both keep lane until b has left lane and danger with c occurs
# //  /\ (leavingLane(lane_b,initialLane_a)) // b leaves lane eventually


# //two vehicle cut-out does not have an extension
def cutOut(
    ax: Real,
    ax_vel: Real,
    ay: Real,
    ay_vel: Real,
    bx: Real,
    bx_vel: Real,
    by: Real,
    by_vel: Real,
    lane_a: Real,
    initialLane_a: Real,
    lane_b: Real,
    length_a: Real,
    width_a: Real,
    length_b: Real,
    width_b: Real,
):
    return And(
        atLane(lane_a, initialLane_a),
        leavingLane(lane_b, initialLane_a),
        Ev(
            And(
                danger_dur(
                    ax,
                    ax_vel,
                    ay,
                    ay_vel,
                    bx,
                    bx_vel,
                    by,
                    by_vel,
                    length_a,
                    width_a,
                    length_b,
                    width_b,
                ),
                BoundedEv((0.0, 0.6), atLane(lane_b, initialLane_a).negation()),
            )
        ),
    )


#  // this is possibly already an extension: no extension would be to include an aheadOf


def scenario1(
    ax: Real,
    ax_vel: Real,
    ay: Real,
    ay_vel: Real,
    bx: Real,
    bx_vel: Real,
    by: Real,
    by_vel: Real,
    lane_a: Real,
    initialLane_a: Real,
    lane_b: Real,
    length_a: Real,
    width_a: Real,
    length_b: Real,
    width_b: Real,
):
    f1 = safe_dur(
        ax,
        ax_vel,
        ay,
        ay_vel,
        bx,
        bx_vel,
        by,
        by_vel,
        length_a,
        width_a,
        length_b,
        width_b,
    )
    f2 = Until(
        atLane(lane_a, initialLane_a),
        danger_dur(
            ax,
            ax_vel,
            ay,
            ay_vel,
            bx,
            bx_vel,
            by,
            by_vel,
            length_a,
            width_a,
            length_b,
            width_b,
        ),
    )  # SV lane keep until danger

    f3 = cutIn(
        ax,
        ax_vel,
        ay,
        ay_vel,
        bx,
        bx_vel,
        by,
        by_vel,
        lane_a,
        initialLane_a,
        lane_b,
        length_a,
        width_a,
        length_b,
        width_b,
    )  # POV cut-in
    return And(f1, f2, f3)


# //--scenario 5: SV a changes lane i -> j, POV b cuts in by changing lane -> i
def scenario5(
    ax: Real,
    ax_vel: Real,
    ay: Real,
    ay_vel: Real,
    bx: Real,
    bx_vel: Real,
    by: Real,
    by_vel: Real,
    lane_a: Real,
    initialLane_a: Real,
    lane_b: Real,
    length_a: Real,
    width_a: Real,
    length_b: Real,
    width_b: Real,
):
    f1 = safe_dur(
        ax,
        ax_vel,
        ay,
        ay_vel,
        bx,
        bx_vel,
        by,
        by_vel,
        length_a,
        width_a,
        length_b,
        width_b,
    )
    f2 = leavingLane(lane_a, initialLane_a)
    f3 = cutIn(
        ax,
        ax_vel,
        ay,
        ay_vel,
        bx,
        bx_vel,
        by,
        by_vel,
        lane_a,
        initialLane_a,
        lane_b,
        length_a,
        width_a,
        length_b,
        width_b,
    )
    return And(f1, f2, f3)


# //--scenario 6: SV a changes lane i -> j, POV b changes lane i -> j dangerously
def scenario6(
    ax: Real,
    ax_vel: Real,
    ay: Real,
    ay_vel: Real,
    bx: Real,
    bx_vel: Real,
    by: Real,
    by_vel: Real,
    lane_a: Real,
    initialLane_a: Real,
    lane_b: Real,
    length_a: Real,
    width_a: Real,
    length_b: Real,
    width_b: Real,
):
    f1 = safe_dur(
        ax,
        ax_vel,
        ay,
        ay_vel,
        bx,
        bx_vel,
        by,
        by_vel,
        length_a,
        width_a,
        length_b,
        width_b,
    )
    f2 = leavingLane(lane_a, initialLane_a)
    f3 = cutOut(
        ax,
        ax_vel,
        ay,
        ay_vel,
        bx,
        bx_vel,
        by,
        by_vel,
        lane_a,
        initialLane_a,
        lane_b,
        length_a,
        width_a,
        length_b,
        width_b,
    )
    return And(f1, f2, f3)


def get_scenario(id: int, alpha=0.1):
    if id == 1:
        return scenario1(
            L.unit('ax'),
            L.unit('ax_vel'),
            #    L.unit('ax_acc'),
            L.unit('ay'),
            L.unit('ay_vel'),
            #    L.unit('ay_acc'),
            L.unit('bx'),
            L.unit('bx_vel'),
            #    L.unit('bx_acc'),
            L.unit('by'),
            L.unit('by_vel'),
            #    L.unit('by_acc'),
            lane_a=L.unit('lane_a'),
            initialLane_a=2.0,
            lane_b=L.unit('lane_b'),
            #   initialLane_b=initial_lane_b,
            length_a=car_length,
            width_a=car_width,
            length_b=car_length,
            width_b=car_width,
        )
    elif id == 3:
        return scenario3(
            L.unit('ax'),
            L.unit('ax_vel'),
            L.unit('ax_acc'),
            L.unit('ay'),
            L.unit('ay_vel'),
            #    L.unit('ay_acc'),
            L.unit('bx'),
            L.unit('bx_vel'),
            L.unit('bx_acc'),
            L.unit('by'),
            L.unit('by_vel'),
            #    L.unit('by_acc'),
            lane_a=L.unit('lane_a'),
            initialLane_a=1.0,
            lane_b=L.unit('lane_b'),
            initialLane_b=1.0,
            length_a=car_length,
            width_a=car_width,
            length_b=car_length,
            width_b=car_width,
        )
    elif id == 4:
        return scenario4(
            L.unit('ax'),
            L.unit('ax_vel'),
            L.unit('ax_acc'),
            L.unit('ay'),
            L.unit('ay_vel'),
            #    L.unit('ay_acc'),
            L.unit('bx'),
            L.unit('bx_vel'),
            L.unit('bx_acc'),
            L.unit('by'),
            L.unit('by_vel'),
            #    L.unit('by_acc'),
            lane_a=L.unit('lane_a'),
            initialLane_a=1.0,
            lane_b=L.unit('lane_b'),
            initialLane_b=1.0,
            length_a=car_length,
            width_a=car_width,
            length_b=car_length,
            width_b=car_width,
        )
    elif id == 5:
        return scenario5(
            L.unit('ax'),
            L.unit('ax_vel'),
            #    L.unit('ax_acc'),
            L.unit('ay'),
            L.unit('ay_vel'),
            #    L.unit('ay_acc'),
            L.unit('bx'),
            L.unit('bx_vel'),
            #   L.unit('bx_acc'),
            L.unit('by'),
            L.unit('by_vel'),
            #    L.unit('by_acc'),
            lane_a=L.unit('lane_a'),
            initialLane_a=1.0,
            lane_b=L.unit('lane_b'),
            length_a=car_length,
            width_a=car_width,
            length_b=car_length,
            width_b=car_width,
        )
    elif id == 6:
        return scenario6(
            L.unit('ax'),
            L.unit('ax_vel'),
            #    L.unit('ax_acc'),
            L.unit('ay'),
            L.unit('ay_vel'),
            #    L.unit('ay_acc'),
            L.unit('bx'),
            L.unit('bx_vel'),
            #   L.unit('bx_acc'),
            L.unit('by'),
            L.unit('by_vel'),
            #    L.unit('by_acc'),
            lane_a=L.unit('lane_a'),
            initialLane_a=1.0,
            lane_b=L.unit('lane_b'),
            length_a=car_length,
            width_a=car_width,
            length_b=car_length,
            width_b=car_width,
        )
    elif id == 7:
        return scenario7(
            L.unit('ax'),
            L.unit('ax_vel'),
            L.unit('ax_acc'),
            L.unit('ay'),
            L.unit('ay_vel'),
            #    L.unit('ay_acc'),
            L.unit('bx'),
            L.unit('bx_vel'),
            L.unit('bx_acc'),
            L.unit('by'),
            L.unit('by_vel'),
            #    L.unit('by_acc'),
            lane_a=L.unit('lane_a'),
            initialLane_a=1.0,
            lane_b=L.unit('lane_b'),
            initialLane_b=2.0,
            length_a=car_length,
            width_a=car_width,
            length_b=car_length,
            width_b=car_width,
        )
    elif id == 8:
        return scenario8(
            L.unit('ax'),
            L.unit('ax_vel'),
            L.unit('ax_acc'),
            L.unit('ay'),
            L.unit('ay_vel'),
            #    L.unit('ay_acc'),
            L.unit('bx'),
            L.unit('bx_vel'),
            L.unit('bx_acc'),
            L.unit('by'),
            L.unit('by_vel'),
            #    L.unit('by_acc'),
            lane_a=L.unit('lane_a'),
            initialLane_a=1.0,
            lane_b=L.unit('lane_b'),
            initialLane_b=1.0,
            length_a=car_length,
            width_a=car_width,
            length_b=car_length,
            width_b=car_width,
            alpha=alpha,
        )
    else:
        raise ValueError(f"Unknown scenario id {id}")


def get_danger(margin=0.0):
    return danger(
        L.unit('ax'),
        L.unit('ax_vel'),
        L.unit('ay'),
        L.unit('ay_vel'),
        L.unit('bx'),
        L.unit('bx_vel'),
        L.unit('by'),
        L.unit('by_vel'),
        length_a=car_length - margin,
        width_a=car_width - margin,
        length_b=car_length - margin,
        width_b=car_width - margin,
    )
