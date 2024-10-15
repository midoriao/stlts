from __future__ import annotations

import gurobipy as gp
import numpy as np
from scipy import integrate

from .linear_expression import LinearExpression as L
from .linear_expression import LinearInequality
from .milp_helper import BinaryExpressionValue
from .trace import Trace
from .validator import ValidatorRules


class SystemModel:
    """
    The system model.
    """

    def __init__(
        self,
        model: gp.Model,
        N: int,
        n_gamma_digits: int | None = None,
        gamma_unit_length: float = 0.01,
    ):
        self.milp = model
        self.N = N
        self.state: dict[str, gp.MVar] = {}
        self.state_bounds: dict[str, tuple[float, float]] = {}
        self.interpolation_method: dict[str, tuple] = {}
        self.mode: dict[str, gp.MVar] = {}
        self.rules = ValidatorRules()
        self.gamma_length = self.milp.addMVar(self.N, name='system_gamma_length')

        # binary expansion option
        if n_gamma_digits is None:
            self.gamma_length_binary = None
        else:
            self.gamma_length_binary = [
                BinaryExpressionValue(
                    self.milp,
                    size=n_gamma_digits,
                    unit=gamma_unit_length,
                    name=f'gamma_length_binary[{i}]',
                )
                for i in range(N)
            ]
            self.milp.addConstrs(
                (
                    self.gamma_length[i].item() - self.gamma_length_binary[i].value
                    <= gamma_unit_length / 2
                    for i in range(N)
                ),
                name='gamma_diff_binary_ub',
            )
            self.milp.addConstrs(
                (
                    self.gamma_length[i] - self.gamma_length_binary[i].value >= 0
                    for i in range(N)
                ),
                name='gamma_diff_binary_lb',
            )

    def set_gamma_length(self, i: int, gamma_diff: gp.LinExpr | gp.Var) -> None:
        """
        Relate the length of `i`-th time step with some milp expression.
        `i` should be in [1, N].
        """
        self.milp.addConstr(self.gamma_length[i - 1] == gamma_diff)

    def add_state(
        self,
        name: str,
        lb: float,
        ub: float,
        qualitative: bool = False,
    ) -> gp.MVar:
        self.state[name] = self.milp.addMVar(
            self.N + 1,
            lb=lb,
            ub=ub,
            vtype=gp.GRB.CONTINUOUS if not qualitative else gp.GRB.INTEGER,
            name=name,
        )
        self.state_bounds[name] = (lb, ub)
        return self.state[name]

    def add_mode(
        self,
        name: str,
    ) -> gp.MVar:
        self.mode[name] = self.milp.addMVar(
            self.N + 1,
            vtype=gp.GRB.BINARY,
            name=name,
        )

        return self.mode[name]

    def add_dynamics(
        self,
        var: str,
        lb: float = 0.0,
        ub: float | None = None,
        mode: str | None = None,
        constant: bool = False,
    ):
        """
        Add dynamics of the variable.
        Currently only support linear dynamics.
        """
        if ub is None:
            ub = lb

        if constant:
            self.interpolation_method[var] = ('constant',)
        else:
            self.interpolation_method[var] = ('linear',)

        if mode is None:
            self.milp.addConstr(
                self.state[var][1:] - self.state[var][:-1] >= self.gamma_length * lb,
                name=f"dynamics_lb[mode=global][var={var}]",
            )
            self.milp.addConstr(
                self.state[var][1:] - self.state[var][:-1] <= self.gamma_length * ub,
                name=f"dynamics_ub[mode=global][var={var}]",
            )
            if not constant:
                self.rules.append(L.f(1.0, f'd_{var}') >= lb, tag='dynamics')
                self.rules.append(L.f(1.0, f'd_{var}') <= ub, tag='dynamics')
        else:
            self.milp.addConstrs(
                (
                    (self.mode[mode][i].item() == 1)
                    >> (
                        self.state[var][i + 1].item() - self.state[var][i].item()
                        >= self.gamma_length[i].item() * lb
                    )
                    for i in range(self.N)
                ),
                name=f"dynamics_lb[mode={mode}][var={var}]",
            )
            self.milp.addConstrs(
                (
                    (self.mode[mode][i].item() == 1)
                    >> (
                        self.state[var][i + 1].item() - self.state[var][i].item()
                        <= self.gamma_length[i].item() * ub
                    )
                    for i in range(self.N)
                ),
                name=f"dynamics_ub[mode={mode}][var={var}]",
            )

            if not constant:
                self.rules.append(
                    L.f(1.0, f'd_{var}') >= lb, lambda x: x[mode] == 1, tag='dynamics'
                )
                self.rules.append(
                    L.f(1.0, f'd_{var}') <= ub, lambda x: x[mode] == 1, tag='dynamics'
                )

    def add_double_integrator_dynamics(
        self,
        x: str,
        dx: str,
        ddx: str,
    ):
        assert self.gamma_length_binary is not None

        self.interpolation_method[dx] = ('linear',)
        self.interpolation_method[x] = ('integral', dx)

        self.milp.addConstr(self.state[ddx][-1].item() == self.state[ddx][-2].item())

        for i in range(self.N):
            self.milp.addConstr(
                self.state[dx][i + 1].item() - self.state[dx][i].item()
                == self.gamma_length_binary[i].multiply(
                    self.milp, self.state[ddx][i].item()
                )
            )
            self.milp.addConstr(
                self.state[x][i + 1].item() - self.state[x][i].item()
                == self.gamma_length_binary[i].multiply(
                    self.milp,
                    0.5 * (self.state[dx][i + 1].item() + self.state[dx][i].item()),
                )
            )

    def add_invariants(
        self,
        mode: str,
        *inequalities: LinearInequality,
    ):
        """
        Add invariants to the system.

        Args:
            mode: mode name
            inequalities: list of linear inequalities
        """
        for inequality_id, inequality in enumerate(inequalities):
            expr = inequality.to_gt_zero().expr.apply_to_variables(self.state)

            self.milp.addConstrs(
                (
                    (self.mode[mode][i].item() == 1) >> (expr[i].item() >= 0)
                    for i in range(self.N + 1)
                ),
                name=f"inv_backward[mode={mode}][{inequality_id}]",
            )
            self.milp.addConstrs(
                (
                    (self.mode[mode][i].item() == 1) >> (expr[i + 1].item() >= 0)
                    for i in range(self.N)
                ),
                name=f"inv_forward[mode={mode}][{inequality_id}]",
            )

            self.rules.append(inequality, lambda x: x[mode] == 1, tag='invariant')

    def add_jumps(
        self,
        two_modes: tuple[str, str],
        *guards: LinearInequality,
    ):
        """
        Add jump conditions between two modes.

        Args:
            two_modes: tuple of current mode and next mode
            guards: guard conditions, at the current mode
        """
        for inequality_id, inequality in enumerate(guards):
            expr = inequality.to_gt_zero().expr.apply_to_variables(self.state)[1:]

            # indicator[i] == 1 if and only if the current and next mode are active at time i
            indicator = self.milp.addVars(
                self.N,
                vtype=gp.GRB.BINARY,
                name=f"indicator[mode={two_modes}, inequality={inequality_id}]",
            )
            self.milp.addConstrs((
                indicator[i]
                == gp.and_(
                    self.mode[two_modes[0]][i].item(),
                    self.mode[two_modes[1]][i + 1].item(),
                )
                for i in range(self.N)
            ))

            self.milp.addConstrs(
                ((indicator[i] == 1) >> (expr[i].item() >= 0) for i in range(self.N)),
                name=f"jump[mode={two_modes[0]}->{two_modes[1]}][{inequality_id}]",
            )
            self.rules.append(
                inequality,
                lambda x, y: x[two_modes[1]] == 1 and y[two_modes[0]] == 1,
                tag='jump',
            )

    def set_initial_state(
        self,
        **kwargs: float | bool | tuple[float, float],
    ):
        for var, value in kwargs.items():
            if isinstance(value, float):
                self.milp.addConstr(
                    self.state[var][0].item() == value, name=f"initial_{var}"
                )
                self.rules.append(L.f(1.0, var) >= value, tag='initial')
                self.rules.append(L.f(1.0, var) <= value, tag='initial')
            elif isinstance(value, bool):
                self.milp.addConstr(
                    self.mode[var][0].item() == value, name=f"initial_{var}"
                )
                self.rules.append(L.f(1.0, var) >= value, tag='initial')
                self.rules.append(L.f(1.0, var) <= value, tag='initial')
            elif isinstance(value, tuple):
                self.milp.addRange(
                    self.state[var][0].item(), value[0], value[1], name=f"initial_{var}"
                )
                self.rules.append(L.f(1.0, var) >= value[0], tag='initial')
                self.rules.append(L.f(1.0, var) <= value[1], tag='initial')
            else:
                raise ValueError(f"Unknown type of initial state {var}: {value}")

    def validate(self, trace: Trace) -> bool:
        initial = self.rules.filter_by_tag('initial')
        initial.validate(trace.samples[0])

        dynamics = self.rules.filter_by_tag('dynamics')
        for sample in trace.samples_with_derivatives:
            dynamics.validate(sample)

        invariant = self.rules.filter_by_tag('invariant')
        for sample in trace.samples:
            invariant.validate(sample)

        jump = self.rules.filter_by_tag('jump')
        for sample, sample_before in zip(trace.samples[1:], trace.samples[:-1]):
            jump.validate(sample, sample_before)

        return True

    def get_trace(self, interpolation_interval: float | None = None) -> Trace:
        time = np.zeros(self.N + 1)
        time[1:] = np.cumsum(self.gamma_length.X)

        state = {}
        for var in self.state:
            state[var] = np.array(self.state[var].X)
        for var in self.mode:
            # regularize to exact 1.0 or 0.0
            raw = np.array(self.mode[var].X)
            state[var] = np.where(raw > 0.5, 1.0, 0.0)

        trace = Trace(time, state)

        if interpolation_interval is None:
            return trace

        # interpolation

        size = int((trace.time[-1] - trace.time[0]) / interpolation_interval) + 1

        constant_variables = [
            var
            for var, method in self.interpolation_method.items()
            if method[0] == 'constant'
        ]
        integral_variables = [
            var
            for var, method in self.interpolation_method.items()
            if method[0] == 'integral'
        ]
        linear_variables = [
            var
            for var in state
            if var not in constant_variables and var not in integral_variables
        ]  # default is linear

        trace_linear = trace.interpolate(
            np.linspace(trace.time[0], trace.time[-1], size),
            [*self.mode.keys(), *constant_variables],
            [*linear_variables, *integral_variables],
        )

        # compute interpolation for integral variables
        state_integral = {
            var: trace_linear[var]
            for var in trace.state
            if var not in integral_variables
        }
        time = trace_linear.time

        for var in state:
            if var not in integral_variables:
                state_integral[var] = trace_linear[var]

        for var in integral_variables:
            derivative_var = self.interpolation_method[var][1]
            state_integral[var] = (
                integrate.cumulative_trapezoid(
                    state_integral[derivative_var], time, initial=0
                )
                + trace[var][0]
            )

        return Trace(time, state_integral)
