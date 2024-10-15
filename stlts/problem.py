from __future__ import annotations

import dataclasses

import gurobipy as gp
import numpy as np

from stlts import main_constraints
from stlts.main_constraints import MilpVariables

from .stl import StlFormula
from .system_model import SystemModel
from .trace import Trace


@dataclasses.dataclass(frozen=True)
class StlMilpProblemConfig:
    N: int
    delta: float
    gamma_0: float
    gamma_N: float
    gamma_unit_length: float
    use_binary_expansion: bool
    n_gamma_digits: int | None


def create_stl_milp_problem(
    milp: gp.Model,
    N: int = 2,
    delta: float = 0.01,
    gamma_0: float = 0.0,
    gamma_N: float = 1.0,
    gamma_unit_length: float = 0.01,
    use_binary_expansion: bool = False,
    n_gamma_digits: int | None = None,
) -> StlMilpProblem:
    config = StlMilpProblemConfig(
        N,
        delta,
        gamma_0,
        gamma_N,
        gamma_unit_length,
        use_binary_expansion,
        n_gamma_digits,
    )
    return StlMilpProblem(milp, config)


class StlMilpProblem:
    milp: gp.Model
    config: StlMilpProblemConfig
    milp_variables: MilpVariables
    system_model: SystemModel
    stl_spec: StlFormula

    def __init__(
        self,
        model: gp.Model,
        config: StlMilpProblemConfig,
    ):
        self.milp = model
        self.config = config

    def initialize_milp_formulation(
        self,
        spec: StlFormula,
        ignore_system_model: bool = False,
        system_states: dict[str, gp.MVar] | None = None,
    ) -> MilpVariables:
        self.stl_spec = spec

        self.milp_variables = v = main_constraints.make_milp_variables(
            self.milp,
            spec=spec,
            N=self.config.N,
            gamma_0=self.config.gamma_0,
            gamma_N=self.config.gamma_N,
            gamma_unit_length=self.config.gamma_unit_length,
        )

        if not ignore_system_model:
            for i in range(1, self.config.N + 1):
                self.system_model.set_gamma_length(i, v.gamma[i] - v.gamma[i - 1])
            system_states = self.system_model.state
        assert system_states is not None

        main_constraints.add_timeline_constraints(
            self.milp,
            v,
            self.config.gamma_0,
            self.config.gamma_N,
            self.config.gamma_unit_length,
        )
        main_constraints.add_full_stl_constraints(self.milp, v, spec)
        main_constraints.add_predicate_constraints(
            self.milp, v, system_states, self.config.delta
        )
        return v

    @property
    def delta(self) -> float:
        return self.config.delta

    @property
    def N(self) -> int:
        return self.config.N

    @property
    def state(self) -> dict[str, gp.MVar]:
        return self.system_model.state

    def search_satisfaction(self) -> bool:
        self.milp.optimize()
        return self.milp.SolCount > 0

    @property
    def has_solution(self) -> bool:
        return self.milp.SolCount > 0

    def get_gamma_result(self) -> np.ndarray:
        return np.array(self.milp_variables.get_values()['gamma'])

    def get_trace_result(self, interpolation=False) -> Trace:
        if interpolation:
            return self.system_model.get_trace(0.1)
        else:
            return self.system_model.get_trace()

    def create_system_model(self) -> SystemModel:
        if not self.config.use_binary_expansion:
            self.system_model = SystemModel(self.milp, self.N)
        else:
            if self.config.n_gamma_digits is None:
                # By default, we demand that the largest possible gamma covers the double of the horizon.
                length_in_units = (
                    self.config.gamma_N - self.config.gamma_0
                ) / self.config.gamma_unit_length
                n_gamma_digits = 1 + int(
                    np.ceil(np.log2(length_in_units / self.config.N))
                )
            else:
                n_gamma_digits = self.config.n_gamma_digits
            self.system_model = SystemModel(
                self.milp, self.N, n_gamma_digits, self.config.gamma_unit_length
            )
        return self.system_model
