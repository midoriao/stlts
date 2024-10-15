from __future__ import annotations

import numpy as np
import pandas as pd
import scipy


class Trace:
    state: dict[str, np.ndarray]
    time: np.ndarray

    def __init__(self, time, state: dict[str, np.ndarray]):
        self.state = state
        self.time = time

    @classmethod
    def from_df(cls, df: pd.DataFrame) -> Trace:
        return cls(
            df['t'].to_numpy(),
            {var: df[var].to_numpy() for var in df.columns if var != 't'},
        )

    def __getitem__(self, item):
        return self.state[item]

    @property
    def df(self) -> pd.DataFrame:
        return pd.DataFrame(self.samples)

    def at(self, time: float, var: str) -> float:
        return np.interp(time, self.time, self.state[var]).item()

    def _derivative(self, var: str, i: int) -> float:
        assert 0 <= i < len(self.time) - 1
        return (self.state[var][i + 1] - self.state[var][i]) / (
            self.time[i + 1] - self.time[i]
        ).item()

    @property
    def samples(self) -> list[dict[str, float | bool]]:
        return [
            {
                't': self.time[i],
                **{var: self.state[var][i].item() for var in self.state},
            }
            for i in range(len(self.time))
        ]

    @property
    def samples_with_derivatives(self) -> list[dict[str, float | bool]]:
        return [
            {
                't': self.time[i],
                **{var: self.state[var][i].item() for var in self.state},
                **{
                    f'd_{var}': self._derivative(var, i)
                    for var in self.state
                    if self.state[var].dtype == np.float64
                },
            }
            for i in range(len(self.time) - 1)
            if self.time[i] != self.time[i + 1]  # avoid zero division
        ]

    def interpolate(
        self,
        steps: np.ndarray | None = None,
        constant_states: list[str] | None = None,
        linear_states: list[str] | None = None,
    ) -> Trace:
        if steps is None:
            steps = np.arange(self.time[0], self.time[-1], 0.1)
        if constant_states is None:
            constant_states = []
        if linear_states is None:
            linear_states = [
                state for state in self.state.keys() if state not in constant_states
            ]

        # merge and remove duplicates (tiny difference is considered as duplicate)
        time = np.concatenate((steps, self.time))
        time.sort(kind='mergesort')
        time = time[np.diff(time, append=np.inf) >= 1e-9]

        state = {}
        for var in linear_states:
            state[var] = np.interp(time, self.time, self.state[var])
        for var in constant_states:
            state[var] = scipy.interpolate.interp1d(
                self.time, self.state[var], kind='previous'
            )(time)
        return Trace(time, state)
