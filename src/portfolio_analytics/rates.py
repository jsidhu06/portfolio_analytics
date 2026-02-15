"""Interest-rate and discount-curve utilities."""

from dataclasses import dataclass
from datetime import datetime
from collections.abc import Sequence
import numpy as np
from .utils import get_year_deltas


@dataclass(frozen=True, slots=True)
class DiscountCurve:
    """Deterministic discount curve with log-linear interpolation.

    times are year fractions and must be strictly increasing.
    dfs are discount factors in (0, 1], typically with df(0)=1.
    """

    name: str
    times: np.ndarray
    dfs: np.ndarray
    flat_rate: float | None = None

    def __post_init__(self) -> None:
        t = np.asarray(self.times, dtype=float)
        df = np.asarray(self.dfs, dtype=float)
        if t.ndim != 1 or df.ndim != 1 or t.shape != df.shape:
            raise ValueError("times and dfs must be 1D arrays of the same length")
        if np.any(np.diff(t) <= 0.0):
            raise ValueError("times must be strictly increasing")
        if np.any(df <= 0.0) or np.any(df > 1.0 + 1e-12):
            raise ValueError("discount factors must be in (0, 1]")
        if self.flat_rate is not None and not np.isfinite(float(self.flat_rate)):
            raise ValueError("flat_rate must be finite when provided")
        object.__setattr__(self, "times", t)
        object.__setattr__(self, "dfs", df)

    @classmethod
    def flat(
        cls,
        name: str,
        rate: float,
        end_time: float,
        steps: int = 1,
    ) -> "DiscountCurve":
        if end_time <= 0.0:
            raise ValueError("end_time must be positive")
        if steps < 1:
            raise ValueError("steps must be >= 1")
        times = np.linspace(0.0, float(end_time), int(steps) + 1)
        dfs = np.exp(-float(rate) * times)
        return cls(name=name, times=times, dfs=dfs, flat_rate=float(rate))

    def df(self, t: float | np.ndarray) -> np.ndarray:
        """Log-linear interpolation of discount factors."""
        t = np.asarray(t, dtype=float)
        log_df = np.log(self.dfs)
        out = np.interp(t, self.times, log_df, left=log_df[0], right=log_df[-1])
        return np.exp(out)

    def forward_rate(self, t0: float, t1: float) -> float:
        """Continuously-compounded forward rate between t0 and t1."""
        if t1 <= t0:
            raise ValueError("Need t1 > t0")
        df0 = float(self.df(t0))
        df1 = float(self.df(t1))
        return (np.log(df0) - np.log(df1)) / (t1 - t0)

    def step_forward_rates(self, grid: np.ndarray) -> np.ndarray:
        """Forward rates on each interval of a time grid."""
        grid = np.asarray(grid, dtype=float)
        if np.any(np.diff(grid) <= 0.0):
            raise ValueError("grid must be strictly increasing")
        df_grid = self.df(grid)
        dt = np.diff(grid)
        return (np.log(df_grid[:-1]) - np.log(df_grid[1:])) / dt

    def get_discount_factors(
        self,
        date_list: Sequence[datetime] | Sequence[float] | np.ndarray,
        dtobjects: bool = True,
    ) -> np.ndarray:
        """Get discount factors for given date list or year fractions.

        For dtobjects=True, this uses the earliest date as t=0.
        """
        if dtobjects:
            dlist = get_year_deltas(date_list)
        else:
            dlist = np.array(date_list, dtype=float)
        discount_factors = self.df(dlist)
        return np.array((date_list, discount_factors)).T
