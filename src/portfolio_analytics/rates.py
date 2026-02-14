"Constant short rate discounting"

from dataclasses import dataclass
import logging
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

    def __post_init__(self) -> None:
        t = np.asarray(self.times, dtype=float)
        df = np.asarray(self.dfs, dtype=float)
        if t.ndim != 1 or df.ndim != 1 or t.shape != df.shape:
            raise ValueError("times and dfs must be 1D arrays of the same length")
        if np.any(np.diff(t) <= 0.0):
            raise ValueError("times must be strictly increasing")
        if np.any(df <= 0.0) or np.any(df > 1.0 + 1e-12):
            raise ValueError("discount factors must be in (0, 1]")
        object.__setattr__(self, "times", t)
        object.__setattr__(self, "dfs", df)

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

    def get_discount_factors(self, date_list, dtobjects: bool = True) -> np.ndarray:
        """Get discount factors for given date list or year fractions.

        For dtobjects=True, this uses the earliest date as t=0, matching
        the existing ConstantShortRate behavior.
        """
        if dtobjects:
            dlist = get_year_deltas(date_list)
        else:
            dlist = np.array(date_list, dtype=float)
        discount_factors = self.df(dlist)
        return np.array((date_list, discount_factors)).T


class ConstantShortRate:
    """Class for constant short rate discounting.

    Attributes
    ==========
    name: string
        name of the object
    short_rate: float (positive)
        constant rate for discounting

    Methods
    =======
    get_discount_factors:
        get discount factors given a list/array of datetime objects
        or year fractions
    """

    def __init__(self, name, short_rate):
        self.name = name
        self.short_rate = short_rate
        if short_rate < 0:
            logging.warning(
                "Negative short rate supplied to ConstantShortRate class. "
                "Discount factors will exceed 1 for future dates."
            )

    def get_discount_factors(self, date_list, dtobjects=True) -> np.ndarray:
        """Get discount factors for given date list.

        Applies the formula: DF(t) = exp(-r * t)

        Parameters
        ==========
        date_list: list or tuple
            collection of datetime objects or year fractions
        dtobjects: bool, default True
            if True, interpret date_list as datetime objects
            if False, interpret as year fractions

        Returns
        =======
        discount_factors: np.ndarray
            array of shape (n, 2) with [date_or_time, discount_factor]
            for each input date
        """
        if dtobjects is True:
            dlist = get_year_deltas(date_list)
        else:
            dlist = np.array(date_list)
        discount_factors = np.exp(-self.short_rate * dlist)
        return np.array((date_list, discount_factors)).T

    def df(self, t: float | np.ndarray) -> np.ndarray:
        """Return discount factor(s) for year fractions t."""
        t = np.asarray(t, dtype=float)
        return np.exp(-self.short_rate * t)

    def forward_rate(self, t0: float, t1: float) -> float:
        """Continuously-compounded forward rate between t0 and t1."""
        if t1 <= t0:
            raise ValueError("Need t1 > t0")
        return float(self.short_rate)

    def step_forward_rates(self, grid: np.ndarray) -> np.ndarray:
        """Forward rates on each interval of a time grid."""
        grid = np.asarray(grid, dtype=float)
        if np.any(np.diff(grid) <= 0.0):
            raise ValueError("grid must be strictly increasing")
        return np.full(len(grid) - 1, float(self.short_rate))
