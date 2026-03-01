"""Interest-rate and discount-curve utilities."""

import warnings
from dataclasses import dataclass

import numpy as np

from .exceptions import ValidationError


@dataclass(frozen=True, slots=True)
class DiscountCurve:
    """Deterministic discount curve with log-linear interpolation.

    times are year fractions and must be strictly increasing.
    dfs are positive discount factors, typically with df(0)=1.
    Values > 1 are permitted (negative rates) but trigger a warning.
    """

    times: np.ndarray
    dfs: np.ndarray
    flat_rate: float | None = None

    def __post_init__(self) -> None:
        t = np.asarray(self.times, dtype=float)
        df = np.asarray(self.dfs, dtype=float)
        if t.ndim != 1 or df.ndim != 1 or t.shape != df.shape:
            raise ValidationError("times and dfs must be 1D arrays of the same length")
        if np.any(np.diff(t) <= 0.0):
            raise ValidationError("times must be strictly increasing")
        if np.any(df <= 0.0):
            raise ValidationError("discount factors must be positive")
        if np.any(df > 1.0 + 1e-12):
            warnings.warn(
                "Discount factors > 1 detected (negative rates)",
                stacklevel=2,
            )
        if self.flat_rate is not None and not np.isfinite(float(self.flat_rate)):
            raise ValidationError("flat_rate must be finite when provided")
        if self.flat_rate is not None:
            implied = np.exp(-float(self.flat_rate) * t)
            if not np.allclose(df, implied, rtol=1e-10, atol=1e-12):
                raise ValidationError(
                    "flat_rate is only allowed when consistent with the provided discount factors"
                )
        object.__setattr__(self, "times", t)
        object.__setattr__(self, "dfs", df)

    @classmethod
    def from_forwards(
        cls,
        times: np.ndarray,
        forwards: np.ndarray,
    ) -> "DiscountCurve":
        """Build a curve from piecewise-constant forward rates.

        Parameters
        ----------
        times
            Year-fraction grid including 0.  Shape ``(N+1,)``.
        forwards
            Continuously-compounded forward rate on each interval.  Shape ``(N,)``.
        """
        times = np.asarray(times, dtype=float)
        forwards = np.asarray(forwards, dtype=float)
        if times.ndim != 1 or forwards.ndim != 1:
            raise ValidationError("times and forwards must be 1-D arrays")
        if times.size < 2:
            raise ValidationError("times must include at least [0, T]")
        if forwards.size != times.size - 1:
            raise ValidationError("forwards must have length len(times) - 1")
        if not np.isclose(times[0], 0.0):
            raise ValidationError("times must start at 0.0")
        dt_steps = np.diff(times)
        cum_rate = np.concatenate([[0.0], np.cumsum(forwards * dt_steps)])
        dfs = np.exp(-cum_rate)
        return cls(times=times, dfs=dfs)

    @classmethod
    def from_zero_rates(
        cls,
        times: np.ndarray,
        zero_rates: np.ndarray,
    ) -> "DiscountCurve":
        """Build a curve from continuously-compounded zero (spot) rates.

        Parameters
        ----------
        times
            Year-fraction grid.  Shape ``(N,)``.
            Must be strictly increasing and start at 0.
        zero_rates
            Continuously-compounded zero rates at each time.  Shape ``(N,)``.
            The rate at ``times[0] = 0`` is cosmetic (DF is always 1 there).
        """
        times = np.asarray(times, dtype=float)
        zero_rates = np.asarray(zero_rates, dtype=float)
        if times.ndim != 1 or zero_rates.ndim != 1:
            raise ValidationError("times and zero_rates must be 1-D arrays")
        if times.size != zero_rates.size:
            raise ValidationError("times and zero_rates must have the same length")
        if times.size < 2:
            raise ValidationError("times must include at least [0, T]")
        if not np.isclose(times[0], 0.0):
            raise ValidationError("times must start at 0.0")
        dfs = np.exp(-zero_rates * times)
        return cls(times=times, dfs=dfs)

    @classmethod
    def flat(
        cls,
        rate: float,
        end_time: float,
        steps: int = 1,
    ) -> "DiscountCurve":
        """Build a flat continuously-compounded discount curve.

        Parameters
        ----------
        rate
            Flat continuously-compounded annual rate.
        end_time
            Final maturity in years.
        steps
            Number of intervals used to discretize ``[0, end_time]``.

        Returns
        -------
        DiscountCurve
            Discount curve consistent with the supplied flat rate.
        """
        if end_time <= 0.0:
            raise ValidationError("end_time must be positive")
        if steps < 1:
            raise ValidationError("steps must be >= 1")
        times = np.linspace(0.0, float(end_time), int(steps) + 1)
        dfs = np.exp(-float(rate) * times)
        return cls(times=times, dfs=dfs, flat_rate=float(rate))

    def df(self, t: float | np.ndarray) -> np.ndarray:
        """Interpolate discount factors with log-linear interpolation.

        Parameters
        ----------
        t
            Scalar or array of year fractions.

        Returns
        -------
        np.ndarray
            Interpolated discount factors.
        """
        t = np.asarray(t, dtype=float)
        t_min, t_max = float(self.times[0]), float(self.times[-1])
        outside = (t < t_min) | (t > t_max)
        if np.any(outside):
            warnings.warn(
                f"Extrapolating discount curve outside "
                f"[{t_min:.4f}, {t_max:.4f}] — flat log-DF assumed",
                stacklevel=2,
            )
        log_df = np.log(self.dfs)
        out = np.interp(t, self.times, log_df, left=log_df[0], right=log_df[-1])
        return np.exp(out)

    def forward_rate(self, t0: float, t1: float) -> float:
        """Return continuously-compounded forward rate on ``[t0, t1]``."""
        if t1 <= t0:
            raise ValidationError("Need t1 > t0")
        df0 = float(self.df(t0))
        df1 = float(self.df(t1))
        return (np.log(df0) - np.log(df1)) / (t1 - t0)

    def step_forward_rates(self, grid: np.ndarray) -> np.ndarray:
        """Return forward rates on each interval of a time grid."""
        grid = np.asarray(grid, dtype=float)
        if np.any(np.diff(grid) <= 0.0):
            raise ValidationError("grid must be strictly increasing")
        df_grid = self.df(grid)
        dt = np.diff(grid)
        return (np.log(df_grid[:-1]) - np.log(df_grid[1:])) / dt
