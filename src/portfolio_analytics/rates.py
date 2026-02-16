"""Interest-rate and discount-curve utilities."""

from dataclasses import dataclass
import numpy as np

from .exceptions import ValidationError


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
            raise ValidationError("times and dfs must be 1D arrays of the same length")
        if np.any(np.diff(t) <= 0.0):
            raise ValidationError("times must be strictly increasing")
        if np.any(df <= 0.0) or np.any(df > 1.0 + 1e-12):
            raise ValidationError("discount factors must be in (0, 1]")
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
    def flat(
        cls,
        name: str,
        rate: float,
        end_time: float,
        steps: int = 1,
    ) -> "DiscountCurve":
        if end_time <= 0.0:
            raise ValidationError("end_time must be positive")
        if steps < 1:
            raise ValidationError("steps must be >= 1")
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
            raise ValidationError("Need t1 > t0")
        df0 = float(self.df(t0))
        df1 = float(self.df(t1))
        return (np.log(df0) - np.log(df1)) / (t1 - t0)

    def step_forward_rates(self, grid: np.ndarray) -> np.ndarray:
        """Forward rates on each interval of a time grid."""
        grid = np.asarray(grid, dtype=float)
        if np.any(np.diff(grid) <= 0.0):
            raise ValidationError("grid must be strictly increasing")
        df_grid = self.df(grid)
        dt = np.diff(grid)
        return (np.log(df_grid[:-1]) - np.log(df_grid[1:])) / dt
