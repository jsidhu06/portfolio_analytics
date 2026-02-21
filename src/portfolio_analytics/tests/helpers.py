import datetime as dt

import numpy as np

from portfolio_analytics.rates import DiscountCurve
from portfolio_analytics.utils import calculate_year_fraction


def flat_curve(
    pricing_date: dt.datetime,
    maturity: dt.datetime,
    rate: float,
    name: str = "r",
) -> DiscountCurve:
    ttm = calculate_year_fraction(pricing_date, maturity)
    return DiscountCurve.flat(name, rate, end_time=ttm)


def build_curve_from_forwards(
    *,
    name: str,
    times: np.ndarray,
    forwards: np.ndarray,
) -> DiscountCurve:
    """Build a DiscountCurve from piecewise-constant forward rates.

    Parameters
    ----------
    name : str
        Curve name.
    times : np.ndarray
        Time grid including 0. Shape ``(N+1,)``.
    forwards : np.ndarray
        Forward rate on each interval. Shape ``(N,)``.

    Returns
    -------
    DiscountCurve
    """
    times = np.asarray(times, dtype=float)
    forwards = np.asarray(forwards, dtype=float)
    if times.ndim != 1 or forwards.ndim != 1:
        raise ValueError("times/forwards must be 1D arrays")
    if times.size < 2:
        raise ValueError("times must include at least [0, T]")
    if forwards.size != times.size - 1:
        raise ValueError("forwards must have length len(times)-1")
    if not np.isclose(times[0], 0.0):
        raise ValueError("times must start at 0.0")

    dt_steps = np.diff(times)
    cum_rate = np.concatenate([[0.0], np.cumsum(forwards * dt_steps)])
    dfs = np.exp(-cum_rate)
    return DiscountCurve(name=name, times=times, dfs=dfs)
