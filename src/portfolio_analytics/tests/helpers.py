import datetime as dt
import warnings

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
    """Deprecated — use ``DiscountCurve.from_forwards()`` instead."""
    warnings.warn(
        "build_curve_from_forwards is deprecated; use DiscountCurve.from_forwards()",
        DeprecationWarning,
        stacklevel=2,
    )
    return DiscountCurve.from_forwards(name=name, times=times, forwards=forwards)
