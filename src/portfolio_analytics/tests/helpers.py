import datetime as dt
import warnings
from typing import Any

import numpy as np

from portfolio_analytics.enums import ExerciseType, OptionType
from portfolio_analytics.market_environment import MarketData
from portfolio_analytics.rates import DiscountCurve
from portfolio_analytics.utils import calculate_year_fraction
from portfolio_analytics.valuation import OptionValuation, UnderlyingData, VanillaSpec


# ---------------------------------------------------------------------------
# Scalar constants (canonical source — re-exported by conftest.py)
# ---------------------------------------------------------------------------

PRICING_DATE = dt.datetime(2025, 1, 1)
MATURITY = dt.datetime(2026, 1, 1)
CURRENCY = "USD"
SPOT = 100.0
STRIKE = 100.0
RATE = 0.05
VOL = 0.20


def flat_curve(
    pricing_date: dt.datetime,
    maturity: dt.datetime,
    rate: float,
) -> DiscountCurve:
    ttm = calculate_year_fraction(pricing_date, maturity)
    return DiscountCurve.flat(rate, end_time=ttm)


_CURVE = flat_curve(PRICING_DATE, MATURITY, RATE)


def market_data() -> MarketData:
    return MarketData(PRICING_DATE, _CURVE, currency=CURRENCY)


_MD = market_data()


def underlying(
    spot: float = SPOT,
    vol: float = VOL,
    dividend_curve=None,
    discrete_dividends=None,
) -> UnderlyingData:
    return UnderlyingData(
        initial_value=spot,
        volatility=vol,
        market_data=_MD,
        dividend_curve=dividend_curve,
        discrete_dividends=discrete_dividends,
    )


def spec(
    option_type: OptionType = OptionType.CALL,
    exercise: ExerciseType = ExerciseType.EUROPEAN,
    strike: float = STRIKE,
) -> VanillaSpec:
    return VanillaSpec(
        option_type=option_type,
        exercise_type=exercise,
        strike=strike,
        maturity=MATURITY,
        currency=CURRENCY,
    )


def pv(underlying: Any, spec: Any, method: Any, **kw: Any) -> float:
    """Shortcut: build an OptionValuation and return its present value."""

    return OptionValuation(underlying, spec, method, **kw).present_value()


def build_curve_from_forwards(
    *,
    times: np.ndarray,
    forwards: np.ndarray,
) -> DiscountCurve:
    """Deprecated — use ``DiscountCurve.from_forwards()`` instead."""
    warnings.warn(
        "build_curve_from_forwards is deprecated; use DiscountCurve.from_forwards()",
        DeprecationWarning,
        stacklevel=2,
    )
    return DiscountCurve.from_forwards(times=times, forwards=forwards)
