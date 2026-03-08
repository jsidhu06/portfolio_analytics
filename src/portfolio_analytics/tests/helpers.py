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
_MD = MarketData(PRICING_DATE, _CURVE, currency=CURRENCY)


def market_data(
    rate: float = RATE,
    maturity: dt.datetime = MATURITY,
    currency: str = CURRENCY,
) -> MarketData:
    if rate == RATE and maturity == MATURITY and currency == CURRENCY:
        return _MD
    curve = flat_curve(PRICING_DATE, maturity, rate)
    return MarketData(PRICING_DATE, curve, currency=currency)


def underlying(
    spot: float = SPOT,
    vol: float = VOL,
    rate: float = RATE,
    dividend_rate: float = 0.0,
    dividend_curve: DiscountCurve | None = None,
    discrete_dividends=None,
    maturity: dt.datetime = MATURITY,
) -> UnderlyingData:
    md = market_data(rate=rate, maturity=maturity)
    if dividend_curve is None and dividend_rate != 0.0:
        dividend_curve = flat_curve(PRICING_DATE, maturity, dividend_rate)
    return UnderlyingData(
        initial_value=spot,
        volatility=vol,
        market_data=md,
        dividend_curve=dividend_curve,
        discrete_dividends=discrete_dividends,
    )


def spec(
    option_type: OptionType = OptionType.CALL,
    exercise: ExerciseType = ExerciseType.EUROPEAN,
    strike: float = STRIKE,
    maturity: dt.datetime = MATURITY,
    currency: str = CURRENCY,
) -> VanillaSpec:
    return VanillaSpec(
        option_type=option_type,
        exercise_type=exercise,
        strike=strike,
        maturity=maturity,
        currency=currency,
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
