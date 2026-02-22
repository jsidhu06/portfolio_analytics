"""Shared pytest fixtures for portfolio_analytics tests."""

import datetime as dt

import pytest

from portfolio_analytics.enums import (
    ExerciseType,
    OptionType,
)
from portfolio_analytics.market_environment import MarketData
from portfolio_analytics.rates import DiscountCurve
from portfolio_analytics.valuation import (
    OptionSpec,
    UnderlyingPricingData,
)

from portfolio_analytics.tests.helpers import flat_curve


# ---------------------------------------------------------------------------
# Scalar constants
# ---------------------------------------------------------------------------

PRICING_DATE = dt.datetime(2025, 1, 1)
MATURITY = dt.datetime(2026, 1, 1)
CURRENCY = "USD"
SPOT = 100.0
STRIKE = 100.0
RATE = 0.05
VOL = 0.20


@pytest.fixture()
def pricing_date() -> dt.datetime:
    return PRICING_DATE


@pytest.fixture()
def maturity() -> dt.datetime:
    return MATURITY


@pytest.fixture()
def currency() -> str:
    return CURRENCY


@pytest.fixture()
def risk_free_rate() -> float:
    return RATE


@pytest.fixture()
def vol() -> float:
    return VOL


@pytest.fixture()
def spot() -> float:
    return SPOT


@pytest.fixture()
def strike() -> float:
    return STRIKE


# ---------------------------------------------------------------------------
# Curve / Market Data
# ---------------------------------------------------------------------------


@pytest.fixture()
def discount_curve(
    pricing_date: dt.datetime, maturity: dt.datetime, risk_free_rate: float
) -> DiscountCurve:
    """Flat discount curve over [pricing_date, maturity]."""
    return flat_curve(pricing_date, maturity, risk_free_rate, name="csr")


@pytest.fixture()
def market_data(
    pricing_date: dt.datetime, discount_curve: DiscountCurve, currency: str
) -> MarketData:
    return MarketData(pricing_date, discount_curve, currency=currency)


@pytest.fixture()
def underlying_data(market_data: MarketData) -> UnderlyingPricingData:
    """ATM underlying with no dividends."""
    return UnderlyingPricingData(
        initial_value=SPOT,
        volatility=VOL,
        market_data=market_data,
    )


# ---------------------------------------------------------------------------
# Option specs
# ---------------------------------------------------------------------------


@pytest.fixture()
def euro_call_spec(strike: float, maturity: dt.datetime, currency: str) -> OptionSpec:
    return OptionSpec(
        option_type=OptionType.CALL,
        exercise_type=ExerciseType.EUROPEAN,
        strike=strike,
        maturity=maturity,
        currency=currency,
    )


@pytest.fixture()
def euro_put_spec(strike: float, maturity: dt.datetime, currency: str) -> OptionSpec:
    return OptionSpec(
        option_type=OptionType.PUT,
        exercise_type=ExerciseType.EUROPEAN,
        strike=strike,
        maturity=maturity,
        currency=currency,
    )


# ---------------------------------------------------------------------------
# Common discrete dividends
# ---------------------------------------------------------------------------


@pytest.fixture()
def standard_discrete_dividends(pricing_date: dt.datetime) -> list[tuple[dt.datetime, float]]:
    """Two semi-annual dividends of 0.50 each."""
    return [
        (pricing_date + dt.timedelta(days=90), 0.50),
        (pricing_date + dt.timedelta(days=270), 0.50),
    ]
