"""Tests for forward price and put-call parity utilities."""

import datetime as dt
import numpy as np

from portfolio_analytics.enums import OptionType, ExerciseType, PricingMethod
from portfolio_analytics.market_environment import MarketData
from portfolio_analytics.tests.helpers import flat_curve
from portfolio_analytics.valuation import OptionSpec, UnderlyingPricingData, OptionValuation
from portfolio_analytics.utils import (
    calculate_year_fraction,
    forward_price,
    put_call_parity_rhs,
    put_call_parity_gap,
)


def test_forward_price_continuous_dividend_yield():
    pricing_date = dt.datetime(2025, 1, 1)
    maturity = dt.datetime(2026, 1, 1)
    ttm = calculate_year_fraction(pricing_date, maturity)
    spot = 100.0
    r = 0.05
    q = 0.02

    fwd = forward_price(
        spot=spot,
        pricing_date=pricing_date,
        maturity=maturity,
        short_rate=r,
        dividend_yield=q,
    )

    expected = spot * np.exp((r - q) * ttm)
    assert np.isclose(fwd, expected)


def test_forward_price_discrete_dividends():
    pricing_date = dt.datetime(2025, 1, 1)
    maturity = dt.datetime(2026, 1, 1)
    ttm = calculate_year_fraction(pricing_date, maturity)
    spot = 100.0
    r = 0.05
    dividends = [(dt.datetime(2025, 7, 1), 0.5)]

    fwd = forward_price(
        spot=spot,
        pricing_date=pricing_date,
        maturity=maturity,
        short_rate=r,
        discrete_dividends=dividends,
    )

    t_div = calculate_year_fraction(pricing_date, dividends[0][0])
    pv_div = 0.5 * np.exp(-r * t_div)
    expected = (spot - pv_div) * np.exp(r * ttm)
    assert np.isclose(fwd, expected)


def test_put_call_parity_rhs_and_gap():
    pricing_date = dt.datetime(2025, 1, 1)
    maturity = dt.datetime(2026, 1, 1)
    spot = 100.0
    strike = 100.0
    r = 0.05

    rhs = put_call_parity_rhs(
        spot=spot,
        strike=strike,
        pricing_date=pricing_date,
        maturity=maturity,
        short_rate=r,
    )

    call_price = 10.0
    put_price = call_price - rhs

    gap = put_call_parity_gap(
        call_price=call_price,
        put_price=put_price,
        spot=spot,
        strike=strike,
        pricing_date=pricing_date,
        maturity=maturity,
        short_rate=r,
    )

    assert np.isclose(gap, 0.0)


def test_put_call_parity_bsm_no_dividend():
    pricing_date = dt.datetime(2025, 1, 1)
    maturity = dt.datetime(2026, 1, 1)
    spot = 100.0
    strike = 100.0
    r = 0.05

    curve = flat_curve(pricing_date, maturity, r)
    market_data = MarketData(pricing_date, curve, currency="USD")

    underlying_call = UnderlyingPricingData(
        initial_value=spot,
        volatility=0.2,
        market_data=market_data,
    )
    underlying_put = UnderlyingPricingData(
        initial_value=spot,
        volatility=0.2,
        market_data=market_data,
    )

    call_spec = OptionSpec(
        option_type=OptionType.CALL,
        exercise_type=ExerciseType.EUROPEAN,
        strike=strike,
        maturity=maturity,
        currency="USD",
    )
    put_spec = OptionSpec(
        option_type=OptionType.PUT,
        exercise_type=ExerciseType.EUROPEAN,
        strike=strike,
        maturity=maturity,
        currency="USD",
    )

    call_price = OptionValuation(
        name="CALL",
        underlying=underlying_call,
        spec=call_spec,
        pricing_method=PricingMethod.BSM,
    ).present_value()

    put_price = OptionValuation(
        name="PUT",
        underlying=underlying_put,
        spec=put_spec,
        pricing_method=PricingMethod.BSM,
    ).present_value()

    rhs = put_call_parity_rhs(
        spot=spot,
        strike=strike,
        pricing_date=pricing_date,
        maturity=maturity,
        short_rate=r,
    )

    assert np.isclose(call_price - put_price, rhs, rtol=1e-10)


def test_put_call_parity_bsm_with_dividend_yield():
    pricing_date = dt.datetime(2025, 1, 1)
    maturity = dt.datetime(2026, 1, 1)
    spot = 100.0
    strike = 105.0
    r = 0.04
    q = 0.02

    curve = flat_curve(pricing_date, maturity, r)
    market_data = MarketData(pricing_date, curve, currency="USD")
    underlying_call = UnderlyingPricingData(
        initial_value=spot,
        volatility=0.25,
        market_data=market_data,
        dividend_yield=q,
    )
    underlying_put = UnderlyingPricingData(
        initial_value=spot,
        volatility=0.25,
        market_data=market_data,
        dividend_yield=q,
    )

    call_spec = OptionSpec(
        option_type=OptionType.CALL,
        exercise_type=ExerciseType.EUROPEAN,
        strike=strike,
        maturity=maturity,
        currency="USD",
    )
    put_spec = OptionSpec(
        option_type=OptionType.PUT,
        exercise_type=ExerciseType.EUROPEAN,
        strike=strike,
        maturity=maturity,
        currency="USD",
    )

    call_price = OptionValuation(
        name="CALL_DIV",
        underlying=underlying_call,
        spec=call_spec,
        pricing_method=PricingMethod.BSM,
    ).present_value()

    put_price = OptionValuation(
        name="PUT_DIV",
        underlying=underlying_put,
        spec=put_spec,
        pricing_method=PricingMethod.BSM,
    ).present_value()

    rhs = put_call_parity_rhs(
        spot=spot,
        strike=strike,
        pricing_date=pricing_date,
        maturity=maturity,
        short_rate=r,
        dividend_yield=q,
    )

    assert np.isclose(call_price - put_price, rhs, rtol=1e-10)
