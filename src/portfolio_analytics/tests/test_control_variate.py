"""Tests for Hull-style control variate adjustments."""

import datetime as dt

import numpy as np

from portfolio_analytics.enums import ExerciseType, OptionType, PricingMethod
from portfolio_analytics.market_environment import MarketData
from portfolio_analytics.tests.helpers import flat_curve
from portfolio_analytics.valuation import OptionSpec, OptionValuation, UnderlyingPricingData
from portfolio_analytics.valuation.params import BinomialParams, PDEParams


PRICING_DATE = dt.datetime(2025, 1, 1)
MATURITY = dt.datetime(2026, 1, 1)
CURRENCY = "USD"


def _build_underlying(spot: float, vol: float, rate: float, dividend_rate: float = 0.0):
    curve = flat_curve(PRICING_DATE, MATURITY, rate, name="csr")
    dividend_curve = (
        None
        if dividend_rate == 0.0
        else flat_curve(PRICING_DATE, MATURITY, dividend_rate, name="q")
    )
    market_data = MarketData(PRICING_DATE, curve, currency=CURRENCY)
    return UnderlyingPricingData(
        initial_value=spot,
        volatility=vol,
        market_data=market_data,
        dividend_curve=dividend_curve,
    )


def _build_spec(option_type: OptionType, exercise_type: ExerciseType, strike: float) -> OptionSpec:
    return OptionSpec(
        option_type=option_type,
        exercise_type=exercise_type,
        strike=strike,
        maturity=MATURITY,
        currency=CURRENCY,
    )


def test_control_variate_binomial_matches_adjustment():
    spot = 100.0
    strike = 110.0
    vol = 0.25
    rate = 0.03

    underlying = _build_underlying(spot, vol, rate)
    amer_spec = _build_spec(OptionType.PUT, ExerciseType.AMERICAN, strike)
    euro_spec = _build_spec(OptionType.PUT, ExerciseType.EUROPEAN, strike)

    base_params = BinomialParams(num_steps=300)
    cv_params = BinomialParams(num_steps=300, control_variate_european=True)

    american_raw = OptionValuation(
        "amer_binom_raw",
        underlying,
        amer_spec,
        PricingMethod.BINOMIAL,
        params=base_params,
    ).present_value()

    european_num = OptionValuation(
        "euro_binom_num",
        underlying,
        euro_spec,
        PricingMethod.BINOMIAL,
        params=base_params,
    ).present_value()

    european_bsm = OptionValuation(
        "euro_bsm",
        underlying,
        euro_spec,
        PricingMethod.BSM,
    ).present_value()

    american_cv = OptionValuation(
        "amer_binom_cv",
        underlying,
        amer_spec,
        PricingMethod.BINOMIAL,
        params=cv_params,
    ).present_value()

    expected = american_raw + (european_bsm - european_num)
    assert np.isclose(american_cv, expected, atol=1.0e-8)


def test_control_variate_pde_matches_adjustment():
    spot = 100.0
    strike = 95.0
    vol = 0.2
    rate = 0.04

    underlying = _build_underlying(spot, vol, rate)
    amer_spec = _build_spec(OptionType.PUT, ExerciseType.AMERICAN, strike)
    euro_spec = _build_spec(OptionType.PUT, ExerciseType.EUROPEAN, strike)

    base_params = PDEParams(spot_steps=60, time_steps=60)
    cv_params = PDEParams(spot_steps=60, time_steps=60, control_variate_european=True)

    american_raw = OptionValuation(
        "amer_pde_raw",
        underlying,
        amer_spec,
        PricingMethod.PDE_FD,
        params=base_params,
    ).present_value()

    european_num = OptionValuation(
        "euro_pde_num",
        underlying,
        euro_spec,
        PricingMethod.PDE_FD,
        params=base_params,
    ).present_value()

    european_bsm = OptionValuation(
        "euro_bsm",
        underlying,
        euro_spec,
        PricingMethod.BSM,
    ).present_value()

    american_cv = OptionValuation(
        "amer_pde_cv",
        underlying,
        amer_spec,
        PricingMethod.PDE_FD,
        params=cv_params,
    ).present_value()

    expected = american_raw + (european_bsm - european_num)
    assert np.isclose(american_cv, expected, atol=1.0e-6)
