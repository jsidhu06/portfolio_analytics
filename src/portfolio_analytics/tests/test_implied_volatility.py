"""Tests for implied volatility solver."""

import datetime as dt

import numpy as np
import pytest

from portfolio_analytics.enums import ExerciseType, ImpliedVolMethod, OptionType, PricingMethod
from portfolio_analytics.market_environment import MarketData
from portfolio_analytics.rates import ConstantShortRate
from portfolio_analytics.valuation import (
    ImpliedVolResult,
    OptionSpec,
    OptionValuation,
    UnderlyingPricingData,
    implied_volatility,
)


def _build_valuation(
    *,
    option_type: OptionType,
    spot: float = 100.0,
    strike: float = 100.0,
    vol: float = 0.2,
    rate: float = 0.05,
    dividend_yield: float = 0.0,
) -> OptionValuation:
    pricing_date = dt.datetime(2025, 1, 1)
    maturity = dt.datetime(2026, 1, 1)
    market_data = MarketData(pricing_date, ConstantShortRate("csr", rate), currency="USD")
    underlying = UnderlyingPricingData(
        initial_value=spot,
        volatility=vol,
        market_data=market_data,
        dividend_yield=dividend_yield,
    )
    spec = OptionSpec(
        option_type=option_type,
        exercise_type=ExerciseType.EUROPEAN,
        strike=strike,
        maturity=maturity,
        currency="USD",
    )
    return OptionValuation(
        name="test",
        underlying=underlying,
        spec=spec,
        pricing_method=PricingMethod.BSM,
    )


def _build_discrete_dividend_valuation(
    *,
    option_type: OptionType,
    vol: float,
    rate: float = 0.1,
    spot: float = 52.0,
    strike: float = 50.0,
) -> OptionValuation:
    pricing_date = dt.datetime(2025, 1, 1)
    maturity = pricing_date + dt.timedelta(days=365)
    market_data = MarketData(pricing_date, ConstantShortRate("csr", rate), currency="USD")
    divs = [
        (pricing_date + dt.timedelta(days=90), 0.5),
        (pricing_date + dt.timedelta(days=270), 0.5),
    ]
    underlying = UnderlyingPricingData(
        initial_value=spot,
        volatility=vol,
        market_data=market_data,
        discrete_dividends=divs,
    )
    spec = OptionSpec(
        option_type=option_type,
        exercise_type=ExerciseType.EUROPEAN,
        strike=strike,
        maturity=maturity,
        currency="USD",
    )
    return OptionValuation(
        name="test_divs",
        underlying=underlying,
        spec=spec,
        pricing_method=PricingMethod.BSM,
    )


@pytest.mark.parametrize("method", [ImpliedVolMethod.NEWTON_RAPHSON, ImpliedVolMethod.BISECTION])
def test_implied_volatility_recovers_call_vol(method: ImpliedVolMethod):
    true_vol = 0.25
    initial_vol = 0.12
    pricing_valuation = _build_valuation(option_type=OptionType.CALL, vol=true_vol)
    target_price = pricing_valuation.present_value()

    solver_valuation = _build_valuation(option_type=OptionType.CALL, vol=initial_vol)
    result = implied_volatility(target_price, solver_valuation, method=method)

    assert isinstance(result, ImpliedVolResult)
    assert result.converged
    assert abs(result.implied_vol - 0.25) < 1.0e-6


def test_implied_volatility_recovers_put_vol_with_brentq():
    true_vol = 0.18
    initial_vol = 0.35
    pricing_valuation = _build_valuation(option_type=OptionType.PUT, vol=true_vol)
    target_price = pricing_valuation.present_value()

    solver_valuation = _build_valuation(option_type=OptionType.PUT, vol=initial_vol)
    result = implied_volatility(target_price, solver_valuation, method=ImpliedVolMethod.BRENTQ)

    assert result.converged
    assert abs(result.implied_vol - 0.18) < 1.0e-6


def test_implied_volatility_rejects_out_of_bounds_price():
    valuation = _build_valuation(option_type=OptionType.CALL, vol=0.2)
    target_price = valuation.present_value() + 1000.0

    with pytest.raises(ValueError, match="outside no-arbitrage bounds"):
        implied_volatility(target_price, valuation)


def test_implied_volatility_rejects_unimplemented_method():
    valuation = _build_valuation(option_type=OptionType.CALL, vol=0.2)
    target_price = valuation.present_value()

    with pytest.raises(NotImplementedError, match="not implemented"):
        implied_volatility(target_price, valuation, method=ImpliedVolMethod.BRENNER_SUBRAHMANYAM)


@pytest.mark.parametrize("pricing_method", [PricingMethod.BINOMIAL, PricingMethod.PDE_FD])
def test_implied_volatility_requires_bsm(pricing_method: PricingMethod):
    valuation = _build_valuation(option_type=OptionType.CALL, vol=0.2)
    target_price = valuation.present_value()

    pricing_date = valuation.pricing_date
    maturity = valuation.maturity
    market_data = MarketData(pricing_date, ConstantShortRate("csr", 0.05), currency="USD")
    underlying = UnderlyingPricingData(
        initial_value=100.0,
        volatility=0.2,
        market_data=market_data,
    )
    spec = OptionSpec(
        option_type=OptionType.CALL,
        exercise_type=ExerciseType.EUROPEAN,
        strike=100.0,
        maturity=maturity,
        currency="USD",
    )
    valuation_object = OptionValuation(
        name="test",
        underlying=underlying,
        spec=spec,
        pricing_method=pricing_method,
    )
    with pytest.raises(NotImplementedError, match="BSM"):
        implied_volatility(target_price, valuation_object)


def test_implied_volatility_requires_european():
    valuation = _build_valuation(option_type=OptionType.CALL, vol=0.2)
    target_price = valuation.present_value()

    valuation.exercise_type = ExerciseType.AMERICAN
    with pytest.raises(NotImplementedError, match="European"):
        implied_volatility(target_price, valuation)


def test_implied_volatility_with_dividend_yield():
    true_vol = 0.3
    initial_vol = 0.1
    pricing_valuation = _build_valuation(
        option_type=OptionType.CALL, vol=true_vol, dividend_yield=0.02
    )
    target_price = pricing_valuation.present_value()

    solver_valuation = _build_valuation(
        option_type=OptionType.CALL, vol=initial_vol, dividend_yield=0.02
    )
    result = implied_volatility(target_price, solver_valuation)

    assert result.converged
    assert np.isclose(result.implied_vol, 0.3, atol=1.0e-6)


def test_implied_volatility_with_discrete_dividends():
    true_vol = 0.4
    initial_vol = 0.15
    pricing_valuation = _build_discrete_dividend_valuation(option_type=OptionType.PUT, vol=true_vol)
    target_price = pricing_valuation.present_value()

    solver_valuation = _build_discrete_dividend_valuation(
        option_type=OptionType.PUT, vol=initial_vol
    )
    result = implied_volatility(target_price, solver_valuation)

    assert result.converged
    assert np.isclose(result.implied_vol, true_vol, atol=1.0e-6)
