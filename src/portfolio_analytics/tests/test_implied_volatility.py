"""Tests for implied volatility solver."""

import datetime as dt

import numpy as np
import pytest

from portfolio_analytics.exceptions import UnsupportedFeatureError, ValidationError
from portfolio_analytics.enums import (
    ExerciseType,
    ImpliedVolMethod,
    OptionType,
    PricingMethod,
)
from portfolio_analytics.market_environment import MarketData
from portfolio_analytics.tests.helpers import flat_curve
from portfolio_analytics.tests.helpers import (
    market_data,
    underlying,
    make_vanilla_spec,
)
from portfolio_analytics.stochastic_processes import (
    GBMParams,
    GBMProcess,
    SimulationConfig,
)
from portfolio_analytics.valuation import (
    ImpliedVolResult,
    VanillaSpec,
    OptionValuation,
    implied_volatility,
)
from portfolio_analytics.valuation.params import BinomialParams


def _build_valuation(
    *,
    option_type: OptionType,
    spot: float = 100.0,
    strike: float = 100.0,
    vol: float = 0.2,
    rate: float = 0.05,
    dividend_rate: float = 0.0,
) -> OptionValuation:
    pricing_date = dt.datetime(2025, 1, 1)
    maturity = dt.datetime(2026, 1, 1)
    market_data_obj = market_data(
        pricing_date=pricing_date,
        discount_curve=flat_curve(pricing_date, maturity, rate),
    )
    dividend_curve = flat_curve(pricing_date, maturity, dividend_rate) if dividend_rate else None
    underlying_data = underlying(
        initial_value=spot,
        volatility=vol,
        market_data=market_data_obj,
        dividend_curve=dividend_curve,
    )
    spec = make_vanilla_spec(
        strike=strike,
        maturity=maturity,
        option_type=option_type,
        exercise_type=ExerciseType.EUROPEAN,
    )
    return OptionValuation(
        underlying=underlying_data,
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
    divs = [
        (pricing_date + dt.timedelta(days=90), 0.5),
        (pricing_date + dt.timedelta(days=270), 0.5),
    ]
    market_data_obj = market_data(
        pricing_date=pricing_date,
        discount_curve=flat_curve(pricing_date, maturity, rate),
    )
    underlying_data = underlying(
        initial_value=spot,
        volatility=vol,
        market_data=market_data_obj,
        discrete_dividends=divs,
    )
    spec = make_vanilla_spec(
        strike=strike,
        maturity=maturity,
        option_type=option_type,
        exercise_type=ExerciseType.EUROPEAN,
    )
    return OptionValuation(
        underlying=underlying_data,
        spec=spec,
        pricing_method=PricingMethod.BSM,
    )


def _build_binomial_valuation(
    *,
    option_type: OptionType,
    exercise_type: ExerciseType,
    spot: float = 100.0,
    strike: float = 100.0,
    vol: float = 0.2,
    rate: float = 0.05,
    dividend_rate: float = 0.0,
    num_steps: int = 400,
) -> OptionValuation:
    pricing_date = dt.datetime(2025, 1, 1)
    maturity = dt.datetime(2026, 1, 1)
    market_data_obj = market_data(
        pricing_date=pricing_date,
        discount_curve=flat_curve(pricing_date, maturity, rate),
    )
    dividend_curve = flat_curve(pricing_date, maturity, dividend_rate) if dividend_rate else None
    underlying_data = underlying(
        initial_value=spot,
        volatility=vol,
        market_data=market_data_obj,
        dividend_curve=dividend_curve,
    )
    spec = make_vanilla_spec(
        strike=strike,
        maturity=maturity,
        option_type=option_type,
        exercise_type=exercise_type,
    )
    return OptionValuation(
        underlying=underlying_data,
        spec=spec,
        pricing_method=PricingMethod.BINOMIAL,
        params=BinomialParams(num_steps=num_steps),
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

    with pytest.raises(ValidationError, match="outside no-arbitrage bounds"):
        implied_volatility(target_price, valuation)


def test_implied_volatility_rejects_monte_carlo():
    pricing_date = dt.datetime(2025, 1, 1)
    maturity = dt.datetime(2026, 1, 1)
    curve = flat_curve(pricing_date, maturity, 0.05)
    market_data = MarketData(pricing_date, curve, currency="USD")
    sim_config = SimulationConfig(
        paths=5_000,
        num_steps=50,
        end_date=maturity,
    )
    gbm_params = GBMParams(
        initial_value=100.0,
        volatility=0.2,
    )
    underlying = GBMProcess(market_data, gbm_params, sim_config)
    spec = VanillaSpec(
        option_type=OptionType.CALL,
        exercise_type=ExerciseType.EUROPEAN,
        strike=100.0,
        maturity=maturity,
        currency="USD",
    )
    valuation = OptionValuation(
        underlying=underlying,
        spec=spec,
        pricing_method=PricingMethod.MONTE_CARLO,
    )
    target_price = valuation.present_value()

    with pytest.raises(UnsupportedFeatureError, match="pricing methods"):
        implied_volatility(target_price, valuation)


def test_implied_volatility_with_dividend_curve():
    true_vol = 0.3
    initial_vol = 0.1
    pricing_valuation = _build_valuation(
        option_type=OptionType.CALL, vol=true_vol, dividend_rate=0.02
    )
    target_price = pricing_valuation.present_value()

    solver_valuation = _build_valuation(
        option_type=OptionType.CALL, vol=initial_vol, dividend_rate=0.02
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


@pytest.mark.parametrize("option_type", [OptionType.CALL, OptionType.PUT])
def test_implied_volatility_recovers_american_binomial(option_type: OptionType):
    true_vol = 0.3
    initial_vol = 0.12
    dividend_rate = 0.02 if option_type is OptionType.CALL else 0.0

    pricing_valuation = _build_binomial_valuation(
        option_type=option_type,
        exercise_type=ExerciseType.AMERICAN,
        vol=true_vol,
        dividend_rate=dividend_rate,
    )
    target_price = pricing_valuation.present_value()

    solver_valuation = _build_binomial_valuation(
        option_type=option_type,
        exercise_type=ExerciseType.AMERICAN,
        vol=initial_vol,
        dividend_rate=dividend_rate,
    )
    result = implied_volatility(
        target_price,
        solver_valuation,
        method=ImpliedVolMethod.BISECTION,
        tol=1.0e-6,
    )

    assert result.converged
    assert np.isclose(result.implied_vol, true_vol, atol=5.0e-3)


@pytest.mark.parametrize("method", [ImpliedVolMethod.NEWTON_RAPHSON, ImpliedVolMethod.BRENTQ])
@pytest.mark.parametrize("option_type", [OptionType.CALL, OptionType.PUT])
@pytest.mark.parametrize("true_vol", [1.0e-3, 3.0])
def test_implied_volatility_extreme_vols(option_type, true_vol, method):
    """IV solver should return a volatility that reprices extreme-vol targets."""
    pricing_valuation = _build_valuation(option_type=option_type, vol=true_vol)
    target_price = pricing_valuation.present_value()

    solver_valuation = _build_valuation(option_type=option_type, vol=0.2)
    result = implied_volatility(
        target_price,
        solver_valuation,
        method=method,
        vol_bounds=(1.0e-6, 5.0),
        tol=1.0e-8,
        max_iter=200,
    )
    assert result.converged
    repricing_valuation = _build_valuation(option_type=option_type, vol=result.implied_vol)
    repriced = repricing_valuation.present_value()
    assert np.isclose(repriced, target_price, atol=1.0e-7)


def test_implied_volatility_rejects_call_price_above_spot():
    """European call cannot exceed spot under no-arbitrage bounds."""
    valuation = _build_valuation(option_type=OptionType.CALL, vol=0.2)
    with pytest.raises(ValidationError, match="outside no-arbitrage bounds"):
        implied_volatility(valuation.underlying.initial_value + 1.0, valuation)


def test_implied_volatility_rejects_put_price_above_discounted_strike():
    """European put upper bound is discounted strike."""
    valuation = _build_valuation(option_type=OptionType.PUT, vol=0.2)
    ttm = (valuation.maturity - valuation.pricing_date).days / 365.0
    upper = valuation.strike * np.exp(-0.05 * ttm)
    with pytest.raises(ValidationError, match="outside no-arbitrage bounds"):
        implied_volatility(float(upper + 1.0), valuation)


@pytest.mark.parametrize("method", [ImpliedVolMethod.BISECTION, ImpliedVolMethod.BRENTQ])
def test_implied_volatility_converges_near_lower_bound(method):
    """Near the lower bound, solver should still produce a price-consistent root."""
    true_vol = 2.0e-3
    pricing_valuation = _build_valuation(option_type=OptionType.CALL, vol=true_vol)
    target_price = pricing_valuation.present_value()

    solver_valuation = _build_valuation(option_type=OptionType.CALL, vol=0.2)
    result = implied_volatility(
        target_price,
        solver_valuation,
        method=method,
        vol_bounds=(1.0e-6, 5.0),
        tol=1.0e-8,
        max_iter=200,
    )
    assert result.converged
    repricing_valuation = _build_valuation(option_type=OptionType.CALL, vol=result.implied_vol)
    repriced = repricing_valuation.present_value()
    assert np.isclose(repriced, target_price, atol=1.0e-7)


@pytest.mark.parametrize("method", [ImpliedVolMethod.BISECTION, ImpliedVolMethod.BRENTQ])
def test_implied_volatility_converges_near_upper_bound(method):
    """Solver should converge when true volatility is near the upper bound."""
    true_vol = 4.9
    pricing_valuation = _build_valuation(option_type=OptionType.PUT, vol=true_vol)
    target_price = pricing_valuation.present_value()

    solver_valuation = _build_valuation(option_type=OptionType.PUT, vol=0.2)
    result = implied_volatility(
        target_price,
        solver_valuation,
        method=method,
        vol_bounds=(1.0e-6, 5.0),
        tol=1.0e-8,
        max_iter=300,
    )
    assert result.converged
    assert np.isclose(result.implied_vol, true_vol, rtol=2.0e-2, atol=1.0e-3)


def test_implied_volatility_rejects_non_positive_target_price():
    """Zero target price is rejected by validation/no-arbitrage guards."""
    valuation = _build_valuation(option_type=OptionType.CALL, vol=0.2)
    with pytest.raises(ValidationError, match="outside no-arbitrage bounds|non-negative"):
        implied_volatility(0.0, valuation)
