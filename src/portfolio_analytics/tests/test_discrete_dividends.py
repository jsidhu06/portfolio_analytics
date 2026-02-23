"""Tests for discrete dividend consistency across pricing engines."""

import datetime as dt

import numpy as np

from portfolio_analytics.enums import ExerciseType, OptionType, PricingMethod
from portfolio_analytics.enums import DayCountConvention
from portfolio_analytics.market_environment import MarketData
from portfolio_analytics.rates import DiscountCurve
from portfolio_analytics.tests.helpers import flat_curve
from portfolio_analytics.stochastic_processes import (
    GBMParams,
    GBMProcess,
    SimulationConfig,
)
from portfolio_analytics.utils import pv_discrete_dividends
from portfolio_analytics.valuation import OptionSpec, OptionValuation, UnderlyingPricingData
from portfolio_analytics.valuation.params import BinomialParams, MonteCarloParams, PDEParams


def _build_case():
    pricing_date = dt.datetime(2025, 1, 1)
    maturity = pricing_date + dt.timedelta(days=365)
    r = 0.1
    spot = 52.0
    strike = 50.0
    vol = 0.4

    curve = flat_curve(pricing_date, maturity, r, name="csr")
    market_data = MarketData(pricing_date, curve, currency="USD")

    divs = [
        (pricing_date + dt.timedelta(days=90), 0.5),
        (pricing_date + dt.timedelta(days=270), 0.5),
    ]

    spec = OptionSpec(
        option_type=OptionType.PUT,
        exercise_type=ExerciseType.EUROPEAN,
        strike=strike,
        maturity=maturity,
        currency="USD",
    )

    underlying = UnderlyingPricingData(
        initial_value=spot,
        volatility=vol,
        market_data=market_data,
        discrete_dividends=divs,
    )

    sim_config = SimulationConfig(
        paths=200_000,
        day_count_convention=DayCountConvention.ACT_365F,
        frequency="W",
        end_date=maturity,
    )
    gbm_params = GBMParams(initial_value=spot, volatility=vol, discrete_dividends=divs)
    gbm = GBMProcess("gbm_div", market_data, gbm_params, sim_config)

    return market_data, spec, underlying, gbm, divs


def test_discrete_dividend_engine_consistency():
    market_data, spec, underlying, gbm, divs = _build_case()

    # MC vs PDE (jump model vs PDE jump condition)
    mc_val = OptionValuation(
        "put_mc", gbm, spec, PricingMethod.MONTE_CARLO, params=MonteCarloParams(random_seed=42)
    )
    mc_pv = mc_val.present_value()

    div_dates = [divs[0][0], divs[1][0]]
    assert all(div_date in mc_val.underlying.special_dates for div_date in div_dates)
    assert all(div_date in mc_val.underlying.time_grid for div_date in div_dates)
    pde_val = OptionValuation(
        "put_pde",
        underlying,
        spec,
        PricingMethod.PDE_FD,
        params=PDEParams(spot_steps=140, time_steps=140),
    )
    pde_pv = pde_val.present_value()

    assert np.isclose(mc_pv, pde_pv, rtol=0.01)

    # Binomial vs BSM (prepaid-forward adjustment)
    bsm_val = OptionValuation("put_bsm", underlying, spec, PricingMethod.BSM)
    binom_val = OptionValuation(
        "put_binom", underlying, spec, PricingMethod.BINOMIAL, params=BinomialParams(num_steps=1500)
    )

    bsm_pv = bsm_val.present_value()
    binom_pv = binom_val.present_value()

    assert np.isclose(bsm_pv, binom_pv, rtol=0.01)

    # Volatility-adjusted BSM/Binomial should be closer to MC/PDE
    pv_divs = pv_discrete_dividends(
        dividends=divs,
        curve_date=market_data.pricing_date,
        end_date=spec.maturity,
        discount_curve=market_data.discount_curve,
    )
    vol_multiplier = underlying.initial_value / (underlying.initial_value - pv_divs)
    adjusted_underlying = underlying.replace(volatility=underlying.volatility * vol_multiplier)

    bsm_adj = OptionValuation(
        "put_bsm_adj", adjusted_underlying, spec, PricingMethod.BSM
    ).present_value()
    binom_adj = OptionValuation(
        "put_binom_adj",
        adjusted_underlying,
        spec,
        PricingMethod.BINOMIAL,
        params=BinomialParams(num_steps=1500),
    ).present_value()

    assert np.isclose(mc_pv, bsm_adj, rtol=0.02)
    assert np.isclose(pde_pv, binom_adj, rtol=0.02)


def test_binomial_accepts_discrete_dividends_with_nonflat_curve():
    pricing_date = dt.datetime(2025, 1, 1)
    maturity = pricing_date + dt.timedelta(days=365)
    nonflat_curve = DiscountCurve(
        name="nonflat",
        times=np.array([0.0, 0.5, 1.0]),
        dfs=np.array([1.0, np.exp(-0.03 * 0.5), np.exp(-0.06 * 1.0)]),
    )
    market_data = MarketData(pricing_date, nonflat_curve, currency="USD")
    divs = [
        (pricing_date + dt.timedelta(days=90), 0.5),
        (pricing_date + dt.timedelta(days=270), 0.5),
    ]
    underlying = UnderlyingPricingData(
        initial_value=52.0,
        volatility=0.4,
        market_data=market_data,
        discrete_dividends=divs,
    )
    spec = OptionSpec(
        option_type=OptionType.PUT,
        exercise_type=ExerciseType.EUROPEAN,
        strike=50.0,
        maturity=maturity,
        currency="USD",
    )

    OptionValuation(
        "put_binom_nonflat",
        underlying,
        spec,
        PricingMethod.BINOMIAL,
        params=BinomialParams(num_steps=400),
    ).present_value()
