"""Tests for discrete dividend consistency across pricing engines."""

import datetime as dt

import numpy as np

from portfolio_analytics.enums import ExerciseType, OptionType, PricingMethod
from portfolio_analytics.market_environment import MarketData
from portfolio_analytics.rates import DiscountCurve
from portfolio_analytics.stochastic_processes import (
    GBMParams,
    GBMProcess,
    SimulationConfig,
)
from portfolio_analytics.tests.helpers import (
    market_data,
    underlying,
    spec,
    PRICING_DATE,
)
from portfolio_analytics.utils import pv_discrete_dividends
from portfolio_analytics.valuation import OptionValuation, UnderlyingData
from portfolio_analytics.valuation.params import BinomialParams, MonteCarloParams, PDEParams

# Scenario: Hull-style discrete dividend case
_SPOT = 52.0
_STRIKE = 50.0
_VOL = 0.4
_RATE = 0.1
_MATURITY = PRICING_DATE + dt.timedelta(days=365)
_DIVS = [
    (PRICING_DATE + dt.timedelta(days=90), 0.5),
    (PRICING_DATE + dt.timedelta(days=270), 0.5),
]


def _build_case():
    md = market_data(rate=_RATE, maturity=_MATURITY)
    sp = spec(OptionType.PUT, ExerciseType.EUROPEAN, strike=_STRIKE, maturity=_MATURITY)
    ud = underlying(spot=_SPOT, vol=_VOL, rate=_RATE, discrete_dividends=_DIVS, maturity=_MATURITY)

    sim_config = SimulationConfig(paths=200_000, frequency="W", end_date=_MATURITY)
    gbm_params = GBMParams(initial_value=_SPOT, volatility=_VOL, discrete_dividends=_DIVS)
    gbm = GBMProcess(md, gbm_params, sim_config)

    return md, sp, ud, gbm


def test_discrete_dividend_engine_consistency():
    md, sp, ud, gbm = _build_case()

    # MC vs PDE (jump model vs PDE jump condition)
    mc_val = OptionValuation(
        gbm, sp, PricingMethod.MONTE_CARLO, params=MonteCarloParams(random_seed=42)
    )
    mc_pv = mc_val.present_value()

    div_dates = [_DIVS[0][0], _DIVS[1][0]]
    assert all(div_date in mc_val.underlying.observation_dates for div_date in div_dates)
    assert all(div_date in mc_val.underlying.time_grid for div_date in div_dates)
    pde_val = OptionValuation(
        ud, sp, PricingMethod.PDE_FD, params=PDEParams(spot_steps=140, time_steps=140)
    )
    pde_pv = pde_val.present_value()

    assert np.isclose(mc_pv, pde_pv, rtol=0.01)

    # Binomial vs BSM (prepaid-forward adjustment)
    bsm_pv = OptionValuation(ud, sp, PricingMethod.BSM).present_value()
    binom_pv = OptionValuation(
        ud, sp, PricingMethod.BINOMIAL, params=BinomialParams(num_steps=1500)
    ).present_value()

    assert np.isclose(bsm_pv, binom_pv, rtol=0.01)

    # Volatility-adjusted BSM/Binomial should be closer to MC/PDE
    pv_divs = pv_discrete_dividends(
        dividends=_DIVS,
        curve_date=md.pricing_date,
        end_date=sp.maturity,
        discount_curve=md.discount_curve,
    )
    vol_multiplier = ud.initial_value / (ud.initial_value - pv_divs)
    adjusted_ud = ud.replace(volatility=ud.volatility * vol_multiplier)

    bsm_adj = OptionValuation(adjusted_ud, sp, PricingMethod.BSM).present_value()
    binom_adj = OptionValuation(
        adjusted_ud, sp, PricingMethod.BINOMIAL, params=BinomialParams(num_steps=1500)
    ).present_value()

    assert np.isclose(mc_pv, bsm_adj, rtol=0.02)
    assert np.isclose(pde_pv, binom_adj, rtol=0.02)


def test_binomial_accepts_discrete_dividends_with_nonflat_curve():
    nonflat_curve = DiscountCurve(
        times=np.array([0.0, 0.5, 1.0]),
        dfs=np.array([1.0, np.exp(-0.03 * 0.5), np.exp(-0.06 * 1.0)]),
    )
    md = MarketData(PRICING_DATE, nonflat_curve, currency="USD")
    ud = UnderlyingData(
        initial_value=_SPOT,
        volatility=_VOL,
        market_data=md,
        discrete_dividends=_DIVS,
    )
    sp = spec(OptionType.PUT, ExerciseType.EUROPEAN, strike=_STRIKE, maturity=_MATURITY)

    OptionValuation(
        ud, sp, PricingMethod.BINOMIAL, params=BinomialParams(num_steps=400)
    ).present_value()
