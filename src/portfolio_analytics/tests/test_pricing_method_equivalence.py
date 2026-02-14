"""Equivalence tests across pricing methods (PDE FD, BSM, Binomial, MC)."""

import datetime as dt

import numpy as np
import pytest

from portfolio_analytics.enums import DayCountConvention, ExerciseType, OptionType, PricingMethod
from portfolio_analytics.market_environment import MarketData
from portfolio_analytics.rates import ConstantShortRate
from portfolio_analytics.stochastic_processes import (
    GBMParams,
    GeometricBrownianMotion,
    SimulationConfig,
)
from portfolio_analytics.utils import pv_discrete_dividends
from portfolio_analytics.valuation import OptionSpec, OptionValuation, UnderlyingPricingData
from portfolio_analytics.valuation.params import BinomialParams, MonteCarloParams, PDEParams


PRICING_DATE = dt.datetime(2025, 1, 1)
MATURITY = PRICING_DATE + dt.timedelta(days=365)
RISK_FREE = 0.05
VOL = 0.2
CURRENCY = "USD"

PDE_CFG = PDEParams(spot_steps=140, time_steps=140, max_iter=20_000)
MC_CFG_EU = MonteCarloParams(random_seed=42)
MC_CFG_AM = MonteCarloParams(random_seed=42, deg=3)


def _market_data() -> MarketData:
    return MarketData(PRICING_DATE, ConstantShortRate("r", RISK_FREE), currency=CURRENCY)


def _underlying(
    *,
    spot: float,
    dividend_yield: float = 0.0,
    discrete_dividends: list[tuple[dt.datetime, float]] | None = None,
) -> UnderlyingPricingData:
    return UnderlyingPricingData(
        initial_value=spot,
        volatility=VOL,
        market_data=_market_data(),
        dividend_yield=dividend_yield,
        discrete_dividends=discrete_dividends,
    )


def _gbm(
    *,
    spot: float,
    dividend_yield: float = 0.0,
    discrete_dividends: list[tuple[dt.datetime, float]] | None = None,
    paths: int = 200_000,
) -> GeometricBrownianMotion:
    sim_config = SimulationConfig(
        paths=paths,
        day_count_convention=DayCountConvention.ACT_365F,
        frequency="W",
        end_date=MATURITY,
    )
    gbm_params = GBMParams(
        initial_value=spot,
        volatility=VOL,
        dividend_yield=dividend_yield,
        discrete_dividends=discrete_dividends,
    )
    return GeometricBrownianMotion("gbm", _market_data(), gbm_params, sim_config)


def _spec(*, strike: float, option_type: OptionType, exercise_type: ExerciseType) -> OptionSpec:
    return OptionSpec(
        option_type=option_type,
        exercise_type=exercise_type,
        strike=strike,
        maturity=MATURITY,
        currency=CURRENCY,
    )


@pytest.mark.parametrize(
    "spot,strike,option_type",
    [
        (90.0, 100.0, OptionType.CALL),
        (110.0, 100.0, OptionType.PUT),
    ],
)
@pytest.mark.parametrize("dividend_yield", [0.0, 0.03])
def test_pde_fd_european_close_to_bsm(spot, strike, option_type, dividend_yield):
    """PDE FD European should match BSM for no/continuous dividends."""
    ud = _underlying(spot=spot, dividend_yield=dividend_yield)
    spec = _spec(strike=strike, option_type=option_type, exercise_type=ExerciseType.EUROPEAN)

    pde_pv = OptionValuation(
        "pde_eu", ud, spec, PricingMethod.PDE_FD, params=PDE_CFG
    ).present_value()
    bsm_pv = OptionValuation("bsm_eu", ud, spec, PricingMethod.BSM).present_value()

    assert np.isclose(pde_pv, bsm_pv, rtol=0.02)


@pytest.mark.parametrize(
    "spot,strike,option_type",
    [
        (90.0, 100.0, OptionType.CALL),
        (110.0, 100.0, OptionType.PUT),
    ],
)
@pytest.mark.parametrize("dividend_yield", [0.0, 0.03])
def test_pde_fd_american_close_to_mc(spot, strike, option_type, dividend_yield):
    """PDE FD American should match MC (LSM) for no/continuous dividends."""
    ud = _underlying(spot=spot, dividend_yield=dividend_yield)
    gbm = _gbm(spot=spot, dividend_yield=dividend_yield)
    spec = _spec(strike=strike, option_type=option_type, exercise_type=ExerciseType.AMERICAN)

    pde_pv = OptionValuation(
        "pde_am", ud, spec, PricingMethod.PDE_FD, params=PDE_CFG
    ).present_value()
    mc_pv = OptionValuation(
        "mc_am", gbm, spec, PricingMethod.MONTE_CARLO, params=MC_CFG_AM
    ).present_value()

    assert np.isclose(pde_pv, mc_pv, rtol=0.02)


def test_discrete_dividend_equivalence_across_methods():
    """Discrete divs: PDE/MC align; BSM/Binomial align; vol-adjusted BSM/Binomial near PDE/MC."""
    spot = 52.0
    strike = 50.0
    divs = [
        (PRICING_DATE + dt.timedelta(days=90), 0.5),
        (PRICING_DATE + dt.timedelta(days=270), 0.5),
    ]

    ud = _underlying(spot=spot, discrete_dividends=divs)
    gbm = _gbm(spot=spot, discrete_dividends=divs, paths=200_000)

    spec_eu = _spec(strike=strike, option_type=OptionType.PUT, exercise_type=ExerciseType.EUROPEAN)

    pde_pv = OptionValuation(
        "pde_div", ud, spec_eu, PricingMethod.PDE_FD, params=PDE_CFG
    ).present_value()
    mc_pv = OptionValuation(
        "mc_div", gbm, spec_eu, PricingMethod.MONTE_CARLO, params=MC_CFG_EU
    ).present_value()

    assert np.isclose(pde_pv, mc_pv, rtol=0.02)

    bsm_pv = OptionValuation("bsm_div", ud, spec_eu, PricingMethod.BSM).present_value()
    binom_pv = OptionValuation(
        "binom_div",
        ud,
        spec_eu,
        PricingMethod.BINOMIAL,
        params=BinomialParams(num_steps=1500),
    ).present_value()

    assert np.isclose(bsm_pv, binom_pv, rtol=0.02)

    pv_divs = pv_discrete_dividends(
        dividends=divs,
        pricing_date=ud.pricing_date,
        maturity=spec_eu.maturity,
        short_rate=ud.discount_curve.short_rate,
    )
    vol_multiplier = ud.initial_value / (ud.initial_value - pv_divs)
    adjusted_ud = ud.replace(volatility=ud.volatility * vol_multiplier)

    bsm_adj = OptionValuation("bsm_adj", adjusted_ud, spec_eu, PricingMethod.BSM).present_value()
    binom_adj = OptionValuation(
        "binom_adj",
        adjusted_ud,
        spec_eu,
        PricingMethod.BINOMIAL,
        params=BinomialParams(num_steps=1500),
    ).present_value()

    assert np.isclose(bsm_adj, binom_adj, rtol=0.02)
    assert np.isclose(pde_pv, bsm_adj, rtol=0.02)
    assert np.isclose(mc_pv, binom_adj, rtol=0.02)


@pytest.mark.parametrize(
    "spot,strike",
    [
        (90.0, 100.0),
        (110.0, 100.0),
    ],
)
def test_discrete_dividend_american_matches_mc(spot, strike):
    """American discrete dividend: PDE FD should align with MC."""
    divs = [
        (PRICING_DATE + dt.timedelta(days=120), 0.6),
        (PRICING_DATE + dt.timedelta(days=240), 0.6),
    ]
    ud = _underlying(spot=spot, discrete_dividends=divs)
    gbm = _gbm(spot=spot, discrete_dividends=divs, paths=60_000)
    spec_am = _spec(strike=strike, option_type=OptionType.PUT, exercise_type=ExerciseType.AMERICAN)

    pde_pv = OptionValuation(
        "pde_am_div", ud, spec_am, PricingMethod.PDE_FD, params=PDE_CFG
    ).present_value()
    mc_pv = OptionValuation(
        "mc_am_div", gbm, spec_am, PricingMethod.MONTE_CARLO, params=MC_CFG_AM
    ).present_value()

    assert np.isclose(pde_pv, mc_pv, rtol=0.02)
