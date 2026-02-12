"""Tests for Asian option pricing consistency across methods."""

import datetime as dt
import logging

import numpy as np
import pytest

from portfolio_analytics.enums import AsianAveraging, DayCountConvention, OptionType, PricingMethod
from portfolio_analytics.market_environment import MarketData
from portfolio_analytics.rates import ConstantShortRate
from portfolio_analytics.stochastic_processes import (
    GBMParams,
    GeometricBrownianMotion,
    SimulationConfig,
)
from portfolio_analytics.valuation import OptionValuation, UnderlyingPricingData
from portfolio_analytics.valuation.core import AsianOptionSpec
from portfolio_analytics.valuation.params import BinomialParams, MonteCarloParams


logger = logging.getLogger(__name__)


PRICING_DATE = dt.datetime(2025, 1, 1)
CURRENCY = "USD"
MC_PATHS = 200_000
MC_SEED = 42
NUM_STEPS = 60
ASIAN_TREE_AVERAGES = 100


def _market_data(short_rate: float) -> MarketData:
    return MarketData(PRICING_DATE, ConstantShortRate("r", short_rate), currency=CURRENCY)


def _gbm_underlying(
    *,
    spot: float,
    vol: float,
    short_rate: float,
    dividend_yield: float,
    maturity: dt.datetime,
    paths: int,
    num_steps: int,
) -> GeometricBrownianMotion:
    sim_config = SimulationConfig(
        paths=paths,
        day_count_convention=DayCountConvention.ACT_365F,
        num_steps=num_steps,
        end_date=maturity,
    )
    gbm_params = GBMParams(initial_value=spot, volatility=vol, dividend_yield=dividend_yield)
    return GeometricBrownianMotion("gbm", _market_data(short_rate), gbm_params, sim_config)


def _binomial_underlying(
    *, spot: float, vol: float, short_rate: float, dividend_yield: float
) -> UnderlyingPricingData:
    return UnderlyingPricingData(
        initial_value=spot,
        volatility=vol,
        market_data=_market_data(short_rate),
        dividend_yield=dividend_yield,
    )


def _asian_spec(*, strike: float, maturity: dt.datetime, call_put: OptionType) -> AsianOptionSpec:
    return AsianOptionSpec(
        averaging=AsianAveraging.ARITHMETIC,
        call_put=call_put,
        strike=strike,
        maturity=maturity,
        currency=CURRENCY,
    )


@pytest.mark.parametrize(
    "spot,strike,vol,short_rate,dividend_yield,days,call_put",
    [
        (100.0, 100.0, 0.2, 0.03, 0.0, 365, OptionType.CALL),
        (100.0, 100.0, 0.2, 0.03, 0.02, 365, OptionType.CALL),
        (110.0, 100.0, 0.25, 0.01, 0.0, 270, OptionType.PUT),
        (110.0, 100.0, 0.25, 0.01, 0.015, 270, OptionType.PUT),
        (95.0, 90.0, 0.3, 0.05, 0.0, 540, OptionType.CALL),
        (95.0, 90.0, 0.3, 0.05, 0.025, 540, OptionType.CALL),
        (105.0, 110.0, 0.18, 0.02, 0.0, 180, OptionType.PUT),
        (105.0, 110.0, 0.18, 0.02, 0.01, 180, OptionType.PUT),
    ],
)
def test_asian_binomial_hull_close_to_mc(
    spot, strike, vol, short_rate, dividend_yield, days, call_put
):
    maturity = PRICING_DATE + dt.timedelta(days=days)
    spec = _asian_spec(strike=strike, maturity=maturity, call_put=call_put)

    mc_underlying = _gbm_underlying(
        spot=spot,
        vol=vol,
        short_rate=short_rate,
        dividend_yield=dividend_yield,
        maturity=maturity,
        paths=MC_PATHS,
        num_steps=NUM_STEPS,
    )
    mc_pv = OptionValuation(
        "asian_mc", mc_underlying, spec, PricingMethod.MONTE_CARLO
    ).present_value(params=MonteCarloParams(random_seed=MC_SEED))

    binom_underlying = _binomial_underlying(
        spot=spot, vol=vol, short_rate=short_rate, dividend_yield=dividend_yield
    )

    binom_mc_pv = OptionValuation(
        "asian_mc", binom_underlying, spec, PricingMethod.BINOMIAL
    ).present_value(
        params=BinomialParams(
            num_steps=NUM_STEPS * 2,
            mc_paths=MC_PATHS,
            random_seed=MC_SEED,
        )
    )

    hull_pv = OptionValuation(
        "asian_hull", binom_underlying, spec, PricingMethod.BINOMIAL
    ).present_value(
        params=BinomialParams(
            num_steps=NUM_STEPS,
            asian_tree_averages=ASIAN_TREE_AVERAGES,
        )
    )

    logger.info(
        "Asian %s S=%.2f K=%.2f vol=%.2f r=%.2f q=%.2f days=%d\nMC=%.6f Hull=%.6f BinomMC=%.6f",
        call_put.value,
        spot,
        strike,
        vol,
        short_rate,
        dividend_yield,
        days,
        mc_pv,
        hull_pv,
        binom_mc_pv,
    )

    assert np.isclose(mc_pv, hull_pv, rtol=0.02)
    assert np.isclose(binom_mc_pv, hull_pv, rtol=0.02)
