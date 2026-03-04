"""Compare Greeks between portfolio_analytics and QuantLib for reference.

Sections
--------
1. European BSM analytical Greeks vs QuantLib AnalyticEuropeanEngine
2. European binomial tree Greeks vs QuantLib analytical
3. American PDE numerical Greeks vs QuantLib FdBlackScholesVanillaEngine
4. European MC pathwise Greeks vs QuantLib analytical
"""

from __future__ import annotations

import datetime as dt
import logging
from typing import TYPE_CHECKING

import numpy as np
import pytest

from portfolio_analytics.enums import (
    ExerciseType,
    OptionType,
    PricingMethod,
)
from portfolio_analytics.market_environment import MarketData
from portfolio_analytics.rates import DiscountCurve
from portfolio_analytics.tests.helpers import flat_curve
from portfolio_analytics.valuation import (
    OptionSpec,
    OptionValuation,
    UnderlyingPricingData,
)
from portfolio_analytics.stochastic_processes import (
    GBMParams,
    GBMProcess,
    SimulationConfig,
)
from portfolio_analytics.valuation.params import (
    BinomialParams,
    MonteCarloParams,
    PDEParams,
)

if TYPE_CHECKING:
    import QuantLib as ql_typing

ql = pytest.importorskip("QuantLib")

logger = logging.getLogger(__name__)

# ── Shared constants ────────────────────────────────────────────────────

PRICING_DATE = dt.datetime(2025, 1, 1)
MATURITY = PRICING_DATE + dt.timedelta(days=365)
RISK_FREE = 0.05
VOL = 0.20
CURRENCY = "USD"

BINOM_CFG = BinomialParams(num_steps=1500)
PDE_CFG = PDEParams(spot_steps=200, time_steps=200, max_iter=20_000)
MC_CFG = MonteCarloParams(random_seed=42)

# ── Portfolio-analytics helpers ─────────────────────────────────────────


def _market_data() -> MarketData:
    return MarketData(
        PRICING_DATE,
        flat_curve(PRICING_DATE, MATURITY, RISK_FREE),
        currency=CURRENCY,
    )


def _spec(
    *,
    strike: float,
    option_type: OptionType,
    exercise_type: ExerciseType = ExerciseType.EUROPEAN,
) -> OptionSpec:
    return OptionSpec(
        option_type=option_type,
        exercise_type=exercise_type,
        strike=strike,
        maturity=MATURITY,
        currency=CURRENCY,
    )


def _underlying(
    *,
    spot: float,
    dividend_curve: DiscountCurve | None = None,
) -> UnderlyingPricingData:
    return UnderlyingPricingData(
        initial_value=spot,
        volatility=VOL,
        market_data=_market_data(),
        dividend_curve=dividend_curve,
    )


def _gbm(
    *,
    spot: float,
    dividend_curve: DiscountCurve | None = None,
    paths: int = 500_000,
) -> GBMProcess:
    return GBMProcess(
        _market_data(),
        GBMParams(
            initial_value=spot,
            volatility=VOL,
            dividend_curve=dividend_curve,
        ),
        SimulationConfig(
            paths=paths,
            frequency="W",
            end_date=MATURITY,
        ),
    )


# ── QuantLib helpers ────────────────────────────────────────────────────


def _ql_setup() -> "ql_typing.Date":
    eval_date = ql.Date(PRICING_DATE.day, PRICING_DATE.month, PRICING_DATE.year)
    ql.Settings.instance().evaluationDate = eval_date
    return eval_date


def _ql_process(
    eval_date: "ql_typing.Date",
    *,
    spot: float,
    dividend_yield: float = 0.0,
) -> "ql_typing.BlackScholesMertonProcess":
    """BSM process with flat rate, dividend yield, and vol."""
    return ql.BlackScholesMertonProcess(
        ql.QuoteHandle(ql.SimpleQuote(spot)),
        ql.YieldTermStructureHandle(ql.FlatForward(eval_date, dividend_yield, ql.Actual365Fixed())),
        ql.YieldTermStructureHandle(ql.FlatForward(eval_date, RISK_FREE, ql.Actual365Fixed())),
        ql.BlackVolTermStructureHandle(
            ql.BlackConstantVol(eval_date, ql.TARGET(), VOL, ql.Actual365Fixed())
        ),
    )


def _ql_european_option(
    *,
    spot: float,
    strike: float,
    option_type: OptionType,
    dividend_yield: float = 0.0,
) -> "ql_typing.VanillaOption":
    """QuantLib European with AnalyticEuropeanEngine."""
    eval_date = _ql_setup()
    ql_maturity = ql.Date(MATURITY.day, MATURITY.month, MATURITY.year)
    ql_type = ql.Option.Put if option_type is OptionType.PUT else ql.Option.Call
    option = ql.VanillaOption(
        ql.PlainVanillaPayoff(ql_type, strike),
        ql.EuropeanExercise(ql_maturity),
    )
    process = _ql_process(eval_date, spot=spot, dividend_yield=dividend_yield)
    option.setPricingEngine(ql.AnalyticEuropeanEngine(process))
    return option


def _ql_american_fd_option(
    *,
    spot: float,
    strike: float,
    option_type: OptionType,
    dividend_yield: float = 0.0,
    grid_points: int = 200,
    time_steps: int = 400,
) -> "ql_typing.VanillaOption":
    """QuantLib American with FdBlackScholesVanillaEngine."""
    eval_date = _ql_setup()
    ql_maturity = ql.Date(MATURITY.day, MATURITY.month, MATURITY.year)
    ql_type = ql.Option.Put if option_type is OptionType.PUT else ql.Option.Call
    option = ql.VanillaOption(
        ql.PlainVanillaPayoff(ql_type, strike),
        ql.AmericanExercise(eval_date, ql_maturity),
    )
    process = _ql_process(eval_date, spot=spot, dividend_yield=dividend_yield)
    engine = ql.FdBlackScholesVanillaEngine(
        process,
        ql.DividendVector([], []),
        grid_points,
        time_steps,
    )
    option.setPricingEngine(engine)
    return option


# Convention conversions (QuantLib → portfolio_analytics):
#   vega:  QL returns d(V)/d(σ);  PA returns d(V)/d(σ)/100  (per 1 vol-pt)
#   theta: QL returns d(V)/d(t) per year; PA returns per calendar day (/365)
#   rho:   QL returns d(V)/d(r);  PA returns d(V)/d(r)/100 (per 1% rate)


# ═══════════════════════════════════════════════════════════════════════
# 1. European BSM analytical Greeks vs QuantLib AnalyticEuropeanEngine
# ═══════════════════════════════════════════════════════════════════════

_EU_CASES = [
    (100.0, 100.0, OptionType.CALL),  # ATM call
    (100.0, 100.0, OptionType.PUT),  # ATM put
    (90.0, 100.0, OptionType.CALL),  # OTM call
    (110.0, 100.0, OptionType.PUT),  # OTM put
]


@pytest.mark.parametrize("spot,strike,option_type", _EU_CASES)
@pytest.mark.parametrize("dividend_yield", [0.0, 0.03])
def test_european_bsm_greeks_vs_quantlib(spot, strike, option_type, dividend_yield):
    """BSM analytical delta/gamma/vega/theta/rho match QuantLib."""
    q_curve = flat_curve(PRICING_DATE, MATURITY, dividend_yield) if dividend_yield else None
    ov = OptionValuation(
        _underlying(spot=spot, dividend_curve=q_curve),
        _spec(strike=strike, option_type=option_type),
        PricingMethod.BSM,
    )
    ql_opt = _ql_european_option(
        spot=spot,
        strike=strike,
        option_type=option_type,
        dividend_yield=dividend_yield,
    )

    pairs = {
        "delta": (ov.delta(), ql_opt.delta()),
        "gamma": (ov.gamma(), ql_opt.gamma()),
        "vega": (ov.vega(), ql_opt.vega() / 100),
        "theta": (ov.theta(), ql_opt.theta() / 365),
        "rho": (ov.rho(), ql_opt.rho() / 100),
    }

    for name, (pa_val, ql_val) in pairs.items():
        logger.info(
            "BSM %s %s S=%.0f K=%.0f q=%.2f | PA=%.8f QL=%.8f",
            name,
            option_type.value,
            spot,
            strike,
            dividend_yield,
            pa_val,
            ql_val,
        )
        assert np.isclose(pa_val, ql_val, rtol=1e-4), f"{name}: PA {pa_val:.8f} vs QL {ql_val:.8f}"


# ═══════════════════════════════════════════════════════════════════════
# 2. European binomial tree Greeks vs QuantLib analytical
# ═══════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize("spot,strike,option_type", _EU_CASES)
@pytest.mark.parametrize("dividend_yield", [0.0, 0.03])
def test_european_binomial_greeks_vs_quantlib(spot, strike, option_type, dividend_yield):
    """Binomial tree delta/gamma/theta converge to QuantLib analytical."""
    q_curve = flat_curve(PRICING_DATE, MATURITY, dividend_yield) if dividend_yield else None
    ov = OptionValuation(
        _underlying(spot=spot, dividend_curve=q_curve),
        _spec(strike=strike, option_type=option_type),
        PricingMethod.BINOMIAL,
        params=BINOM_CFG,
    )
    ql_opt = _ql_european_option(
        spot=spot,
        strike=strike,
        option_type=option_type,
        dividend_yield=dividend_yield,
    )

    pairs = {
        "delta": (ov.delta(), ql_opt.delta(), 0.005),
        "gamma": (ov.gamma(), ql_opt.gamma(), 0.02),
        "theta": (ov.theta(), ql_opt.theta() / 365, 0.02),
    }

    for name, (pa_val, ql_val, tol) in pairs.items():
        logger.info(
            "Binom %s %s S=%.0f K=%.0f q=%.2f | PA=%.6f QL=%.6f",
            name,
            option_type.value,
            spot,
            strike,
            dividend_yield,
            pa_val,
            ql_val,
        )
        assert np.isclose(pa_val, ql_val, rtol=tol), f"{name}: PA {pa_val:.6f} vs QL {ql_val:.6f}"


# ═══════════════════════════════════════════════════════════════════════
# 3. American PDE numerical Greeks vs QuantLib FdBlackScholesVanillaEngine
# ═══════════════════════════════════════════════════════════════════════

_AM_CASES = [
    (110.0, 100.0, OptionType.CALL),  # ITM call
    (90.0, 100.0, OptionType.CALL),  # OTM call
    (90.0, 100.0, OptionType.PUT),  # ITM put
    (110.0, 100.0, OptionType.PUT),  # OTM put
]


@pytest.mark.parametrize("spot,strike,option_type", _AM_CASES)
@pytest.mark.parametrize("dividend_yield", [0.0, 0.03])
def test_american_pde_delta_vs_quantlib(spot, strike, option_type, dividend_yield):
    """PDE bump-and-revalue delta aligns with QuantLib FD grid delta."""
    q_curve = flat_curve(PRICING_DATE, MATURITY, dividend_yield) if dividend_yield else None
    ov = OptionValuation(
        _underlying(spot=spot, dividend_curve=q_curve),
        _spec(
            strike=strike,
            option_type=option_type,
            exercise_type=ExerciseType.AMERICAN,
        ),
        PricingMethod.PDE_FD,
        params=PDE_CFG,
    )
    ql_opt = _ql_american_fd_option(
        spot=spot,
        strike=strike,
        option_type=option_type,
        dividend_yield=dividend_yield,
    )

    pa_delta = ov.delta()
    ql_delta = ql_opt.delta()
    logger.info(
        "PDE AM delta %s S=%.0f K=%.0f q=%.2f | PA=%.6f QL=%.6f",
        option_type.value,
        spot,
        strike,
        dividend_yield,
        pa_delta,
        ql_delta,
    )
    assert np.isclose(pa_delta, ql_delta, rtol=0.02), (
        f"delta: PA {pa_delta:.6f} vs QL {ql_delta:.6f}"
    )


@pytest.mark.parametrize("spot,strike,option_type", _AM_CASES)
@pytest.mark.parametrize("dividend_yield", [0.0, 0.03])
def test_american_pde_gamma_vs_quantlib(spot, strike, option_type, dividend_yield):
    """PDE grid gamma vs QuantLib FD grid gamma."""
    q_curve = flat_curve(PRICING_DATE, MATURITY, dividend_yield) if dividend_yield else None
    ov = OptionValuation(
        _underlying(spot=spot, dividend_curve=q_curve),
        _spec(
            strike=strike,
            option_type=option_type,
            exercise_type=ExerciseType.AMERICAN,
        ),
        PricingMethod.PDE_FD,
        params=PDE_CFG,
    )
    ql_opt = _ql_american_fd_option(
        spot=spot,
        strike=strike,
        option_type=option_type,
        dividend_yield=dividend_yield,
    )

    pa_gamma = ov.gamma()
    ql_gamma = ql_opt.gamma()
    logger.info(
        "PDE AM gamma %s S=%.0f K=%.0f q=%.2f | PA=%.6f QL=%.6f",
        option_type.value,
        spot,
        strike,
        dividend_yield,
        pa_gamma,
        ql_gamma,
    )
    assert np.isclose(pa_gamma, ql_gamma, rtol=0.05), (
        f"gamma: PA {pa_gamma:.6f} vs QL {ql_gamma:.6f}"
    )


# ═══════════════════════════════════════════════════════════════════════
# 4. European MC pathwise Greeks vs QuantLib analytical
# ═══════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize(
    "spot,strike,option_type",
    [
        (100.0, 100.0, OptionType.CALL),  # ATM call
        (100.0, 100.0, OptionType.PUT),  # ATM put
        (90.0, 100.0, OptionType.CALL),  # OTM call
    ],
)
def test_european_mc_pathwise_greeks_vs_quantlib(spot, strike, option_type):
    """MC pathwise delta/vega and pathwise-FD gamma match QuantLib analytical."""
    ov = OptionValuation(
        _gbm(spot=spot, paths=500_000),
        _spec(strike=strike, option_type=option_type),
        PricingMethod.MONTE_CARLO,
        params=MC_CFG,
    )
    ql_opt = _ql_european_option(
        spot=spot,
        strike=strike,
        option_type=option_type,
    )

    pairs = {
        "delta": (ov.delta(), ql_opt.delta(), 0.03),
        "gamma": (ov.gamma(), ql_opt.gamma(), 0.10),
        "vega": (ov.vega(), ql_opt.vega() / 100, 0.05),
    }

    for name, (pa_val, ql_val, tol) in pairs.items():
        logger.info(
            "MC pw %s %s S=%.0f K=%.0f | PA=%.6f QL=%.6f",
            name,
            option_type.value,
            spot,
            strike,
            pa_val,
            ql_val,
        )
        assert np.isclose(pa_val, ql_val, rtol=tol), f"{name}: PA {pa_val:.6f} vs QL {ql_val:.6f}"
