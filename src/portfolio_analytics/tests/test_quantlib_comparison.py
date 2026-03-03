"""Compare PDE FD American pricing vs QuantLib for reference."""

from __future__ import annotations
import datetime as dt
import logging
from typing import TYPE_CHECKING, Sequence

import numpy as np
import pytest

from portfolio_analytics.enums import (
    AsianAveraging,
    ExerciseType,
    OptionType,
    PricingMethod,
)
from portfolio_analytics.market_environment import MarketData
from portfolio_analytics.rates import DiscountCurve
from portfolio_analytics.tests.helpers import flat_curve
from portfolio_analytics.valuation import (
    AsianOptionSpec,
    OptionSpec,
    OptionValuation,
    UnderlyingPricingData,
)
from portfolio_analytics.stochastic_processes import GBMParams, GBMProcess, SimulationConfig
from portfolio_analytics.valuation.params import BinomialParams, MonteCarloParams, PDEParams
from portfolio_analytics.utils import calculate_year_fraction

if TYPE_CHECKING:
    import QuantLib as ql_typing

ql = pytest.importorskip("QuantLib")


logger = logging.getLogger(__name__)


PRICING_DATE = dt.datetime(2025, 1, 1)
MATURITY = PRICING_DATE + dt.timedelta(days=365)
RISK_FREE = 0.1
VOL = 0.4
CURRENCY = "USD"

PDE_CFG = PDEParams(spot_steps=200, time_steps=200, max_iter=20_000)
BINOM_CFG = BinomialParams(num_steps=500)
MC_CFG = MonteCarloParams(random_seed=42, deg=3)


def _market_data(r_curve: DiscountCurve | None = None) -> MarketData:
    curve = r_curve if r_curve is not None else flat_curve(PRICING_DATE, MATURITY, RISK_FREE)
    return MarketData(PRICING_DATE, curve, currency=CURRENCY)


def _ql_curve_from_times(
    *,
    times: np.ndarray,
    dfs: np.ndarray,
) -> ql_typing.YieldTermStructureHandle:
    day_count = ql.Actual365Fixed()
    ql.Settings.instance().evaluationDate = ql.Date(
        PRICING_DATE.day, PRICING_DATE.month, PRICING_DATE.year
    )
    dates = [ql.Settings.instance().evaluationDate]
    for t in times[1:]:
        days = int(round(float(t) * 365.0))
        dates.append(ql.Settings.instance().evaluationDate + days)
    return ql.YieldTermStructureHandle(ql.DiscountCurve(dates, list(dfs), day_count))


def _spec(
    *,
    strike: float,
    option_type: OptionType,
    exercise_type: ExerciseType = ExerciseType.AMERICAN,
) -> OptionSpec:
    return OptionSpec(
        option_type=option_type,
        exercise_type=exercise_type,
        strike=strike,
        maturity=MATURITY,
        currency=CURRENCY,
    )


def _quantlib_dividend_schedule(
    discrete_dividends: Sequence[tuple[dt.datetime, float]] | None,
) -> ql_typing.DividendVector:
    if discrete_dividends:
        div_dates = []
        dividends = []
        for ex_date, amount in discrete_dividends:
            if PRICING_DATE <= ex_date <= MATURITY:
                div_dates.append(ql.Date(ex_date.day, ex_date.month, ex_date.year))
                dividends.append(float(amount))
        return ql.DividendVector(div_dates, dividends)
    return ql.DividendVector([], [])


def _quantlib_american_with_curves(
    *,
    spot: float,
    strike: float,
    option_type: OptionType,
    rf_curve: ql_typing.YieldTermStructureHandle,
    div_curve: ql_typing.YieldTermStructureHandle,
    discrete_dividends: Sequence[tuple[dt.datetime, float]] | None,
    grid_points: int = 200,
    time_steps: int = 400,
) -> float:
    ql.Settings.instance().evaluationDate = ql.Date(
        PRICING_DATE.day, PRICING_DATE.month, PRICING_DATE.year
    )

    ql_maturity = ql.Date(MATURITY.day, MATURITY.month, MATURITY.year)

    spot_handle = ql.QuoteHandle(ql.SimpleQuote(float(spot)))
    vol_handle = ql.BlackVolTermStructureHandle(
        ql.BlackConstantVol(
            ql.Settings.instance().evaluationDate,
            ql.TARGET(),
            float(VOL),
            ql.Actual365Fixed(),
        )
    )

    if option_type is OptionType.PUT:
        payoff = ql.PlainVanillaPayoff(ql.Option.Put, float(strike))
    else:
        payoff = ql.PlainVanillaPayoff(ql.Option.Call, float(strike))

    exercise = ql.AmericanExercise(ql.Settings.instance().evaluationDate, ql_maturity)
    option = ql.VanillaOption(payoff, exercise)

    dividend_schedule = _quantlib_dividend_schedule(discrete_dividends)

    process = ql.BlackScholesMertonProcess(spot_handle, div_curve, rf_curve, vol_handle)

    engine = ql.FdBlackScholesVanillaEngine(
        process,
        dividend_schedule,
        int(grid_points),
        int(time_steps),
    )
    option.setPricingEngine(engine)
    return float(option.NPV())


def _quantlib_american(
    *,
    spot: float,
    strike: float,
    option_type: OptionType,
    dividend_yield: float,
    discrete_dividends: Sequence[tuple[dt.datetime, float]] | None,
    grid_points: int = 200,
    time_steps: int = 400,
) -> float:
    ql.Settings.instance().evaluationDate = ql.Date(
        PRICING_DATE.day, PRICING_DATE.month, PRICING_DATE.year
    )
    rf_curve = ql.YieldTermStructureHandle(
        ql.FlatForward(
            ql.Settings.instance().evaluationDate,
            float(RISK_FREE),
            ql.Actual365Fixed(),
        )
    )
    div_curve = ql.YieldTermStructureHandle(
        ql.FlatForward(
            ql.Settings.instance().evaluationDate,
            float(dividend_yield),
            ql.Actual365Fixed(),
        )
    )
    return _quantlib_american_with_curves(
        spot=spot,
        strike=strike,
        option_type=option_type,
        rf_curve=rf_curve,
        div_curve=div_curve,
        discrete_dividends=discrete_dividends,
        grid_points=grid_points,
        time_steps=time_steps,
    )


def _quantlib_american_curves(
    *,
    spot: float,
    strike: float,
    option_type: OptionType,
    r_times: np.ndarray,
    r_dfs: np.ndarray,
    q_times: np.ndarray,
    q_dfs: np.ndarray,
    discrete_dividends: Sequence[tuple[dt.datetime, float]] | None = None,
    grid_points: int = 200,
    time_steps: int = 400,
) -> float:
    rf_curve = _ql_curve_from_times(times=r_times, dfs=r_dfs)
    div_curve = _ql_curve_from_times(times=q_times, dfs=q_dfs)
    return _quantlib_american_with_curves(
        spot=spot,
        strike=strike,
        option_type=option_type,
        rf_curve=rf_curve,
        div_curve=div_curve,
        discrete_dividends=discrete_dividends,
        grid_points=grid_points,
        time_steps=time_steps,
    )


def _pde_fd_american(
    *,
    spot: float,
    strike: float,
    option_type: OptionType,
    r_curve: DiscountCurve | None = None,
    dividend_curve: DiscountCurve | None = None,
    discrete_dividends: Sequence[tuple[dt.datetime, float]] | None = None,
) -> float:
    ud = UnderlyingPricingData(
        initial_value=spot,
        volatility=VOL,
        market_data=_market_data(r_curve),
        dividend_curve=dividend_curve,
        discrete_dividends=discrete_dividends,
    )
    spec = _spec(strike=strike, option_type=option_type)
    return OptionValuation(ud, spec, PricingMethod.PDE_FD, PDE_CFG).present_value()


@pytest.mark.parametrize(
    "spot,strike,option_type",
    [
        (52.0, 50.0, OptionType.PUT),
        (60.0, 55.0, OptionType.CALL),
    ],
)
def test_american_fd_vs_quantlib_no_div(spot, strike, option_type):
    pde = _pde_fd_american(
        spot=spot,
        strike=strike,
        option_type=option_type,
        dividend_curve=None,
        discrete_dividends=None,
    )
    ql_price = _quantlib_american(
        spot=spot,
        strike=strike,
        option_type=option_type,
        dividend_yield=0.0,
        discrete_dividends=None,
    )

    logger.info(
        "No-div American %s S=%.2f K=%.2f | PDE=%.6f QL=%.6f",
        option_type.value,
        spot,
        strike,
        pde,
        ql_price,
    )

    assert np.isclose(pde, ql_price, rtol=0.01)


@pytest.mark.parametrize(
    "spot,strike,option_type",
    [
        (52.0, 50.0, OptionType.PUT),
        (60.0, 55.0, OptionType.CALL),
    ],
)
def test_american_fd_vs_quantlib_continuous_div(spot, strike, option_type):
    q_curve = flat_curve(PRICING_DATE, MATURITY, 0.03)
    pde = _pde_fd_american(
        spot=spot,
        strike=strike,
        option_type=option_type,
        dividend_curve=q_curve,
        discrete_dividends=None,
    )
    ql_price = _quantlib_american(
        spot=spot,
        strike=strike,
        option_type=option_type,
        dividend_yield=0.03,
        discrete_dividends=None,
    )

    logger.info(
        "Cont-div American %s S=%.2f K=%.2f q=0.03 | PDE=%.6f QL=%.6f",
        option_type.value,
        spot,
        strike,
        pde,
        ql_price,
    )

    assert np.isclose(pde, ql_price, rtol=0.01)


@pytest.mark.parametrize("spot,strike,option_type", [(52.0, 50.0, OptionType.PUT)])
def test_american_fd_vs_quantlib_nonflat_rate_curve(spot, strike, option_type):
    times = np.array([0.0, 0.25, 0.5, 1.0])
    forwards = np.array([0.03, 0.05, 0.04])
    r_curve = DiscountCurve.from_forwards(times=times, forwards=forwards)

    pde = _pde_fd_american(
        spot=spot,
        strike=strike,
        option_type=option_type,
        r_curve=r_curve,
        dividend_curve=None,
    )
    ql_price = _quantlib_american_curves(
        spot=spot,
        strike=strike,
        option_type=option_type,
        r_times=r_curve.times,
        r_dfs=r_curve.dfs,
        q_times=np.array([0.0, 1.0]),
        q_dfs=np.array([1.0, 1.0]),
    )

    logger.info(
        "Forward-curve American %s S=%.2f K=%.2f | PDE=%.6f QL=%.6f",
        option_type.value,
        spot,
        strike,
        pde,
        ql_price,
    )

    assert np.isclose(pde, ql_price, rtol=0.01)


@pytest.mark.parametrize("spot,strike,option_type", [(60.0, 55.0, OptionType.CALL)])
def test_american_fd_vs_quantlib_flat_rate_with_dividend_curve(spot, strike, option_type):
    r_times = np.array([0.0, 1.0])
    r_forwards = np.array([0.04])
    r_curve = DiscountCurve.from_forwards(times=r_times, forwards=r_forwards)

    q_times = np.array([0.0, 0.25, 0.5, 1.0])
    q_forwards = np.array([0.00, 0.02, 0.04])
    q_curve = DiscountCurve.from_forwards(times=q_times, forwards=q_forwards)

    pde = _pde_fd_american(
        spot=spot,
        strike=strike,
        option_type=option_type,
        r_curve=r_curve,
        dividend_curve=q_curve,
    )
    ql_price = _quantlib_american_curves(
        spot=spot,
        strike=strike,
        option_type=option_type,
        r_times=r_curve.times,
        r_dfs=r_curve.dfs,
        q_times=q_curve.times,
        q_dfs=q_curve.dfs,
    )

    logger.info(
        "Dividend-curve American %s S=%.2f K=%.2f | PDE=%.6f QL=%.6f",
        option_type.value,
        spot,
        strike,
        pde,
        ql_price,
    )

    assert np.isclose(pde, ql_price, rtol=0.01)


@pytest.mark.parametrize(
    "spot,strike,option_type", [(52.0, 50.0, OptionType.PUT), (52.0, 50.0, OptionType.CALL)]
)
def test_american_fd_vs_quantlib_nonflat_rate_and_dividend_curves(spot, strike, option_type):
    r_times = np.array([0.0, 0.25, 0.5, 1.0])
    r_forwards = np.array([0.03, 0.05, 0.04])
    r_curve = DiscountCurve.from_forwards(times=r_times, forwards=r_forwards)

    q_times = np.array([0.0, 0.25, 0.5, 1.0])
    q_forwards = np.array([0.01, 0.02, 0.00])
    q_curve = DiscountCurve.from_forwards(times=q_times, forwards=q_forwards)

    pde = _pde_fd_american(
        spot=spot,
        strike=strike,
        option_type=option_type,
        r_curve=r_curve,
        dividend_curve=q_curve,
    )
    ql_price = _quantlib_american_curves(
        spot=spot,
        strike=strike,
        option_type=option_type,
        r_times=r_curve.times,
        r_dfs=r_curve.dfs,
        q_times=q_curve.times,
        q_dfs=q_curve.dfs,
    )

    logger.info(
        "Forward+dividend curves American %s S=%.2f K=%.2f | PDE=%.6f QL=%.6f",
        option_type.value,
        spot,
        strike,
        pde,
        ql_price,
    )

    assert np.isclose(pde, ql_price, rtol=0.01)


@pytest.mark.parametrize(
    "spot,strike,option_type", [(52.0, 50.0, OptionType.PUT), (52.0, 50.0, OptionType.CALL)]
)
def test_american_fd_vs_quantlib_nonflat_rate_with_discrete_divs(spot, strike, option_type):
    r_times = np.array([0.0, 0.25, 0.5, 1.0])
    r_forwards = np.array([0.03, 0.05, 0.04])
    r_curve = DiscountCurve.from_forwards(times=r_times, forwards=r_forwards)

    q_times = np.array([0.0, 1.0])
    q_dfs = np.array([1.0, 1.0])

    divs = [
        (PRICING_DATE + dt.timedelta(days=90), 0.5),
        (PRICING_DATE + dt.timedelta(days=270), 0.5),
    ]

    pde = _pde_fd_american(
        spot=spot,
        strike=strike,
        option_type=option_type,
        r_curve=r_curve,
        discrete_dividends=divs,
    )
    ql_price = _quantlib_american_curves(
        spot=spot,
        strike=strike,
        option_type=option_type,
        r_times=r_curve.times,
        r_dfs=r_curve.dfs,
        q_times=q_times,
        q_dfs=q_dfs,
        discrete_dividends=divs,
    )

    logger.info(
        "Forward-curve + discrete-div American %s S=%.2f K=%.2f | PDE=%.6f QL=%.6f",
        option_type.value,
        spot,
        strike,
        pde,
        ql_price,
    )

    assert np.isclose(pde, ql_price, rtol=0.01)


# ---------------------------------------------------------------------------
# Helpers for boundary-dividend comparison tests
# ---------------------------------------------------------------------------


def _quantlib_european(
    *,
    spot: float,
    strike: float,
    option_type: OptionType,
    discrete_dividends: Sequence[tuple[dt.datetime, float]] | None,
) -> float:
    """QuantLib European price using AnalyticDividendEuropeanEngine (Merton approach)."""
    eval_date = ql.Date(PRICING_DATE.day, PRICING_DATE.month, PRICING_DATE.year)
    ql.Settings.instance().evaluationDate = eval_date
    ql_maturity = ql.Date(MATURITY.day, MATURITY.month, MATURITY.year)

    rf_curve = ql.YieldTermStructureHandle(
        ql.FlatForward(eval_date, float(RISK_FREE), ql.Actual365Fixed())
    )
    div_curve = ql.YieldTermStructureHandle(ql.FlatForward(eval_date, 0.0, ql.Actual365Fixed()))
    spot_handle = ql.QuoteHandle(ql.SimpleQuote(float(spot)))
    vol_handle = ql.BlackVolTermStructureHandle(
        ql.BlackConstantVol(eval_date, ql.TARGET(), float(VOL), ql.Actual365Fixed())
    )

    ql_type = ql.Option.Put if option_type is OptionType.PUT else ql.Option.Call
    payoff = ql.PlainVanillaPayoff(ql_type, float(strike))
    exercise = ql.EuropeanExercise(ql_maturity)
    option = ql.VanillaOption(payoff, exercise)

    dividend_schedule = _quantlib_dividend_schedule(discrete_dividends)
    process = ql.BlackScholesMertonProcess(spot_handle, div_curve, rf_curve, vol_handle)
    engine = ql.AnalyticDividendEuropeanEngine(process, dividend_schedule)
    option.setPricingEngine(engine)
    return float(option.NPV())


def _bsm_european(
    *,
    spot: float,
    strike: float,
    option_type: OptionType,
    discrete_dividends: Sequence[tuple[dt.datetime, float]] | None,
) -> float:
    ud = UnderlyingPricingData(
        initial_value=spot,
        volatility=VOL,
        market_data=_market_data(),
        discrete_dividends=discrete_dividends,
    )
    spec = _spec(strike=strike, option_type=option_type, exercise_type=ExerciseType.EUROPEAN)
    return OptionValuation(ud, spec, PricingMethod.BSM).present_value()


def _binomial_european(
    *,
    spot: float,
    strike: float,
    option_type: OptionType,
    discrete_dividends: Sequence[tuple[dt.datetime, float]] | None,
) -> float:
    ud = UnderlyingPricingData(
        initial_value=spot,
        volatility=VOL,
        market_data=_market_data(),
        discrete_dividends=discrete_dividends,
    )
    spec = _spec(strike=strike, option_type=option_type, exercise_type=ExerciseType.EUROPEAN)
    return OptionValuation(ud, spec, PricingMethod.BINOMIAL, BINOM_CFG).present_value()


def _mc_american(
    *,
    spot: float,
    strike: float,
    option_type: OptionType,
    discrete_dividends: Sequence[tuple[dt.datetime, float]] | None,
) -> float:
    md = _market_data()
    gbm_params = GBMParams(
        initial_value=spot,
        volatility=VOL,
        discrete_dividends=discrete_dividends,
    )
    sim_config = SimulationConfig(
        paths=100_000,
        end_date=MATURITY,
        num_steps=200,
    )
    gbm = GBMProcess(md, gbm_params, sim_config)
    spec = _spec(strike=strike, option_type=option_type)
    return OptionValuation(gbm, spec, PricingMethod.MONTE_CARLO, MC_CFG).present_value()


# ---------------------------------------------------------------------------
# Boundary-dividend tests: discrete dividends on pricing date / maturity / both
# ---------------------------------------------------------------------------

_BOUNDARY_DIV_CASES = [
    pytest.param(
        [
            (PRICING_DATE + dt.timedelta(days=90), 0.75),
            (PRICING_DATE + dt.timedelta(days=270), 0.75),
        ],
        id="interior",
    ),
    pytest.param(
        [
            (PRICING_DATE, 0.75),
            (PRICING_DATE + dt.timedelta(days=180), 0.75),
        ],
        id="on_pricing_date",
    ),
    pytest.param(
        [
            (PRICING_DATE + dt.timedelta(days=180), 0.75),
            (MATURITY, 0.75),
        ],
        id="on_maturity",
    ),
    pytest.param(
        [
            (PRICING_DATE, 0.75),
            (MATURITY, 0.75),
        ],
        id="on_both_boundaries",
    ),
]


@pytest.mark.parametrize("option_type", [OptionType.PUT, OptionType.CALL])
@pytest.mark.parametrize("divs", _BOUNDARY_DIV_CASES)
def test_european_discrete_div_boundary_vs_quantlib(divs, option_type):
    """BSM and Binomial European prices match QuantLib with boundary dividends."""
    spot, strike = 52.0, 50.0

    ql_price = _quantlib_european(
        spot=spot,
        strike=strike,
        option_type=option_type,
        discrete_dividends=divs,
    )
    bsm_price = _bsm_european(
        spot=spot,
        strike=strike,
        option_type=option_type,
        discrete_dividends=divs,
    )
    binom_price = _binomial_european(
        spot=spot,
        strike=strike,
        option_type=option_type,
        discrete_dividends=divs,
    )

    logger.info(
        "European %s | QL=%.6f BSM=%.6f Binom=%.6f",
        option_type.value,
        ql_price,
        bsm_price,
        binom_price,
    )

    assert np.isclose(bsm_price, ql_price, rtol=1e-4), f"BSM {bsm_price:.6f} vs QL {ql_price:.6f}"
    assert np.isclose(binom_price, ql_price, rtol=0.01), (
        f"Binomial {binom_price:.6f} vs QL {ql_price:.6f}"
    )


@pytest.mark.parametrize("option_type", [OptionType.PUT, OptionType.CALL])
@pytest.mark.parametrize("divs", _BOUNDARY_DIV_CASES)
def test_american_discrete_div_boundary_vs_quantlib(divs, option_type):
    """PDE and MC American prices match QuantLib with boundary dividends."""
    spot, strike = 52.0, 50.0

    ql_price = _quantlib_american(
        spot=spot,
        strike=strike,
        option_type=option_type,
        dividend_yield=0.0,
        discrete_dividends=divs,
    )
    pde_price = _pde_fd_american(
        spot=spot,
        strike=strike,
        option_type=option_type,
        discrete_dividends=divs,
    )
    mc_price = _mc_american(
        spot=spot,
        strike=strike,
        option_type=option_type,
        discrete_dividends=divs,
    )

    logger.info(
        "American %s | QL=%.6f PDE=%.6f MC=%.6f",
        option_type.value,
        ql_price,
        pde_price,
        mc_price,
    )

    assert np.isclose(pde_price, ql_price, rtol=0.01), f"PDE {pde_price:.6f} vs QL {ql_price:.6f}"
    assert np.isclose(mc_price, ql_price, rtol=0.015), f"MC {mc_price:.6f} vs QL {ql_price:.6f}"


# ═══════════════════════════════════════════════════════════════════════════
# Asian option comparison — discrete fixing dates vs QuantLib
# ═══════════════════════════════════════════════════════════════════════════

# Shared Asian constants
_ASIAN_SPOT = 100.0
_ASIAN_STRIKE = 100.0
_ASIAN_VOL = 0.20
_ASIAN_RATE = 0.05
_ASIAN_MATURITY = PRICING_DATE + dt.timedelta(days=365)
_ASIAN_MC_PATHS = 500_000
_ASIAN_MC_SEED = 42
_ASIAN_NUM_STEPS = 60  # dense grid for MC simulation


def _dt_to_ql(d: dt.datetime) -> "ql_typing.Date":
    return ql.Date(d.day, d.month, d.year)


def _ql_asian_process(
    *,
    spot: float = _ASIAN_SPOT,
    vol: float = _ASIAN_VOL,
    rf_rate: float | None = None,
    rf_handle: "ql_typing.YieldTermStructureHandle | None" = None,
    div_rate: float = 0.0,
    div_handle: "ql_typing.YieldTermStructureHandle | None" = None,
) -> "ql_typing.BlackScholesMertonProcess":
    eval_date = _dt_to_ql(PRICING_DATE)
    ql.Settings.instance().evaluationDate = eval_date
    spot_h = ql.QuoteHandle(ql.SimpleQuote(spot))
    if rf_handle is None:
        rf_handle = ql.YieldTermStructureHandle(
            ql.FlatForward(
                eval_date, rf_rate if rf_rate is not None else _ASIAN_RATE, ql.Actual365Fixed()
            )
        )
    if div_handle is None:
        div_handle = ql.YieldTermStructureHandle(
            ql.FlatForward(eval_date, div_rate, ql.Actual365Fixed())
        )
    vol_h = ql.BlackVolTermStructureHandle(
        ql.BlackConstantVol(eval_date, ql.TARGET(), vol, ql.Actual365Fixed())
    )
    return ql.BlackScholesMertonProcess(spot_h, div_handle, rf_handle, vol_h)


def _ql_discrete_asian_price(
    *,
    fixing_dates_ql: list,
    option_type_ql: int,
    averaging_ql,
    engine_factory,
    strike: float = _ASIAN_STRIKE,
    spot: float = _ASIAN_SPOT,
    vol: float = _ASIAN_VOL,
    rf_rate: float | None = None,
    rf_handle: "ql_typing.YieldTermStructureHandle | None" = None,
    div_rate: float = 0.0,
    div_handle: "ql_typing.YieldTermStructureHandle | None" = None,
) -> float:
    process = _ql_asian_process(
        spot=spot,
        vol=vol,
        rf_rate=rf_rate,
        rf_handle=rf_handle,
        div_rate=div_rate,
        div_handle=div_handle,
    )
    payoff = ql.PlainVanillaPayoff(option_type_ql, strike)
    exercise = ql.EuropeanExercise(_dt_to_ql(_ASIAN_MATURITY))
    opt = ql.DiscreteAveragingAsianOption(averaging_ql, fixing_dates_ql, payoff, exercise)
    opt.setPricingEngine(engine_factory(process))
    return opt.NPV()


def _pa_asian_market_data(
    r_curve: DiscountCurve | None = None,
) -> MarketData:

    ttm = calculate_year_fraction(PRICING_DATE, _ASIAN_MATURITY)
    curve = r_curve if r_curve is not None else DiscountCurve.flat(_ASIAN_RATE, end_time=ttm)
    return MarketData(PRICING_DATE, curve, currency=CURRENCY)


def _pa_asian_gbm(
    *,
    fixing_dates: tuple[dt.datetime, ...],
    spot: float = _ASIAN_SPOT,
    vol: float = _ASIAN_VOL,
    r_curve: DiscountCurve | None = None,
    dividend_curve: DiscountCurve | None = None,
) -> GBMProcess:
    md = _pa_asian_market_data(r_curve)
    params = GBMParams(initial_value=spot, volatility=vol, dividend_curve=dividend_curve)
    sim_cfg = SimulationConfig(
        paths=_ASIAN_MC_PATHS,
        end_date=_ASIAN_MATURITY,
        num_steps=_ASIAN_NUM_STEPS,
    )
    return GBMProcess(md, params, sim_cfg)


# ── Test 1: Monthly fixings — arithmetic call ───────────────────────────

_MONTHLY_FIXINGS = tuple(
    dt.datetime(2025, m, 1) if m <= 12 else dt.datetime(2026, m - 12, 1) for m in range(2, 14)
)


def test_asian_arithmetic_call_monthly_vs_quantlib():
    """Monthly-fixing arithmetic Asian call: our MC vs QuantLib TW analytic."""
    ql_fixings = [_dt_to_ql(d) for d in _MONTHLY_FIXINGS]

    ql_tw = _ql_discrete_asian_price(
        fixing_dates_ql=ql_fixings,
        option_type_ql=ql.Option.Call,
        averaging_ql=ql.Average.Arithmetic,
        engine_factory=lambda p: ql.TurnbullWakemanAsianEngine(p),
    )

    spec = AsianOptionSpec(
        averaging=AsianAveraging.ARITHMETIC,
        call_put=OptionType.CALL,
        strike=_ASIAN_STRIKE,
        maturity=_ASIAN_MATURITY,
        currency=CURRENCY,
        fixing_dates=_MONTHLY_FIXINGS,
    )
    gbm = _pa_asian_gbm(fixing_dates=_MONTHLY_FIXINGS)
    pa_mc = OptionValuation(
        gbm,
        spec,
        PricingMethod.MONTE_CARLO,
        params=MonteCarloParams(random_seed=_ASIAN_MC_SEED),
    ).present_value()

    logger.info(
        "Asian arith call monthly | QL_TW=%.6f PA_MC=%.6f",
        ql_tw,
        pa_mc,
    )
    assert np.isclose(pa_mc, ql_tw, rtol=0.015), f"PA {pa_mc:.6f} vs QL {ql_tw:.6f}"


# ── Test 2: Quarterly fixings — geometric put ───────────────────────────

_QUARTERLY_FIXINGS = (
    dt.datetime(2025, 4, 1),
    dt.datetime(2025, 7, 1),
    dt.datetime(2025, 10, 1),
    dt.datetime(2026, 1, 1),
)


def test_asian_geometric_put_quarterly_vs_quantlib():
    """Quarterly-fixing geometric Asian put: our MC vs QuantLib analytic."""
    ql_fixings = [_dt_to_ql(d) for d in _QUARTERLY_FIXINGS]

    ql_analytic = _ql_discrete_asian_price(
        fixing_dates_ql=ql_fixings,
        option_type_ql=ql.Option.Put,
        averaging_ql=ql.Average.Geometric,
        engine_factory=lambda p: ql.AnalyticDiscreteGeometricAveragePriceAsianEngine(p),
    )

    spec = AsianOptionSpec(
        averaging=AsianAveraging.GEOMETRIC,
        call_put=OptionType.PUT,
        strike=_ASIAN_STRIKE,
        maturity=_ASIAN_MATURITY,
        currency=CURRENCY,
        fixing_dates=_QUARTERLY_FIXINGS,
    )
    gbm = _pa_asian_gbm(fixing_dates=_QUARTERLY_FIXINGS)
    pa_mc = OptionValuation(
        gbm,
        spec,
        PricingMethod.MONTE_CARLO,
        params=MonteCarloParams(random_seed=_ASIAN_MC_SEED),
    ).present_value()

    logger.info(
        "Asian geom put quarterly | QL_analytic=%.6f PA_MC=%.6f",
        ql_analytic,
        pa_mc,
    )
    assert np.isclose(pa_mc, ql_analytic, rtol=0.015), f"PA {pa_mc:.6f} vs QL {ql_analytic:.6f}"


# ── Test 3: Non-flat rate and dividend curves — arithmetic put ──────────

_BIMONTHLY_FIXINGS = tuple(
    dt.datetime(2025, m, 15) for m in range(2, 13, 2)
)  # Feb, Apr, Jun, Aug, Oct, Dec — 6 dates


def test_asian_arithmetic_put_nonflat_curves_vs_quantlib():
    """Arithmetic Asian put with non-flat rate/div curves: our MC vs QuantLib TW."""
    ttm = calculate_year_fraction(PRICING_DATE, _ASIAN_MATURITY)

    # Non-flat rate curve: 3% → 5% → 4%
    r_times = np.array([0.0, 0.25, 0.5, ttm])
    r_forwards = np.array([0.03, 0.05, 0.04])
    r_curve = DiscountCurve.from_forwards(times=r_times, forwards=r_forwards)

    # Non-flat dividend curve: 1% → 2% → 0%
    q_times = np.array([0.0, 0.25, 0.5, ttm])
    q_forwards = np.array([0.01, 0.02, 0.00])
    q_curve = DiscountCurve.from_forwards(times=q_times, forwards=q_forwards)

    # QuantLib handles from discount factors
    rf_handle = _ql_curve_from_times(times=r_curve.times, dfs=r_curve.dfs)
    div_handle = _ql_curve_from_times(times=q_curve.times, dfs=q_curve.dfs)

    ql_fixings = [_dt_to_ql(d) for d in _BIMONTHLY_FIXINGS]

    ql_tw = _ql_discrete_asian_price(
        fixing_dates_ql=ql_fixings,
        option_type_ql=ql.Option.Put,
        averaging_ql=ql.Average.Arithmetic,
        engine_factory=lambda p: ql.TurnbullWakemanAsianEngine(p),
        rf_handle=rf_handle,
        div_handle=div_handle,
    )

    spec = AsianOptionSpec(
        averaging=AsianAveraging.ARITHMETIC,
        call_put=OptionType.PUT,
        strike=_ASIAN_STRIKE,
        maturity=_ASIAN_MATURITY,
        currency=CURRENCY,
        fixing_dates=_BIMONTHLY_FIXINGS,
    )
    gbm = _pa_asian_gbm(
        fixing_dates=_BIMONTHLY_FIXINGS,
        r_curve=r_curve,
        dividend_curve=q_curve,
    )
    pa_mc = OptionValuation(
        gbm,
        spec,
        PricingMethod.MONTE_CARLO,
        params=MonteCarloParams(random_seed=_ASIAN_MC_SEED),
    ).present_value()

    logger.info(
        "Asian arith put nonflat curves | QL_TW=%.6f PA_MC=%.6f",
        ql_tw,
        pa_mc,
    )
    assert np.isclose(pa_mc, ql_tw, rtol=0.015), f"PA {pa_mc:.6f} vs QL {ql_tw:.6f}"
