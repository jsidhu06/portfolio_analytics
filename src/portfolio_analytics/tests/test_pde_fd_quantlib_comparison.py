"""Compare PDE FD American pricing vs QuantLib for reference."""

import datetime as dt
import logging
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from portfolio_analytics.enums import ExerciseType, OptionType, PricingMethod
from portfolio_analytics.market_environment import MarketData
from portfolio_analytics.rates import DiscountCurve
from portfolio_analytics.utils import calculate_year_fraction
from portfolio_analytics.valuation import OptionSpec, OptionValuation, UnderlyingPricingData
from portfolio_analytics.valuation.params import PDEParams

if TYPE_CHECKING:
    import QuantLib as ql_typing
else:
    ql_typing = Any

ql = pytest.importorskip("QuantLib")


logger = logging.getLogger(__name__)


PRICING_DATE = dt.datetime(2025, 1, 1)
MATURITY = PRICING_DATE + dt.timedelta(days=365)
RISK_FREE = 0.1
VOL = 0.4
CURRENCY = "USD"

PDE_CFG = PDEParams(spot_steps=200, time_steps=200, max_iter=20_000)


def _market_data() -> MarketData:
    ttm = calculate_year_fraction(PRICING_DATE, MATURITY)
    curve = DiscountCurve.flat("r", RISK_FREE, end_time=ttm)
    return MarketData(PRICING_DATE, curve, currency=CURRENCY)


def _build_curve_from_forwards(
    *,
    name: str,
    times: np.ndarray,
    forwards: np.ndarray,
) -> DiscountCurve:
    times = np.asarray(times, dtype=float)
    forwards = np.asarray(forwards, dtype=float)
    if times.ndim != 1 or forwards.ndim != 1:
        raise ValueError("times/forwards must be 1D arrays")
    if times.size < 2:
        raise ValueError("times must include at least [0, T]")
    if forwards.size != times.size - 1:
        raise ValueError("forwards must have length len(times)-1")
    if not np.isclose(times[0], 0.0):
        raise ValueError("times must start at 0.0")

    dfs = np.empty_like(times)
    dfs[0] = 1.0
    for i in range(forwards.size):
        dt = times[i + 1] - times[i]
        dfs[i + 1] = dfs[i] * np.exp(-forwards[i] * dt)
    return DiscountCurve(name=name, times=times, dfs=dfs)


def _ql_curve_from_times(
    *,
    times: np.ndarray,
    dfs: np.ndarray,
) -> "ql_typing.YieldTermStructureHandle":
    day_count = ql.Actual365Fixed()
    ql.Settings.instance().evaluationDate = ql.Date(
        PRICING_DATE.day, PRICING_DATE.month, PRICING_DATE.year
    )
    dates = [ql.Settings.instance().evaluationDate]
    for t in times[1:]:
        days = int(round(float(t) * 365.0))
        dates.append(ql.Settings.instance().evaluationDate + days)
    return ql.YieldTermStructureHandle(ql.DiscountCurve(dates, list(dfs), day_count))


def _spec(*, strike: float, option_type: OptionType) -> OptionSpec:
    return OptionSpec(
        option_type=option_type,
        exercise_type=ExerciseType.AMERICAN,
        strike=strike,
        maturity=MATURITY,
        currency=CURRENCY,
    )


def _quantlib_american(
    *,
    spot: float,
    strike: float,
    option_type: OptionType,
    dividend_yield: float,
    discrete_dividends: list[tuple[dt.datetime, float]] | None,
    grid_points: int = 200,
    time_steps: int = 400,
) -> float:
    ql.Settings.instance().evaluationDate = ql.Date(
        PRICING_DATE.day, PRICING_DATE.month, PRICING_DATE.year
    )

    ql_maturity = ql.Date(MATURITY.day, MATURITY.month, MATURITY.year)

    spot_handle = ql.QuoteHandle(ql.SimpleQuote(float(spot)))
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

    if discrete_dividends:
        div_dates = []
        dividends = []
        for ex_date, amount in discrete_dividends:
            if PRICING_DATE < ex_date < MATURITY:
                div_dates.append(ql.Date(ex_date.day, ex_date.month, ex_date.year))
                dividends.append(float(amount))
        dividend_schedule = ql.DividendVector(div_dates, dividends)
    else:
        dividend_schedule = ql.DividendVector([], [])

    process = ql.BlackScholesMertonProcess(spot_handle, div_curve, rf_curve, vol_handle)

    engine = ql.FdBlackScholesVanillaEngine(
        process,
        dividend_schedule,
        int(grid_points),
        int(time_steps),
    )
    option.setPricingEngine(engine)
    return float(option.NPV())


def _quantlib_american_curves(
    *,
    spot: float,
    strike: float,
    option_type: OptionType,
    r_times: np.ndarray,
    r_dfs: np.ndarray,
    q_times: np.ndarray,
    q_dfs: np.ndarray,
    grid_points: int = 200,
    time_steps: int = 400,
) -> float:
    ql.Settings.instance().evaluationDate = ql.Date(
        PRICING_DATE.day, PRICING_DATE.month, PRICING_DATE.year
    )

    ql_maturity = ql.Date(MATURITY.day, MATURITY.month, MATURITY.year)

    spot_handle = ql.QuoteHandle(ql.SimpleQuote(float(spot)))
    rf_curve = _ql_curve_from_times(times=r_times, dfs=r_dfs)
    div_curve = _ql_curve_from_times(times=q_times, dfs=q_dfs)
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

    process = ql.BlackScholesMertonProcess(spot_handle, div_curve, rf_curve, vol_handle)

    engine = ql.FdBlackScholesVanillaEngine(
        process,
        ql.DividendVector([], []),
        int(grid_points),
        int(time_steps),
    )
    option.setPricingEngine(engine)
    return float(option.NPV())


def _pde_fd_american(
    *,
    spot: float,
    strike: float,
    option_type: OptionType,
    dividend_yield: float,
    discrete_dividends: list[tuple[dt.datetime, float]] | None,
) -> float:
    ud = UnderlyingPricingData(
        initial_value=spot,
        volatility=VOL,
        market_data=_market_data(),
        dividend_yield=dividend_yield,
        discrete_dividends=discrete_dividends,
    )
    spec = _spec(strike=strike, option_type=option_type)
    return OptionValuation("pde_am", ud, spec, PricingMethod.PDE_FD, PDE_CFG).present_value()


def _pde_fd_american_curves(
    *,
    spot: float,
    strike: float,
    option_type: OptionType,
    r_curve: DiscountCurve,
    q_curve: DiscountCurve | None,
) -> float:
    md = MarketData(PRICING_DATE, r_curve, currency=CURRENCY)
    ud = UnderlyingPricingData(
        initial_value=spot,
        volatility=VOL,
        market_data=md,
        dividend_yield=0.0,
        dividend_curve=q_curve,
        discrete_dividends=None,
    )
    spec = _spec(strike=strike, option_type=option_type)
    return OptionValuation("pde_am", ud, spec, PricingMethod.PDE_FD, PDE_CFG).present_value()


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
        dividend_yield=0.0,
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
    pde = _pde_fd_american(
        spot=spot,
        strike=strike,
        option_type=option_type,
        dividend_yield=0.03,
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


def test_american_fd_vs_quantlib_discrete_div():
    spot = 52.0
    strike = 50.0
    divs = [
        (PRICING_DATE + dt.timedelta(days=90), 0.5),
        (PRICING_DATE + dt.timedelta(days=270), 0.5),
    ]

    pde = _pde_fd_american(
        spot=spot,
        strike=strike,
        option_type=OptionType.PUT,
        dividend_yield=0.0,
        discrete_dividends=divs,
    )
    ql_price = _quantlib_american(
        spot=spot,
        strike=strike,
        option_type=OptionType.PUT,
        dividend_yield=0.0,
        discrete_dividends=divs,
    )

    logger.info(
        "Disc-div American %s S=%.2f K=%.2f divs=%d | PDE=%.6f QL=%.6f",
        OptionType.PUT.value,
        spot,
        strike,
        len(divs),
        pde,
        ql_price,
    )

    assert np.isclose(pde, ql_price, rtol=0.01)


@pytest.mark.parametrize("spot,strike,option_type", [(52.0, 50.0, OptionType.PUT)])
def test_american_fd_vs_quantlib_forward_curve_only(spot, strike, option_type):
    times = np.array([0.0, 0.25, 0.5, 1.0])
    forwards = np.array([0.03, 0.05, 0.04])
    r_curve = _build_curve_from_forwards(name="r_curve", times=times, forwards=forwards)

    pde = _pde_fd_american_curves(
        spot=spot,
        strike=strike,
        option_type=option_type,
        r_curve=r_curve,
        q_curve=None,
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

    assert np.isclose(pde, ql_price, rtol=0.02)


@pytest.mark.parametrize("spot,strike,option_type", [(60.0, 55.0, OptionType.CALL)])
def test_american_fd_vs_quantlib_dividend_curve_only(spot, strike, option_type):
    r_times = np.array([0.0, 1.0])
    r_forwards = np.array([0.04])
    r_curve = _build_curve_from_forwards(name="r_flat", times=r_times, forwards=r_forwards)

    q_times = np.array([0.0, 0.25, 0.5, 1.0])
    q_forwards = np.array([0.00, 0.02, 0.04])
    q_curve = _build_curve_from_forwards(name="q_curve", times=q_times, forwards=q_forwards)

    pde = _pde_fd_american_curves(
        spot=spot,
        strike=strike,
        option_type=option_type,
        r_curve=r_curve,
        q_curve=q_curve,
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

    assert np.isclose(pde, ql_price, rtol=0.02)


@pytest.mark.parametrize("spot,strike,option_type", [(52.0, 50.0, OptionType.PUT)])
def test_american_fd_vs_quantlib_forward_and_dividend_curves(spot, strike, option_type):
    r_times = np.array([0.0, 0.25, 0.5, 1.0])
    r_forwards = np.array([0.03, 0.05, 0.04])
    r_curve = _build_curve_from_forwards(name="r_curve", times=r_times, forwards=r_forwards)

    q_times = np.array([0.0, 0.25, 0.5, 1.0])
    q_forwards = np.array([0.01, 0.02, 0.00])
    q_curve = _build_curve_from_forwards(name="q_curve", times=q_times, forwards=q_forwards)

    pde = _pde_fd_american_curves(
        spot=spot,
        strike=strike,
        option_type=option_type,
        r_curve=r_curve,
        q_curve=q_curve,
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

    assert np.isclose(pde, ql_price, rtol=0.02)
