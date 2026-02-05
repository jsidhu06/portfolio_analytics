"""Compare PDE FD American pricing vs QuantLib for reference."""

import datetime as dt
import logging

import numpy as np
import pytest

from portfolio_analytics.enums import ExerciseType, OptionType, PricingMethod
from portfolio_analytics.market_environment import MarketData
from portfolio_analytics.rates import ConstantShortRate
from portfolio_analytics.valuation import OptionSpec, OptionValuation, UnderlyingPricingData
from portfolio_analytics.valuation.params import PDEParams

ql = pytest.importorskip("QuantLib")

logger = logging.getLogger(__name__)


PRICING_DATE = dt.datetime(2025, 1, 1)
MATURITY = PRICING_DATE + dt.timedelta(days=365)
RISK_FREE = 0.1
VOL = 0.4
CURRENCY = "USD"

PDE_CFG = PDEParams(spot_steps=200, time_steps=200, max_iter=20_000)


def _market_data() -> MarketData:
    return MarketData(PRICING_DATE, ConstantShortRate("r", RISK_FREE), currency=CURRENCY)


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
