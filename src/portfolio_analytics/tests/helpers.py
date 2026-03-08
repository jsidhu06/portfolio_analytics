import datetime as dt
import warnings
from typing import Any, Sequence

import numpy as np

from portfolio_analytics.enums import DayCountConvention, ExerciseType, OptionType
from portfolio_analytics.market_environment import MarketData
from portfolio_analytics.rates import DiscountCurve
from portfolio_analytics.stochastic_processes import GBMParams, GBMProcess, SimulationConfig
from portfolio_analytics.utils import calculate_year_fraction
from portfolio_analytics.valuation import OptionValuation, UnderlyingData, VanillaSpec


# ---------------------------------------------------------------------------
# Scalar constants (canonical source — re-exported by conftest.py)
# ---------------------------------------------------------------------------

PRICING_DATE = dt.datetime(2025, 1, 1)
MATURITY = dt.datetime(2026, 1, 1)
CURRENCY = "USD"
SPOT = 100.0
STRIKE = 100.0
RATE = 0.05
VOL = 0.20


def flat_curve(
    pricing_date: dt.datetime,
    maturity: dt.datetime,
    rate: float,
) -> DiscountCurve:
    ttm = calculate_year_fraction(pricing_date, maturity)
    return DiscountCurve.flat(rate, end_time=ttm)


_CURVE = flat_curve(PRICING_DATE, MATURITY, RATE)
_MD = MarketData(
    PRICING_DATE,
    _CURVE,
    currency=CURRENCY,
    day_count_convention=DayCountConvention.ACT_365F,
)


def market_data(
    *,
    pricing_date: dt.datetime,
    discount_curve: DiscountCurve,
    currency: str = CURRENCY,
    day_count_convention: DayCountConvention = DayCountConvention.ACT_365F,
) -> MarketData:
    return MarketData(
        pricing_date,
        discount_curve,
        currency=currency,
        day_count_convention=day_count_convention,
    )


def flat_market_data(
    *,
    pricing_date: dt.datetime = PRICING_DATE,
    maturity: dt.datetime = MATURITY,
    rate: float = RATE,
    currency: str = CURRENCY,
    day_count_convention: DayCountConvention = DayCountConvention.ACT_365F,
) -> MarketData:
    """Build MarketData using a flat risk-free curve for convenience."""
    return market_data(
        pricing_date=pricing_date,
        discount_curve=flat_curve(pricing_date, maturity, rate),
        currency=currency,
        day_count_convention=day_count_convention,
    )


def underlying(
    *,
    initial_value: float = SPOT,
    volatility: float = VOL,
    market_data: MarketData = _MD,
    discrete_dividends: Sequence[tuple[dt.datetime, float]] | None = None,
    dividend_curve: DiscountCurve | None = None,
) -> UnderlyingData:
    """Build UnderlyingData; signature mirrors the production dataclass."""
    return UnderlyingData(
        initial_value=initial_value,
        volatility=volatility,
        market_data=market_data,
        dividend_curve=dividend_curve,
        discrete_dividends=discrete_dividends,
    )


def flat_underlying(
    *,
    initial_value: float = SPOT,
    volatility: float = VOL,
    pricing_date: dt.datetime = PRICING_DATE,
    maturity: dt.datetime = MATURITY,
    rate: float = RATE,
    currency: str = CURRENCY,
    day_count_convention: DayCountConvention = DayCountConvention.ACT_365F,
    discrete_dividends: Sequence[tuple[dt.datetime, float]] | None = None,
    dividend_curve: DiscountCurve | None = None,
) -> UnderlyingData:
    """Convenience wrapper: build UnderlyingData with a flat risk-free curve."""
    md = flat_market_data(
        pricing_date=pricing_date,
        maturity=maturity,
        rate=rate,
        currency=currency,
        day_count_convention=day_count_convention,
    )
    return underlying(
        initial_value=initial_value,
        volatility=volatility,
        market_data=md,
        discrete_dividends=discrete_dividends,
        dividend_curve=dividend_curve,
    )


def spec(
    option_type: OptionType = OptionType.CALL,
    exercise: ExerciseType = ExerciseType.EUROPEAN,
    strike: float = STRIKE,
    maturity: dt.datetime = MATURITY,
    currency: str = CURRENCY,
) -> VanillaSpec:
    return VanillaSpec(
        option_type=option_type,
        exercise_type=exercise,
        strike=strike,
        maturity=maturity,
        currency=currency,
    )


def pv(underlying: Any, spec: Any, method: Any, **kw: Any) -> float:
    """Shortcut: build an OptionValuation and return its present value."""

    return OptionValuation(underlying, spec, method, **kw).present_value()


def build_curve_from_forwards(
    *,
    times: np.ndarray,
    forwards: np.ndarray,
) -> DiscountCurve:
    """Deprecated — use ``DiscountCurve.from_forwards()`` instead."""
    warnings.warn(
        "build_curve_from_forwards is deprecated; use DiscountCurve.from_forwards()",
        DeprecationWarning,
        stacklevel=2,
    )
    return DiscountCurve.from_forwards(times=times, forwards=forwards)


def gbm(
    *,
    market_data: MarketData,
    process_params: GBMParams,
    sim_config: SimulationConfig,
    random_seed: int | None = None,
) -> GBMProcess:
    """Build GBMProcess; signature mirrors constructor plus optional eager simulate."""
    process = GBMProcess(market_data, process_params, sim_config)
    if random_seed is not None:
        process.simulate(random_seed=random_seed)
    return process


def flat_gbm(
    *,
    initial_value: float,
    volatility: float,
    pricing_date: dt.datetime,
    maturity: dt.datetime,
    rate: float,
    paths: int,
    num_steps: int,
    currency: str = CURRENCY,
    dividend_curve: DiscountCurve | None = None,
    discrete_dividends: Sequence[tuple[dt.datetime, float]] | None = None,
    random_seed: int | None = None,
) -> GBMProcess:
    """Convenience wrapper: build GBMProcess under flat risk-free assumptions."""
    md = flat_market_data(
        pricing_date=pricing_date,
        maturity=maturity,
        rate=rate,
        currency=currency,
    )
    params = GBMParams(
        initial_value=initial_value,
        volatility=volatility,
        dividend_curve=dividend_curve,
        discrete_dividends=discrete_dividends,
    )
    sim_config = SimulationConfig(paths=paths, num_steps=num_steps, end_date=maturity)
    return gbm(
        market_data=md,
        process_params=params,
        sim_config=sim_config,
        random_seed=random_seed,
    )


def make_vanilla_spec(
    *,
    strike: float,
    maturity: dt.datetime,
    option_type: OptionType,
    exercise_type: ExerciseType = ExerciseType.EUROPEAN,
    currency: str = CURRENCY,
) -> VanillaSpec:
    """Backward-compatible wrapper around ``spec``."""
    return spec(
        option_type=option_type,
        exercise=exercise_type,
        strike=strike,
        maturity=maturity,
        currency=currency,
    )
