"""Equivalence tests across pricing methods (PDE FD, BSM, Binomial, MC)."""

from typing import Sequence
import datetime as dt

import numpy as np
import pytest

from portfolio_analytics.exceptions import StabilityError, UnsupportedFeatureError
from portfolio_analytics.enums import (
    DayCountConvention,
    ExerciseType,
    OptionType,
    PDEEarlyExercise,
    PDEMethod,
    PDESpaceGrid,
    PricingMethod,
)
from portfolio_analytics.market_environment import MarketData
from portfolio_analytics.rates import DiscountCurve
from portfolio_analytics.tests.helpers import flat_curve, build_curve_from_forwards
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


def _market_data(r_curve: DiscountCurve | None = None) -> MarketData:
    curve = r_curve if r_curve is not None else flat_curve(PRICING_DATE, MATURITY, RISK_FREE)
    return MarketData(PRICING_DATE, curve, currency=CURRENCY)


def _nonflat_r_curve() -> DiscountCurve:
    times = np.array([0.0, 0.25, 0.5, 1.0])
    forwards = np.array([0.03, 0.05, 0.04])
    return build_curve_from_forwards(name="r_curve_nonflat", times=times, forwards=forwards)


def _underlying(
    *,
    spot: float,
    r_curve: DiscountCurve | None = None,
    dividend_curve: DiscountCurve | None = None,
    discrete_dividends: Sequence[tuple[dt.datetime, float]] | None = None,
) -> UnderlyingPricingData:
    return UnderlyingPricingData(
        initial_value=spot,
        volatility=VOL,
        market_data=_market_data(r_curve),
        dividend_curve=dividend_curve,
        discrete_dividends=discrete_dividends,
    )


def _gbm(
    *,
    spot: float,
    r_curve: DiscountCurve | None = None,
    dividend_curve: DiscountCurve | None = None,
    discrete_dividends: Sequence[tuple[dt.datetime, float]] | None = None,
    paths: int = 200_000,
    name: str = "gbm",
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
        dividend_curve=dividend_curve,
        discrete_dividends=discrete_dividends,
    )
    return GeometricBrownianMotion(name, _market_data(r_curve), gbm_params, sim_config)


def _spec(*, strike: float, option_type: OptionType, exercise_type: ExerciseType) -> OptionSpec:
    return OptionSpec(
        option_type=option_type,
        exercise_type=exercise_type,
        strike=strike,
        maturity=MATURITY,
        currency=CURRENCY,
    )


def _underlying_curves(
    *,
    spot: float,
    r_curve: DiscountCurve,
    q_curve: DiscountCurve | None,
) -> UnderlyingPricingData:
    return _underlying(spot=spot, r_curve=r_curve, dividend_curve=q_curve)


@pytest.mark.parametrize(
    "spot,strike,option_type",
    [
        (90.0, 100.0, OptionType.CALL),
        (110.0, 100.0, OptionType.PUT),
    ],
)
@pytest.mark.parametrize("dividend_yield", [0.0, 0.03])
def test_pde_fd_european_close_to_bsm(spot, strike, option_type, dividend_yield):
    """PDE FD European should match BSM for continuous dividend curves."""
    q_curve = flat_curve(PRICING_DATE, MATURITY, dividend_yield)
    ud = _underlying(spot=spot, dividend_curve=q_curve)
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
    """PDE FD American should match MC (LSM) for continuous dividend curves."""
    q_curve = flat_curve(PRICING_DATE, MATURITY, dividend_yield)
    ud = _underlying(spot=spot, dividend_curve=q_curve)
    gbm = _gbm(spot=spot, dividend_curve=q_curve)
    spec = _spec(strike=strike, option_type=option_type, exercise_type=ExerciseType.AMERICAN)

    pde_pv = OptionValuation(
        "pde_am", ud, spec, PricingMethod.PDE_FD, params=PDE_CFG
    ).present_value()
    mc_pv = OptionValuation(
        "mc_am", gbm, spec, PricingMethod.MONTE_CARLO, params=MC_CFG_AM
    ).present_value()

    assert np.isclose(pde_pv, mc_pv, rtol=0.02)


@pytest.mark.parametrize(
    "r_curve",
    [
        flat_curve(PRICING_DATE, MATURITY, RISK_FREE),
        _nonflat_r_curve(),
    ],
    ids=["flat", "nonflat"],
)
def test_discrete_dividend_equivalence_across_methods(r_curve):
    """Discrete divs: PDE/MC align; BSM/Binomial align; vol-adjusted BSM/Binomial near PDE/MC."""
    spot = 52.0
    strike = 50.0
    divs = [
        (PRICING_DATE + dt.timedelta(days=90), 0.5),
        (PRICING_DATE + dt.timedelta(days=270), 0.5),
    ]

    ud = _underlying(spot=spot, r_curve=r_curve, discrete_dividends=divs)
    gbm = _gbm(
        spot=spot,
        r_curve=r_curve,
        discrete_dividends=divs,
        paths=200_000,
        name="gbm_curve_div",
    )

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
        curve_date=ud.pricing_date,
        end_date=spec_eu.maturity,
        discount_curve=r_curve,
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
@pytest.mark.parametrize(
    "r_curve",
    [
        flat_curve(PRICING_DATE, MATURITY, RISK_FREE),
        _nonflat_r_curve(),
    ],
    ids=["flat", "nonflat"],
)
def test_discrete_dividend_american_matches_mc(spot, strike, r_curve):
    """American discrete dividend: PDE FD should align with MC."""
    divs = [
        (PRICING_DATE + dt.timedelta(days=120), 0.6),
        (PRICING_DATE + dt.timedelta(days=240), 0.6),
    ]
    ud = _underlying(spot=spot, r_curve=r_curve, discrete_dividends=divs)
    gbm = _gbm(
        spot=spot,
        r_curve=r_curve,
        discrete_dividends=divs,
        paths=60_000,
        name="gbm_curve_div",
    )
    spec_am = _spec(strike=strike, option_type=OptionType.PUT, exercise_type=ExerciseType.AMERICAN)

    pde_pv = OptionValuation(
        "pde_am_div", ud, spec_am, PricingMethod.PDE_FD, params=PDE_CFG
    ).present_value()
    mc_pv = OptionValuation(
        "mc_am_div", gbm, spec_am, PricingMethod.MONTE_CARLO, params=MC_CFG_AM
    ).present_value()

    assert np.isclose(pde_pv, mc_pv, rtol=0.02)


def test_pde_fd_grid_method_equivalence_european():
    """PDE FD variants should be in the same neighborhood for European options."""
    spot = 100.0
    strike = 100.0
    q_curve = flat_curve(PRICING_DATE, MATURITY, 0.01)
    ud = _underlying(spot=spot, dividend_curve=q_curve)
    spec = _spec(strike=strike, option_type=OptionType.CALL, exercise_type=ExerciseType.EUROPEAN)

    base_params = PDEParams(spot_steps=160, time_steps=240)
    baseline = OptionValuation(
        "pde_base", ud, spec, PricingMethod.PDE_FD, params=base_params
    ).present_value()

    for method in (PDEMethod.IMPLICIT, PDEMethod.EXPLICIT, PDEMethod.CRANK_NICOLSON):
        for grid in (PDESpaceGrid.SPOT, PDESpaceGrid.LOG_SPOT):
            params = PDEParams(
                spot_steps=160,
                time_steps=240,
                method=method,
                space_grid=grid,
                american_solver=PDEEarlyExercise.INTRINSIC,
            )

            if method is PDEMethod.EXPLICIT and grid is PDESpaceGrid.SPOT:
                with pytest.raises(StabilityError, match="Explicit spot-grid scheme unstable"):
                    OptionValuation(
                        "pde_var", ud, spec, PricingMethod.PDE_FD, params=params
                    ).present_value()
                continue

            pv = OptionValuation(
                "pde_var", ud, spec, PricingMethod.PDE_FD, params=params
            ).present_value()
            assert np.isclose(pv, baseline, rtol=0.005)


def test_pde_fd_grid_method_equivalence_american():
    """PDE FD American variants should be in the same neighborhood."""
    spot = 95.0
    strike = 100.0
    q_curve = flat_curve(PRICING_DATE, MATURITY, 0.0)
    ud = _underlying(spot=spot, dividend_curve=q_curve)
    spec = _spec(strike=strike, option_type=OptionType.PUT, exercise_type=ExerciseType.AMERICAN)

    base_params = PDEParams(spot_steps=160, time_steps=240)
    baseline = OptionValuation(
        "pde_base_am", ud, spec, PricingMethod.PDE_FD, params=base_params
    ).present_value()

    for method in (PDEMethod.IMPLICIT, PDEMethod.EXPLICIT, PDEMethod.CRANK_NICOLSON):
        for grid in (PDESpaceGrid.SPOT, PDESpaceGrid.LOG_SPOT):
            for solver in (PDEEarlyExercise.INTRINSIC, PDEEarlyExercise.GAUSS_SEIDEL):
                params = PDEParams(
                    spot_steps=160,
                    time_steps=240,
                    method=method,
                    space_grid=grid,
                    american_solver=solver,
                    max_iter=20_000,
                )

                if method is PDEMethod.EXPLICIT and solver is PDEEarlyExercise.GAUSS_SEIDEL:
                    with pytest.raises(
                        UnsupportedFeatureError, match="GAUSS_SEIDEL is not supported"
                    ):
                        OptionValuation(
                            "pde_var_am", ud, spec, PricingMethod.PDE_FD, params=params
                        ).present_value()
                    continue

                if method is PDEMethod.EXPLICIT and grid is PDESpaceGrid.SPOT:
                    with pytest.raises(StabilityError, match="Explicit spot-grid scheme unstable"):
                        OptionValuation(
                            "pde_var_am", ud, spec, PricingMethod.PDE_FD, params=params
                        ).present_value()
                    continue

                pv = OptionValuation(
                    "pde_var_am", ud, spec, PricingMethod.PDE_FD, params=params
                ).present_value()
                assert np.isclose(pv, baseline, rtol=0.005)


@pytest.mark.parametrize(
    "spot,strike,option_type,r_times,r_forwards,q_times,q_forwards",
    [
        (
            52.0,
            50.0,
            OptionType.PUT,
            np.array([0.0, 0.25, 0.5, 1.0]),
            np.array([0.03, 0.05, 0.04]),
            np.array([0.0, 1.0]),
            np.array([0.0]),
        ),
        (
            60.0,
            55.0,
            OptionType.CALL,
            np.array([0.0, 1.0]),
            np.array([0.04]),
            np.array([0.0, 0.25, 0.5, 1.0]),
            np.array([0.00, 0.02, 0.04]),
        ),
        (
            52.0,
            50.0,
            OptionType.PUT,
            np.array([0.0, 0.25, 0.5, 1.0]),
            np.array([0.03, 0.05, 0.04]),
            np.array([0.0, 0.25, 0.5, 1.0]),
            np.array([0.01, 0.02, 0.00]),
        ),
    ],
)
def test_european_method_equivalence_with_forward_curves(
    spot,
    strike,
    option_type,
    r_times,
    r_forwards,
    q_times,
    q_forwards,
):
    """European BSM, MC, PDE FD, and binomial should agree under time-varying curves."""
    r_curve = build_curve_from_forwards(name="r_curve", times=r_times, forwards=r_forwards)
    q_curve = build_curve_from_forwards(name="q_curve", times=q_times, forwards=q_forwards)

    ud = _underlying_curves(spot=spot, r_curve=r_curve, q_curve=q_curve)
    gbm = _gbm(
        spot=spot,
        r_curve=r_curve,
        dividend_curve=q_curve,
        paths=150_000,
        name="gbm_curve",
    )
    spec = _spec(strike=strike, option_type=option_type, exercise_type=ExerciseType.EUROPEAN)

    bsm_pv = OptionValuation("bsm_curve_eu", ud, spec, PricingMethod.BSM).present_value()
    pde_pv = OptionValuation(
        "pde_curve_eu", ud, spec, PricingMethod.PDE_FD, params=PDE_CFG
    ).present_value()
    binom_pv = OptionValuation(
        "binom_curve_eu",
        ud,
        spec,
        PricingMethod.BINOMIAL,
        params=BinomialParams(num_steps=1500),
    ).present_value()
    mc_pv = OptionValuation(
        "mc_curve_eu",
        gbm,
        spec,
        PricingMethod.MONTE_CARLO,
        params=MC_CFG_EU,
    ).present_value()

    assert np.isclose(pde_pv, bsm_pv, rtol=0.01)
    assert np.isclose(binom_pv, bsm_pv, rtol=0.01)
    assert np.isclose(mc_pv, bsm_pv, rtol=0.01)


@pytest.mark.parametrize(
    "spot,strike,option_type,r_times,r_forwards,q_times,q_forwards",
    [
        (
            52.0,
            50.0,
            OptionType.PUT,
            np.array([0.0, 0.25, 0.5, 1.0]),
            np.array([0.03, 0.05, 0.04]),
            np.array([0.0, 1.0]),
            np.array([0.0]),
        ),
        (
            60.0,
            55.0,
            OptionType.CALL,
            np.array([0.0, 1.0]),
            np.array([0.04]),
            np.array([0.0, 0.25, 0.5, 1.0]),
            np.array([0.00, 0.02, 0.04]),
        ),
        (
            52.0,
            50.0,
            OptionType.PUT,
            np.array([0.0, 0.25, 0.5, 1.0]),
            np.array([0.03, 0.05, 0.04]),
            np.array([0.0, 0.25, 0.5, 1.0]),
            np.array([0.01, 0.02, 0.00]),
        ),
    ],
)
def test_american_method_equivalence_with_forward_curves(
    spot,
    strike,
    option_type,
    r_times,
    r_forwards,
    q_times,
    q_forwards,
):
    """American MC, PDE FD, and binomial should agree under time-varying curves."""
    r_curve = build_curve_from_forwards(name="r_curve", times=r_times, forwards=r_forwards)
    q_curve = build_curve_from_forwards(name="q_curve", times=q_times, forwards=q_forwards)

    ud = _underlying_curves(spot=spot, r_curve=r_curve, q_curve=q_curve)
    gbm = _gbm(
        spot=spot,
        r_curve=r_curve,
        dividend_curve=q_curve,
        paths=150_000,
        name="gbm_curve",
    )
    spec = _spec(strike=strike, option_type=option_type, exercise_type=ExerciseType.AMERICAN)

    pde_pv = OptionValuation(
        "pde_curve_am", ud, spec, PricingMethod.PDE_FD, params=PDE_CFG
    ).present_value()
    binom_pv = OptionValuation(
        "binom_curve_am",
        ud,
        spec,
        PricingMethod.BINOMIAL,
        params=BinomialParams(num_steps=1500),
    ).present_value()
    mc_pv = OptionValuation(
        "mc_curve_am",
        gbm,
        spec,
        PricingMethod.MONTE_CARLO,
        params=MC_CFG_AM,
    ).present_value()

    assert np.isclose(mc_pv, pde_pv, rtol=0.01)
    assert np.isclose(mc_pv, binom_pv, rtol=0.01)
