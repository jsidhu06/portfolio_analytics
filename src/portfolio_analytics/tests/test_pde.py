"""PDE FD grid/method/solver equivalence tests.

Verifies that different PDE finite-difference schemes (Explicit, Implicit,
Crank-Nicolson, Explicit-Hull) and space grids (SPOT, LOG_SPOT) produce
consistent prices.  These are internal engine tests — cross-method and
QuantLib comparisons live in test_quantlib_comparison.py.
"""

import datetime as dt

import numpy as np
import pytest

from portfolio_analytics.exceptions import StabilityError, UnsupportedFeatureError
from portfolio_analytics.enums import (
    ExerciseType,
    OptionType,
    PDEEarlyExercise,
    PDEMethod,
    PDESpaceGrid,
    PricingMethod,
)
from portfolio_analytics.market_environment import MarketData
from portfolio_analytics.rates import DiscountCurve
from portfolio_analytics.tests.helpers import flat_curve
from portfolio_analytics.valuation import VanillaSpec, OptionValuation, UnderlyingPricingData
from portfolio_analytics.valuation.params import PDEParams


PRICING_DATE = dt.datetime(2025, 1, 1)
MATURITY = PRICING_DATE + dt.timedelta(days=365)
RISK_FREE = 0.05
VOL = 0.2
CURRENCY = "USD"


def _market_data() -> MarketData:
    curve = flat_curve(PRICING_DATE, MATURITY, RISK_FREE)
    return MarketData(PRICING_DATE, curve, currency=CURRENCY)


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


def _spec(
    *,
    strike: float,
    option_type: OptionType,
    exercise_type: ExerciseType,
) -> VanillaSpec:
    return VanillaSpec(
        option_type=option_type,
        exercise_type=exercise_type,
        strike=strike,
        maturity=MATURITY,
        currency=CURRENCY,
    )


def test_pde_fd_grid_method_equivalence_european():
    """PDE FD variants should be in the same neighborhood for European options."""
    spot = 100.0
    strike = 100.0
    q_curve = flat_curve(PRICING_DATE, MATURITY, 0.01)
    ud = _underlying(spot=spot, dividend_curve=q_curve)
    spec = _spec(strike=strike, option_type=OptionType.CALL, exercise_type=ExerciseType.EUROPEAN)

    base_params = PDEParams(spot_steps=160, time_steps=240)
    baseline = OptionValuation(ud, spec, PricingMethod.PDE_FD, params=base_params).present_value()

    for method in (
        PDEMethod.IMPLICIT,
        PDEMethod.EXPLICIT,
        PDEMethod.EXPLICIT_HULL,
        PDEMethod.CRANK_NICOLSON,
    ):
        for grid in (PDESpaceGrid.SPOT, PDESpaceGrid.LOG_SPOT):
            params = PDEParams(
                spot_steps=160,
                time_steps=240,
                method=method,
                space_grid=grid,
                american_solver=PDEEarlyExercise.INTRINSIC,
            )

            if (
                method in (PDEMethod.EXPLICIT, PDEMethod.EXPLICIT_HULL)
                and grid is PDESpaceGrid.SPOT
            ):
                with pytest.raises(
                    StabilityError, match="Explicit spot-grid scheme likely unstable"
                ):
                    OptionValuation(ud, spec, PricingMethod.PDE_FD, params=params).present_value()
                continue

            pv = OptionValuation(ud, spec, PricingMethod.PDE_FD, params=params).present_value()
            assert np.isclose(pv, baseline, rtol=0.005)


def test_pde_fd_grid_method_equivalence_american():
    """PDE FD American variants should be in the same neighborhood."""
    spot = 95.0
    strike = 100.0
    q_curve = flat_curve(PRICING_DATE, MATURITY, 0.0)
    ud = _underlying(spot=spot, dividend_curve=q_curve)
    spec = _spec(strike=strike, option_type=OptionType.PUT, exercise_type=ExerciseType.AMERICAN)

    base_params = PDEParams(spot_steps=160, time_steps=240)
    baseline = OptionValuation(ud, spec, PricingMethod.PDE_FD, params=base_params).present_value()

    for method in (
        PDEMethod.IMPLICIT,
        PDEMethod.EXPLICIT,
        PDEMethod.EXPLICIT_HULL,
        PDEMethod.CRANK_NICOLSON,
    ):
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

                if (
                    method in (PDEMethod.EXPLICIT, PDEMethod.EXPLICIT_HULL)
                    and solver is PDEEarlyExercise.GAUSS_SEIDEL
                ):
                    with pytest.raises(
                        UnsupportedFeatureError, match="GAUSS_SEIDEL is not supported"
                    ):
                        OptionValuation(
                            ud, spec, PricingMethod.PDE_FD, params=params
                        ).present_value()
                    continue

                if (
                    method in (PDEMethod.EXPLICIT, PDEMethod.EXPLICIT_HULL)
                    and grid is PDESpaceGrid.SPOT
                ):
                    with pytest.raises(
                        StabilityError, match="Explicit spot-grid scheme likely unstable"
                    ):
                        OptionValuation(
                            ud, spec, PricingMethod.PDE_FD, params=params
                        ).present_value()
                    continue

                pv = OptionValuation(ud, spec, PricingMethod.PDE_FD, params=params).present_value()
                assert np.isclose(pv, baseline, rtol=0.005)
