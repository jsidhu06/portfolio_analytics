"""PDE FD grid/method/solver equivalence tests.

Verifies that different PDE finite-difference schemes (Explicit, Implicit,
Crank-Nicolson, Explicit-Hull) and space grids (SPOT, LOG_SPOT) produce
consistent prices.  These are internal engine tests — cross-method and
QuantLib comparisons live in test_quantlib_comparison.py.
"""

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
from portfolio_analytics.tests.helpers import (
    flat_curve,
    underlying,
    spec,
    PRICING_DATE,
    MATURITY,
)
from portfolio_analytics.valuation import OptionValuation
from portfolio_analytics.valuation.params import PDEParams


def test_pde_fd_grid_method_equivalence_european():
    """PDE FD variants should be in the same neighborhood for European options."""
    q_curve = flat_curve(PRICING_DATE, MATURITY, 0.01)
    ud = underlying(initial_value=100.0, dividend_curve=q_curve)
    sp = spec(strike=100.0, option_type=OptionType.CALL, exercise=ExerciseType.EUROPEAN)

    base_params = PDEParams(spot_steps=160, time_steps=240)
    baseline = OptionValuation(ud, sp, PricingMethod.PDE_FD, params=base_params).present_value()

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
                    OptionValuation(ud, sp, PricingMethod.PDE_FD, params=params).present_value()
                continue

            pv = OptionValuation(ud, sp, PricingMethod.PDE_FD, params=params).present_value()
            assert np.isclose(pv, baseline, rtol=0.005)


def test_pde_fd_grid_method_equivalence_american():
    """PDE FD American variants should be in the same neighborhood."""
    q_curve = flat_curve(PRICING_DATE, MATURITY, 0.0)
    ud = underlying(initial_value=95.0, dividend_curve=q_curve)
    sp = spec(strike=100.0, option_type=OptionType.PUT, exercise=ExerciseType.AMERICAN)

    base_params = PDEParams(spot_steps=160, time_steps=240)
    baseline = OptionValuation(ud, sp, PricingMethod.PDE_FD, params=base_params).present_value()

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
                        OptionValuation(ud, sp, PricingMethod.PDE_FD, params=params).present_value()
                    continue

                if (
                    method in (PDEMethod.EXPLICIT, PDEMethod.EXPLICIT_HULL)
                    and grid is PDESpaceGrid.SPOT
                ):
                    with pytest.raises(
                        StabilityError, match="Explicit spot-grid scheme likely unstable"
                    ):
                        OptionValuation(ud, sp, PricingMethod.PDE_FD, params=params).present_value()
                    continue

                pv = OptionValuation(ud, sp, PricingMethod.PDE_FD, params=params).present_value()
                assert np.isclose(pv, baseline, rtol=0.005)
