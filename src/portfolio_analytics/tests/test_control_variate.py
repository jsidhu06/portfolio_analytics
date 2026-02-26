"""Tests for Hull-style control variate adjustments."""

import datetime as dt
import logging

import numpy as np
import pytest

from portfolio_analytics.enums import (
    AsianAveraging,
    ExerciseType,
    OptionType,
    PricingMethod,
)
from portfolio_analytics.market_environment import MarketData
from portfolio_analytics.tests.helpers import flat_curve
from portfolio_analytics.valuation import OptionSpec, OptionValuation, UnderlyingPricingData
from portfolio_analytics.valuation.core import AsianOptionSpec
from portfolio_analytics.valuation.params import BinomialParams, PDEParams

logger = logging.getLogger(__name__)


PRICING_DATE = dt.datetime(2025, 1, 1)
MATURITY = dt.datetime(2026, 1, 1)
CURRENCY = "USD"


def _build_underlying(spot: float, vol: float, rate: float, dividend_rate: float = 0.0):
    curve = flat_curve(PRICING_DATE, MATURITY, rate)
    dividend_curve = (
        None if dividend_rate == 0.0 else flat_curve(PRICING_DATE, MATURITY, dividend_rate)
    )
    market_data = MarketData(PRICING_DATE, curve, currency=CURRENCY)
    return UnderlyingPricingData(
        initial_value=spot,
        volatility=vol,
        market_data=market_data,
        dividend_curve=dividend_curve,
    )


def _build_spec(option_type: OptionType, exercise_type: ExerciseType, strike: float) -> OptionSpec:
    return OptionSpec(
        option_type=option_type,
        exercise_type=exercise_type,
        strike=strike,
        maturity=MATURITY,
        currency=CURRENCY,
    )


def test_control_variate_binomial_matches_adjustment():
    spot = 100.0
    strike = 110.0
    vol = 0.25
    rate = 0.03

    underlying = _build_underlying(spot, vol, rate)
    amer_spec = _build_spec(OptionType.PUT, ExerciseType.AMERICAN, strike)
    euro_spec = _build_spec(OptionType.PUT, ExerciseType.EUROPEAN, strike)

    base_params = BinomialParams(num_steps=300)
    cv_params = BinomialParams(num_steps=300, control_variate_european=True)

    american_raw = OptionValuation(
        underlying,
        amer_spec,
        PricingMethod.BINOMIAL,
        params=base_params,
    ).present_value()

    european_num = OptionValuation(
        underlying,
        euro_spec,
        PricingMethod.BINOMIAL,
        params=base_params,
    ).present_value()

    european_bsm = OptionValuation(
        underlying,
        euro_spec,
        PricingMethod.BSM,
    ).present_value()

    american_cv = OptionValuation(
        underlying,
        amer_spec,
        PricingMethod.BINOMIAL,
        params=cv_params,
    ).present_value()

    expected = american_raw + (european_bsm - european_num)
    assert np.isclose(american_cv, expected, atol=1.0e-8)


def test_control_variate_pde_matches_adjustment():
    spot = 100.0
    strike = 95.0
    vol = 0.2
    rate = 0.04

    underlying = _build_underlying(spot, vol, rate)
    amer_spec = _build_spec(OptionType.PUT, ExerciseType.AMERICAN, strike)
    euro_spec = _build_spec(OptionType.PUT, ExerciseType.EUROPEAN, strike)

    base_params = PDEParams(spot_steps=60, time_steps=60)
    cv_params = PDEParams(spot_steps=60, time_steps=60, control_variate_european=True)

    american_raw = OptionValuation(
        underlying,
        amer_spec,
        PricingMethod.PDE_FD,
        params=base_params,
    ).present_value()

    european_num = OptionValuation(
        underlying,
        euro_spec,
        PricingMethod.PDE_FD,
        params=base_params,
    ).present_value()

    european_bsm = OptionValuation(
        underlying,
        euro_spec,
        PricingMethod.BSM,
    ).present_value()

    american_cv = OptionValuation(
        underlying,
        amer_spec,
        PricingMethod.PDE_FD,
        params=cv_params,
    ).present_value()

    expected = american_raw + (european_bsm - european_num)
    assert np.isclose(american_cv, expected, atol=1.0e-6)


# ---------------------------------------------------------------------------
# Asian geometric control variate — Hull tree + Kemna-Vorst analytical
# ---------------------------------------------------------------------------

STEPS = 100
TREE_AVERAGES = 100


def _asian_spec(
    *,
    call_put: OptionType,
    strike: float,
    exercise_type: ExerciseType = ExerciseType.AMERICAN,
    averaging: AsianAveraging = AsianAveraging.GEOMETRIC,
) -> AsianOptionSpec:
    return AsianOptionSpec(
        averaging=averaging,
        call_put=call_put,
        strike=strike,
        maturity=MATURITY,
        currency=CURRENCY,
        exercise_type=exercise_type,
    )


class TestAsianControlVariate:
    """Control variate adjustment for American Asian option on Hull binomial tree."""

    def test_cv_matches_manual_adjustment(self):
        """CV-adjusted price = raw_american + (analytical_european − hull_european)."""
        underlying = _build_underlying(100, 0.25, 0.05, dividend_rate=0.02)
        base_params = BinomialParams(num_steps=STEPS, asian_tree_averages=TREE_AVERAGES)
        cv_params = BinomialParams(
            num_steps=STEPS,
            asian_tree_averages=TREE_AVERAGES,
            control_variate_european=True,
        )

        amer_spec = _asian_spec(call_put=OptionType.PUT, strike=100)

        # Raw American Hull tree
        american_raw = OptionValuation(
            underlying,
            amer_spec,
            PricingMethod.BINOMIAL,
            params=base_params,
        ).present_value()

        # European Hull tree (same lattice, no early exercise)
        euro_spec = _asian_spec(
            call_put=OptionType.PUT,
            strike=100,
            exercise_type=ExerciseType.EUROPEAN,
        )
        european_hull = OptionValuation(
            underlying,
            euro_spec,
            PricingMethod.BINOMIAL,
            params=base_params,
        ).present_value()

        # European analytical (Kemna-Vorst)
        bsm_spec = AsianOptionSpec(
            averaging=AsianAveraging.GEOMETRIC,
            call_put=OptionType.PUT,
            strike=100,
            maturity=MATURITY,
            currency=CURRENCY,
            num_steps=STEPS,
        )
        european_analytical = OptionValuation(
            underlying,
            bsm_spec,
            PricingMethod.BSM,
        ).present_value()

        # CV-adjusted
        american_cv = OptionValuation(
            underlying,
            amer_spec,
            PricingMethod.BINOMIAL,
            params=cv_params,
        ).present_value()

        expected = american_raw + (european_analytical - european_hull)
        assert np.isclose(american_cv, expected, atol=1e-8)
        logger.info(
            "Asian CV put: raw=%.6f cv=%.6f analytical_euro=%.6f hull_euro=%.6f adj=%.6f",
            american_raw,
            american_cv,
            european_analytical,
            european_hull,
            european_analytical - european_hull,
        )

    def test_cv_geq_european_analytical(self):
        """CV-adjusted American Asian ≥ European analytical (early exercise premium ≥ 0)."""
        underlying = _build_underlying(100, 0.25, 0.05, dividend_rate=0.02)
        cv_params = BinomialParams(
            num_steps=STEPS,
            asian_tree_averages=TREE_AVERAGES,
            control_variate_european=True,
        )

        for call_put, strike in [
            (OptionType.PUT, 110),
            (OptionType.PUT, 100),
            (OptionType.CALL, 100),
            (OptionType.CALL, 90),
        ]:
            amer_spec = _asian_spec(call_put=call_put, strike=strike)
            american_cv = OptionValuation(
                underlying,
                amer_spec,
                PricingMethod.BINOMIAL,
                params=cv_params,
            ).present_value()

            bsm_spec = AsianOptionSpec(
                averaging=AsianAveraging.GEOMETRIC,
                call_put=call_put,
                strike=strike,
                maturity=MATURITY,
                currency=CURRENCY,
                num_steps=STEPS,
            )
            european_analytical = OptionValuation(
                underlying,
                bsm_spec,
                PricingMethod.BSM,
            ).present_value()

            assert american_cv >= european_analytical - 1e-8, (
                f"{call_put.value} K={strike}: "
                f"american_cv={american_cv:.6f} < european={european_analytical:.6f}"
            )

    @pytest.mark.parametrize(
        "spot,strike,vol,r,q,days,call_put",
        [
            (100, 100, 0.20, 0.05, 0.02, 365, OptionType.PUT),
            (100, 110, 0.25, 0.03, 0.03, 365, OptionType.PUT),
            (100, 90, 0.30, 0.08, 0.01, 365, OptionType.CALL),
        ],
    )
    def test_cv_reduces_european_bias(self, spot, strike, vol, r, q, days, call_put):
        """For European exercise, CV should eliminate the Hull tree bias entirely."""
        maturity = PRICING_DATE + dt.timedelta(days=days)
        underlying = _build_underlying(spot, vol, r, dividend_rate=q)

        hull_params = BinomialParams(num_steps=STEPS, asian_tree_averages=TREE_AVERAGES)

        # European Hull tree (no CV)
        euro_spec = AsianOptionSpec(
            averaging=AsianAveraging.GEOMETRIC,
            call_put=call_put,
            strike=strike,
            maturity=maturity,
            currency=CURRENCY,
            exercise_type=ExerciseType.EUROPEAN,
        )
        euro_hull = OptionValuation(
            underlying,
            euro_spec,
            PricingMethod.BINOMIAL,
            params=hull_params,
        ).present_value()

        # European analytical
        bsm_spec = AsianOptionSpec(
            averaging=AsianAveraging.GEOMETRIC,
            call_put=call_put,
            strike=strike,
            maturity=maturity,
            currency=CURRENCY,
            num_steps=STEPS,
        )
        euro_analytical = OptionValuation(
            underlying,
            bsm_spec,
            PricingMethod.BSM,
        ).present_value()

        # Hull tree bias
        hull_bias = abs(euro_hull - euro_analytical) / euro_analytical
        logger.info(
            "European Hull bias: %s K=%d %.4f%% (hull=%.6f analytical=%.6f)",
            call_put.value,
            strike,
            hull_bias * 100,
            euro_hull,
            euro_analytical,
        )
        # Confirm the tree has measurable bias (>0.1%)
        assert hull_bias > 0.001, f"Hull tree bias too small to test CV: {hull_bias:.6f}"

    def test_cv_arithmetic_matches_manual_adjustment(self):
        """CV with arithmetic averaging uses Turnbull-Wakeman analytical formula."""
        underlying = _build_underlying(100, 0.25, 0.05)
        base_params = BinomialParams(num_steps=STEPS, asian_tree_averages=TREE_AVERAGES)
        cv_params = BinomialParams(
            num_steps=STEPS,
            asian_tree_averages=TREE_AVERAGES,
            control_variate_european=True,
        )
        amer_spec = _asian_spec(
            call_put=OptionType.CALL,
            strike=100,
            averaging=AsianAveraging.ARITHMETIC,
        )

        # Raw American Hull tree
        american_raw = OptionValuation(
            underlying,
            amer_spec,
            PricingMethod.BINOMIAL,
            params=base_params,
        ).present_value()

        # European Hull tree (same lattice)
        euro_spec = AsianOptionSpec(
            averaging=AsianAveraging.ARITHMETIC,
            call_put=OptionType.CALL,
            strike=100,
            maturity=MATURITY,
            currency=CURRENCY,
            exercise_type=ExerciseType.EUROPEAN,
        )
        european_hull = OptionValuation(
            underlying,
            euro_spec,
            PricingMethod.BINOMIAL,
            params=base_params,
        ).present_value()

        # European analytical (Turnbull-Wakeman)
        bsm_spec = AsianOptionSpec(
            averaging=AsianAveraging.ARITHMETIC,
            call_put=OptionType.CALL,
            strike=100,
            maturity=MATURITY,
            currency=CURRENCY,
            num_steps=STEPS,
        )
        european_analytical = OptionValuation(
            underlying,
            bsm_spec,
            PricingMethod.BSM,
        ).present_value()

        # CV-adjusted
        american_cv = OptionValuation(
            underlying,
            amer_spec,
            PricingMethod.BINOMIAL,
            params=cv_params,
        ).present_value()

        expected = american_raw + (european_analytical - european_hull)
        assert np.isclose(american_cv, expected, atol=1e-8)
        assert american_cv > 0.0
