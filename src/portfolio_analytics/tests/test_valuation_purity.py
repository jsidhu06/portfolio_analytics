# tests/test_valuation_purity.py
"""Purity / non-mutation tests for OptionValuation.

These tests assert that:
- present_value() does not mutate UnderlyingPricingData (or curves)
- greeks (delta/gamma/vega) may bump parameters internally, but must restore state

Rationale
---------
Users will commonly reuse one UnderlyingPricingData instance across multiple valuations/methods.
The library should guarantee that pricing/greeks do not leave surprising side-effects behind.
"""

import datetime as dt

import pytest

from portfolio_analytics.enums import (
    ExerciseType,
    OptionType,
    PricingMethod,
)
from portfolio_analytics.market_environment import MarketData
from portfolio_analytics.rates import ConstantShortRate
from portfolio_analytics.valuation import OptionSpec, OptionValuation, UnderlyingPricingData
from portfolio_analytics.valuation import BinomialParams


def _make_ud(
    *,
    spot: float = 100.0,
    vol: float = 0.20,
    pricing_date: dt.datetime,
    discount_curve: ConstantShortRate,
    currency: str = "USD",
    dividend_yield: float = 0.0,
) -> UnderlyingPricingData:
    market_data = MarketData(pricing_date, discount_curve, currency=currency)
    return UnderlyingPricingData(
        initial_value=spot,
        volatility=vol,
        market_data=market_data,
        dividend_yield=dividend_yield,
    )


def _make_spec(
    *,
    option_type: OptionType,
    strike: float,
    maturity: dt.datetime,
    currency: str = "USD",
    exercise_type: ExerciseType = ExerciseType.EUROPEAN,
) -> OptionSpec:
    return OptionSpec(
        option_type=option_type,
        exercise_type=exercise_type,
        strike=strike,
        maturity=maturity,
        currency=currency,
    )


def _snapshot_ud(ud: UnderlyingPricingData) -> dict:
    """Capture the fields we expect not to change after pricing/greeks."""
    # Add to this if UnderlyingPricingData grows.
    return {
        "initial_value": ud.initial_value,
        "volatility": ud.volatility,
        "pricing_date": ud.pricing_date,
        "dividend_yield": ud.dividend_yield,
        # identity + rate (curve object should remain same & rate unchanged)
        "discount_curve_id": id(ud.discount_curve),
        "discount_curve_rate": float(ud.discount_curve.short_rate),
    }


@pytest.fixture()
def common_setup():
    pricing_date = dt.datetime(2025, 1, 1)
    maturity = dt.datetime(2026, 1, 1)
    rate = 0.05
    csr = ConstantShortRate("csr", rate)
    return pricing_date, maturity, csr


class TestValuationPurityPresentValue:
    def test_present_value_does_not_mutate_underlyingdata_across_methods(self, common_setup):
        """Reusing the same UnderlyingPricingData across pricing methods should be safe.

        This test intentionally reuses *one* UnderlyingPricingData instance to match typical user behaviour.
        """
        pricing_date, maturity, csr = common_setup

        ud = _make_ud(
            spot=100.0,
            vol=0.20,
            pricing_date=pricing_date,
            discount_curve=csr,
            dividend_yield=0.02,
        )
        spec = _make_spec(option_type=OptionType.CALL, strike=100.0, maturity=maturity)

        baseline = _snapshot_ud(ud)

        # BSM pricing
        bsm = OptionValuation("call_bsm", ud, spec, PricingMethod.BSM)
        _ = bsm.present_value()
        assert _snapshot_ud(ud) == baseline, "present_value() (BSM) mutated UnderlyingPricingData"

        # Binomial pricing (re-using same ud)
        tree = OptionValuation("call_tree", ud, spec, PricingMethod.BINOMIAL)
        _ = tree.present_value(params=BinomialParams(num_steps=2000))
        assert (
            _snapshot_ud(ud) == baseline
        ), "present_value() (Binomial) mutated UnderlyingPricingData"


class TestValuationPurityGreeks:
    def test_delta_restores_underlying_spot(self, common_setup):
        pricing_date, maturity, csr = common_setup

        ud = _make_ud(spot=100.0, vol=0.20, pricing_date=pricing_date, discount_curve=csr)
        spec = _make_spec(option_type=OptionType.CALL, strike=100.0, maturity=maturity)
        val = OptionValuation("call", ud, spec, PricingMethod.BSM)

        baseline = _snapshot_ud(ud)
        _ = val.delta()  # may bump internally
        assert _snapshot_ud(ud) == baseline, "delta() did not restore UnderlyingPricingData state"

    def test_gamma_restores_underlying_spot(self, common_setup):
        pricing_date, maturity, csr = common_setup

        ud = _make_ud(spot=100.0, vol=0.20, pricing_date=pricing_date, discount_curve=csr)
        spec = _make_spec(option_type=OptionType.CALL, strike=100.0, maturity=maturity)
        val = OptionValuation("call", ud, spec, PricingMethod.BSM)

        baseline = _snapshot_ud(ud)
        _ = val.gamma()  # may bump internally
        assert _snapshot_ud(ud) == baseline, "gamma() did not restore UnderlyingPricingData state"

    def test_vega_restores_underlying_volatility(self, common_setup):
        pricing_date, maturity, csr = common_setup

        ud = _make_ud(spot=100.0, vol=0.20, pricing_date=pricing_date, discount_curve=csr)
        spec = _make_spec(option_type=OptionType.CALL, strike=100.0, maturity=maturity)
        val = OptionValuation("call", ud, spec, PricingMethod.BSM)

        baseline = _snapshot_ud(ud)
        _ = val.vega()  # may bump internally
        assert _snapshot_ud(ud) == baseline, "vega() did not restore UnderlyingPricingData state"

    def test_binomial_greeks_do_not_mutate_underlyingdata(self, common_setup):
        """Binomial greeks are still numerical bumps; ensure restoration works there too."""
        pricing_date, maturity, csr = common_setup

        ud = _make_ud(spot=100.0, vol=0.20, pricing_date=pricing_date, discount_curve=csr)
        spec = _make_spec(option_type=OptionType.CALL, strike=100.0, maturity=maturity)
        val = OptionValuation("call_tree", ud, spec, PricingMethod.BINOMIAL)

        baseline = _snapshot_ud(ud)

        _ = val.delta(params=BinomialParams(num_steps=2000))
        assert (
            _snapshot_ud(ud) == baseline
        ), "Binomial delta() did not restore UnderlyingPricingData state"

        _ = val.gamma(params=BinomialParams(num_steps=2000))
        assert (
            _snapshot_ud(ud) == baseline
        ), "Binomial gamma() did not restore UnderlyingPricingData state"

        _ = val.vega(params=BinomialParams(num_steps=2000))
        assert (
            _snapshot_ud(ud) == baseline
        ), "Binomial vega() did not restore UnderlyingPricingData state"
