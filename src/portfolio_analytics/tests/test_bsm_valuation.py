"""Tests for Black-Scholes-Merton valuation implementation."""

import datetime as dt

import numpy as np

from portfolio_analytics.enums import ExerciseType, OptionType, PricingMethod
from portfolio_analytics.market_environment import MarketData
from portfolio_analytics.tests.conftest import (
    PRICING_DATE,
    MATURITY,
    CURRENCY,
    SPOT,
    STRIKE,
    RATE,
    VOL,
)
from portfolio_analytics.tests.helpers import flat_curve, pv
from portfolio_analytics.valuation import (
    VanillaSpec,
    OptionValuation,
    UnderlyingData,
)

# ---------------------------------------------------------------------------
# Module-level helpers (test_edge_cases style)
# ---------------------------------------------------------------------------

_CURVE = flat_curve(PRICING_DATE, MATURITY, RATE)
_MD = MarketData(PRICING_DATE, _CURVE, currency=CURRENCY)


def _underlying(
    spot: float = SPOT,
    vol: float = VOL,
    dividend_curve=None,
    discrete_dividends=None,
) -> UnderlyingData:
    return UnderlyingData(
        initial_value=spot,
        volatility=vol,
        market_data=_MD,
        dividend_curve=dividend_curve,
        discrete_dividends=discrete_dividends,
    )


def _spec(
    option_type: OptionType = OptionType.CALL,
    exercise: ExerciseType = ExerciseType.EUROPEAN,
    strike: float = STRIKE,
) -> VanillaSpec:
    return VanillaSpec(
        option_type=option_type,
        exercise_type=exercise,
        strike=strike,
        maturity=MATURITY,
        currency=CURRENCY,
    )


def _bsm(ud: UnderlyingData, spec: VanillaSpec) -> float:
    return pv(ud, spec, PricingMethod.BSM)


class TestBSMValuation:
    """Tests for Black-Scholes-Merton valuation implementation."""

    def test_bsm_call_option_atm(self):
        """Test BSM pricing for ATM call option (basic sanity check)."""
        result = _bsm(_underlying(), _spec())
        assert result > 0
        assert np.isclose(result, 10.45, rtol=0.01)

    def test_bsm_discrete_dividends_reduce_call_price(self):
        """Discrete dividends should reduce European call price (all else equal)."""
        spec = _spec()
        pv_no_div = _bsm(_underlying(), spec)
        pv_div = _bsm(
            _underlying(discrete_dividends=[(PRICING_DATE + dt.timedelta(days=180), 1.0)]),
            spec,
        )
        assert pv_div < pv_no_div

    def test_bsm_put_option_atm(self):
        """Test BSM pricing for ATM put option."""
        result = _bsm(_underlying(), _spec(OptionType.PUT))
        assert result > 0
        assert np.isclose(result, 5.57, rtol=0.01)

    def test_bsm_call_itm(self):
        """Test BSM call option in-the-money."""
        result = _bsm(_underlying(spot=110.0), _spec())
        intrinsic = (110.0 - STRIKE) * np.exp(-RATE * 1.0)
        assert result >= intrinsic * 0.95

    def test_bsm_put_otm(self):
        """Test BSM put option out-of-the-money."""
        result = _bsm(_underlying(spot=110.0), _spec(OptionType.PUT))
        assert result > 0
        assert result < 5.0

    def test_bsm_with_dividend_curve(self):
        """Test BSM pricing with dividend curve."""
        q_curve = flat_curve(PRICING_DATE, MATURITY, 0.03)
        spec = _spec()
        pv_no_div = _bsm(_underlying(), spec)
        pv_with_div = _bsm(_underlying(dividend_curve=q_curve), spec)
        assert pv_with_div < pv_no_div

    def test_bsm_call_put_parity(self):
        """Test BSM call-put parity: C - P = S*exp(-q*T) - K*exp(-r*T)."""
        ud = _underlying()
        call_price = _bsm(ud, _spec(OptionType.CALL))
        put_price = _bsm(_underlying(), _spec(OptionType.PUT))

        T = 1.0
        parity_rhs = SPOT * np.exp(-0.0 * T) - STRIKE * np.exp(-RATE * T)
        assert np.isclose(call_price - put_price, parity_rhs, rtol=1e-10)

    def test_bsm_present_value_returns_float(self):
        """Test BSM present_value returns a scalar float."""
        valuation = OptionValuation(_underlying(), _spec(), PricingMethod.BSM)
        result = valuation.present_value()
        assert isinstance(result, float)
