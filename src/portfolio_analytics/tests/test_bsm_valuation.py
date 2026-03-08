"""Tests for Black-Scholes-Merton valuation implementation."""

import datetime as dt

import numpy as np

from portfolio_analytics.enums import OptionType, PricingMethod
from portfolio_analytics.tests.conftest import (
    PRICING_DATE,
    MATURITY,
    RATE,
    SPOT,
    STRIKE,
)
from portfolio_analytics.tests.helpers import flat_curve, pv, underlying, spec
from portfolio_analytics.valuation import (
    VanillaSpec,
    OptionValuation,
    UnderlyingData,
)

# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _bsm(ud: UnderlyingData, sp: VanillaSpec) -> float:
    return pv(ud, sp, PricingMethod.BSM)


class TestBSMValuation:
    """Tests for Black-Scholes-Merton valuation implementation."""

    def test_bsm_call_option_atm(self):
        """Test BSM pricing for ATM call option (basic sanity check)."""
        result = _bsm(underlying(), spec())
        assert result > 0
        assert np.isclose(result, 10.45, rtol=0.01)

    def test_bsm_discrete_dividends_reduce_call_price(self):
        """Discrete dividends should reduce European call price (all else equal)."""
        sp = spec()
        pv_no_div = _bsm(underlying(), sp)
        pv_div = _bsm(
            underlying(discrete_dividends=[(PRICING_DATE + dt.timedelta(days=180), 1.0)]),
            sp,
        )
        assert pv_div < pv_no_div

    def test_bsm_put_option_atm(self):
        """Test BSM pricing for ATM put option."""
        result = _bsm(underlying(), spec(OptionType.PUT))
        assert result > 0
        assert np.isclose(result, 5.57, rtol=0.01)

    def test_bsm_call_itm(self):
        """Test BSM call option in-the-money."""
        result = _bsm(underlying(spot=110.0), spec())
        intrinsic = (110.0 - STRIKE) * np.exp(-RATE * 1.0)
        assert result >= intrinsic * 0.95

    def test_bsm_put_otm(self):
        """Test BSM put option out-of-the-money."""
        result = _bsm(underlying(spot=110.0), spec(OptionType.PUT))
        assert result > 0
        assert result < 5.0

    def test_bsm_with_dividend_curve(self):
        """Test BSM pricing with dividend curve."""
        q_curve = flat_curve(PRICING_DATE, MATURITY, 0.03)
        sp = spec()
        pv_no_div = _bsm(underlying(), sp)
        pv_with_div = _bsm(underlying(dividend_curve=q_curve), sp)
        assert pv_with_div < pv_no_div

    def test_bsm_call_put_parity(self):
        """Test BSM call-put parity: C - P = S*exp(-q*T) - K*exp(-r*T)."""
        ud = underlying()
        call_price = _bsm(ud, spec(OptionType.CALL))
        put_price = _bsm(underlying(), spec(OptionType.PUT))

        T = 1.0
        parity_rhs = SPOT * np.exp(-0.0 * T) - STRIKE * np.exp(-RATE * T)
        assert np.isclose(call_price - put_price, parity_rhs, rtol=1e-10)

    def test_bsm_present_value_returns_float(self):
        """Test BSM present_value returns a scalar float."""
        valuation = OptionValuation(underlying(), spec(), PricingMethod.BSM)
        result = valuation.present_value()
        assert isinstance(result, float)
