"""Tests for Black-Scholes-Merton valuation implementation."""

import datetime as dt

import numpy as np

from portfolio_analytics.enums import ExerciseType, OptionType, PricingMethod
from portfolio_analytics.market_environment import MarketData
from portfolio_analytics.tests.helpers import flat_curve
from portfolio_analytics.valuation import (
    OptionSpec,
    OptionValuation,
    UnderlyingPricingData,
)


class TestBSMValuation:
    """Tests for Black-Scholes-Merton valuation implementation."""

    def setup_method(self):
        """Set up market environment for BSM tests."""
        self.pricing_date = dt.datetime(2025, 1, 1)
        self.maturity = dt.datetime(2026, 1, 1)  # 1 year
        self.strike = 100.0
        self.spot = 100.0
        self.volatility = 0.2
        self.rate = 0.05

        self.curve = flat_curve(self.pricing_date, self.maturity, self.rate)
        self.market_data = MarketData(self.pricing_date, self.curve, currency="USD")

        underlying_params = {
            "initial_value": self.spot,
            "volatility": self.volatility,
            "market_data": self.market_data,
        }

        self.ud = UnderlyingPricingData(**underlying_params)

        self.ud_div = UnderlyingPricingData(
            **{
                **underlying_params,
                "dividend_curve": flat_curve(self.pricing_date, self.maturity, 0.03),
            }
        )

    def test_bsm_call_option_atm(self):
        """Test BSM pricing for ATM call option (basic sanity check)."""
        call_spec = OptionSpec(
            option_type=OptionType.CALL,
            exercise_type=ExerciseType.EUROPEAN,
            strike=self.strike,
            maturity=self.maturity,
            currency="USD",
        )

        valuation = OptionValuation(
            underlying=self.ud,
            spec=call_spec,
            pricing_method=PricingMethod.BSM,
        )

        pv = valuation.present_value()

        # ATM call should have positive value
        assert pv > 0
        # ATM call value should be approx 10.45 for these parameters
        assert np.isclose(pv, 10.45, rtol=0.01)

    def test_bsm_discrete_dividends_reduce_call_price(self):
        """Discrete dividends should reduce European call price (all else equal)."""
        spec = OptionSpec(
            option_type=OptionType.CALL,
            exercise_type=ExerciseType.EUROPEAN,
            strike=self.strike,
            maturity=self.maturity,
            currency="USD",
        )

        ud_no_div = UnderlyingPricingData(
            initial_value=self.spot,
            volatility=self.volatility,
            market_data=self.market_data,
            dividend_curve=None,
            discrete_dividends=[],
        )

        ud_div = UnderlyingPricingData(
            initial_value=self.spot,
            volatility=self.volatility,
            market_data=self.market_data,
            dividend_curve=None,
            discrete_dividends=[(self.pricing_date + dt.timedelta(days=180), 1.0)],
        )

        val_no_div = OptionValuation(ud_no_div, spec, PricingMethod.BSM)
        val_div = OptionValuation(ud_div, spec, PricingMethod.BSM)

        pv_no_div = val_no_div.present_value()
        pv_div = val_div.present_value()

        assert pv_div < pv_no_div

    def test_bsm_put_option_atm(self):
        """Test BSM pricing for ATM put option."""
        put_spec = OptionSpec(
            option_type=OptionType.PUT,
            exercise_type=ExerciseType.EUROPEAN,
            strike=self.strike,
            maturity=self.maturity,
            currency="USD",
        )

        valuation = OptionValuation(
            underlying=self.ud,
            spec=put_spec,
            pricing_method=PricingMethod.BSM,
        )

        pv = valuation.present_value()

        # ATM put should have positive value
        assert pv > 0

        assert np.isclose(pv, 5.57, rtol=0.01)

    def test_bsm_call_itm(self):
        """Test BSM call option in-the-money."""
        ud_itm = UnderlyingPricingData(
            initial_value=110.0,
            volatility=self.volatility,
            market_data=self.market_data,
        )

        call_spec = OptionSpec(
            option_type=OptionType.CALL,
            exercise_type=ExerciseType.EUROPEAN,
            strike=self.strike,
            maturity=self.maturity,
            currency="USD",
        )

        valuation = OptionValuation(
            underlying=ud_itm,
            spec=call_spec,
            pricing_method=PricingMethod.BSM,
        )

        pv = valuation.present_value()

        # ITM call should be worth at least intrinsic value (discounted)
        intrinsic = (110.0 - self.strike) * np.exp(-self.rate * 1.0)
        assert pv >= intrinsic * 0.95  # allow small tolerance for numerical issues

    def test_bsm_put_otm(self):
        """Test BSM put option out-of-the-money."""
        ud_otm = UnderlyingPricingData(
            initial_value=110.0,
            volatility=self.volatility,
            market_data=self.market_data,
        )

        put_spec = OptionSpec(
            option_type=OptionType.PUT,
            exercise_type=ExerciseType.EUROPEAN,
            strike=self.strike,
            maturity=self.maturity,
            currency="USD",
        )

        valuation = OptionValuation(
            underlying=ud_otm,
            spec=put_spec,
            pricing_method=PricingMethod.BSM,
        )

        pv = valuation.present_value()

        # OTM put should have some value due to time value
        assert pv > 0
        # For OTM put, intrinsic is 0 (spot > strike), so time value should be small
        # Put value for OTM should be less than the case when ITM
        assert pv < 5.0

    def test_bsm_with_dividend_curve(self):
        """Test BSM pricing with dividend curve."""
        call_spec = OptionSpec(
            option_type=OptionType.CALL,
            exercise_type=ExerciseType.EUROPEAN,
            strike=self.strike,
            maturity=self.maturity,
            currency="USD",
        )

        valuation = OptionValuation(
            underlying=self.ud,
            spec=call_spec,
            pricing_method=PricingMethod.BSM,
        )

        valuation_div = OptionValuation(
            underlying=self.ud_div,
            spec=call_spec,
            pricing_method=PricingMethod.BSM,
        )

        pv_no_div = valuation.present_value()
        pv_with_div = valuation_div.present_value()

        # Call value should decrease with dividend yield
        assert pv_with_div < pv_no_div

    def test_bsm_call_put_parity(self):
        """Test BSM call-put parity: C - P = S*exp(-q*T) - K*exp(-r*T)."""
        call_spec = OptionSpec(
            option_type=OptionType.CALL,
            exercise_type=ExerciseType.EUROPEAN,
            strike=self.strike,
            maturity=self.maturity,
            currency="USD",
        )

        put_spec = OptionSpec(
            option_type=OptionType.PUT,
            exercise_type=ExerciseType.EUROPEAN,
            strike=self.strike,
            maturity=self.maturity,
            currency="USD",
        )

        call_val = OptionValuation(
            underlying=self.ud,
            spec=call_spec,
            pricing_method=PricingMethod.BSM,
        )

        # Create new underlying for put to avoid state pollution
        ud_put = UnderlyingPricingData(
            initial_value=self.spot,
            volatility=self.volatility,
            market_data=self.market_data,
        )

        put_val = OptionValuation(
            underlying=ud_put,
            spec=put_spec,
            pricing_method=PricingMethod.BSM,
        )

        call_price = call_val.present_value()
        put_price = put_val.present_value()

        # Put-call parity with no dividend
        T = 1.0
        parity_rhs = self.spot * np.exp(-0.0 * T) - self.strike * np.exp(-self.rate * T)

        assert np.isclose(call_price - put_price, parity_rhs, rtol=1e-10)

    def test_bsm_present_value_returns_float(self):
        """Test BSM present_value returns a scalar float."""
        call_spec = OptionSpec(
            option_type=OptionType.CALL,
            exercise_type=ExerciseType.EUROPEAN,
            strike=self.strike,
            maturity=self.maturity,
            currency="USD",
        )

        valuation = OptionValuation(
            underlying=self.ud,
            spec=call_spec,
            pricing_method=PricingMethod.BSM,
        )

        result = valuation.present_value()
        assert isinstance(result, float)
