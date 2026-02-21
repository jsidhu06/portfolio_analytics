"""Tests for Binomial tree option valuation."""

import datetime as dt

import numpy as np

from portfolio_analytics.enums import DayCountConvention, ExerciseType, OptionType, PricingMethod
from portfolio_analytics.market_environment import MarketData
from portfolio_analytics.tests.helpers import flat_curve
from portfolio_analytics.utils import calculate_year_fraction, expected_binomial_payoff
from portfolio_analytics.valuation import (
    BinomialParams,
    OptionSpec,
    OptionValuation,
    UnderlyingPricingData,
)


class TestBinomialValuation:
    """Tests for Binomial tree option valuation."""

    def setup_method(self):
        """Set up market environment for binomial tests."""
        self.pricing_date = dt.datetime(2025, 1, 1)
        self.maturity = dt.datetime(2026, 1, 1)
        self.strike = 100.0
        self.spot = 100.0
        self.volatility = 0.2
        self.rate = 0.05

        self.curve = flat_curve(self.pricing_date, self.maturity, self.rate, name="csr")
        self.market_data = MarketData(self.pricing_date, self.curve, currency="USD")

        self.ud = UnderlyingPricingData(
            initial_value=self.spot,
            volatility=self.volatility,
            market_data=self.market_data,
        )

    def test_binomial_european_call_atm(self):
        """Test binomial European call option."""
        call_spec = OptionSpec(
            option_type=OptionType.CALL,
            exercise_type=ExerciseType.EUROPEAN,
            strike=self.strike,
            maturity=self.maturity,
            currency="USD",
        )

        valuation = OptionValuation(
            name="CALL_BIN",
            underlying=self.ud,
            spec=call_spec,
            pricing_method=PricingMethod.BINOMIAL,
            params=BinomialParams(num_steps=500),
        )

        pv = valuation.present_value()
        assert pv > 0
        # ATM call value should be approx 10.45 for these parameters
        assert np.isclose(pv, 10.45, rtol=0.01)

    def test_binomial_american_call_no_div_equal_to_european(self):
        """Test that American call >= European call (same parameters)."""
        eu_spec = OptionSpec(
            option_type=OptionType.CALL,
            exercise_type=ExerciseType.EUROPEAN,
            strike=self.strike,
            maturity=self.maturity,
            currency="USD",
        )

        am_spec = OptionSpec(
            option_type=OptionType.CALL,
            exercise_type=ExerciseType.AMERICAN,
            strike=self.strike,
            maturity=self.maturity,
            currency="USD",
        )

        ud_eu = UnderlyingPricingData(
            initial_value=self.spot,
            volatility=self.volatility,
            market_data=self.market_data,
        )

        ud_am = UnderlyingPricingData(
            initial_value=self.spot,
            volatility=self.volatility,
            market_data=self.market_data,
        )

        eu_val = OptionValuation(
            name="CALL_EU",
            underlying=ud_eu,
            spec=eu_spec,
            pricing_method=PricingMethod.BINOMIAL,
            params=BinomialParams(num_steps=500),
        )

        am_val = OptionValuation(
            name="CALL_AM",
            underlying=ud_am,
            spec=am_spec,
            pricing_method=PricingMethod.BINOMIAL,
            params=BinomialParams(num_steps=500),
        )

        eu_price = eu_val.present_value()
        am_price = am_val.present_value()

        # American should be >= European (early exercise premium)
        assert np.isclose(am_price, eu_price, rtol=0.005)

    def test_binomial_european_call_discrete_dividends_reduce_price(self):
        """Discrete dividends should reduce European call price in binomial tree."""
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

        val_no_div = OptionValuation(
            "call_no_div",
            ud_no_div,
            spec,
            PricingMethod.BINOMIAL,
            params=BinomialParams(num_steps=500),
        )
        val_div = OptionValuation(
            "call_div", ud_div, spec, PricingMethod.BINOMIAL, params=BinomialParams(num_steps=500)
        )

        pv_no_div = val_no_div.present_value()
        pv_div = val_div.present_value()

        assert pv_div < pv_no_div

    def test_binomial_american_put_early_exercise(self):
        """Test American put has early exercise premium."""
        eu_spec = OptionSpec(
            option_type=OptionType.PUT,
            exercise_type=ExerciseType.EUROPEAN,
            strike=self.strike,
            maturity=self.maturity,
            currency="USD",
        )

        am_spec = OptionSpec(
            option_type=OptionType.PUT,
            exercise_type=ExerciseType.AMERICAN,
            strike=self.strike,
            maturity=self.maturity,
            currency="USD",
        )

        ud_eu = UnderlyingPricingData(
            initial_value=self.spot,
            volatility=self.volatility,
            market_data=self.market_data,
        )

        ud_am = UnderlyingPricingData(
            initial_value=self.spot,
            volatility=self.volatility,
            market_data=self.market_data,
        )

        eu_val = OptionValuation(
            name="PUT_EU",
            underlying=ud_eu,
            spec=eu_spec,
            pricing_method=PricingMethod.BINOMIAL,
            params=BinomialParams(num_steps=100),
        )

        am_val = OptionValuation(
            name="PUT_AM",
            underlying=ud_am,
            spec=am_spec,
            pricing_method=PricingMethod.BINOMIAL,
            params=BinomialParams(num_steps=100),
        )

        eu_price = eu_val.present_value()
        am_price = am_val.present_value()

        # American put should be strictly greater (has early exercise value)
        assert am_price > eu_price

    def test_binomial_convergence(self):
        """Test that binomial prices converge with more steps."""
        call_spec = OptionSpec(
            option_type=OptionType.CALL,
            exercise_type=ExerciseType.EUROPEAN,
            strike=self.strike,
            maturity=self.maturity,
            currency="USD",
        )

        ud1 = UnderlyingPricingData(
            initial_value=self.spot,
            volatility=self.volatility,
            market_data=self.market_data,
        )

        ud2 = UnderlyingPricingData(
            initial_value=self.spot,
            volatility=self.volatility,
            market_data=self.market_data,
        )

        val1 = OptionValuation(
            name="CALL_100",
            underlying=ud1,
            spec=call_spec,
            pricing_method=PricingMethod.BINOMIAL,
            params=BinomialParams(num_steps=100),
        )

        val2 = OptionValuation(
            name="CALL_200",
            underlying=ud2,
            spec=call_spec,
            pricing_method=PricingMethod.BINOMIAL,
            params=BinomialParams(num_steps=200),
        )

        price_100 = val1.present_value()
        price_200 = val2.present_value()

        # Prices should be closer with more steps (convergence)
        # This is a weak test due to oscillation in binomial, but direction should be reasonable
        assert abs(price_200 - price_100) < 1.0

    def test_binomial_pv_matches_expected_binomial_payoff(self):
        n_steps = 250

        spec = OptionSpec(
            option_type=OptionType.CALL,
            exercise_type=ExerciseType.EUROPEAN,
            strike=self.strike,
            maturity=self.maturity,
            currency="USD",
        )
        valuation = OptionValuation(
            "call_binom",
            self.ud,
            spec,
            PricingMethod.BINOMIAL,
            params=BinomialParams(num_steps=n_steps),
        )
        pv_binom = valuation.present_value()

        T = calculate_year_fraction(
            self.pricing_date, self.maturity, day_count_convention=DayCountConvention.ACT_365F
        )
        dt_step = T / n_steps
        u = np.exp(self.volatility * np.sqrt(dt_step))

        expected_payoff = expected_binomial_payoff(
            S0=self.spot,
            n=n_steps,
            T=T,
            option_type=OptionType.CALL,
            K=self.strike,
            r=self.rate,
            q=0,
            u=u,
        )
        pv_expected = np.exp(-self.rate * T) * expected_payoff

        assert np.isclose(pv_binom, pv_expected, rtol=1.0e-4)
