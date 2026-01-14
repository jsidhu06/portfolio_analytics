"""Comprehensive tests for greek calculations (delta, gamma, vega)."""

import pytest
import datetime as dt
import numpy as np
from portfolio_analytics.valuation import (
    OptionSpec,
    UnderlyingData,
    OptionValuation,
)
from portfolio_analytics.enums import (
    OptionType,
    ExerciseType,
    PricingMethod,
    GreekCalculationMethod,
)
from portfolio_analytics.stochastic_processes import (
    GeometricBrownianMotion,
    GBMParams,
    SimulationConfig,
)
from portfolio_analytics.market_environment import MarketData
from portfolio_analytics.rates import ConstantShortRate


class TestGreeksSetup:
    """Base setup for greek tests with common parameters."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up common parameters for all tests."""
        self.pricing_date = dt.datetime(2025, 1, 1)
        self.maturity = dt.datetime(2026, 1, 1)
        self.spot = 100.0
        self.strike = 100.0
        self.rate = 0.05
        self.volatility = 0.20
        self.dividend_yield = 0.0
        self.time_to_maturity = 1.0

        # Discount curve
        self.csr = ConstantShortRate("csr", self.rate)

        # UnderlyingData for BSM and Binomial
        self.ud = UnderlyingData(
            initial_value=self.spot,
            volatility=self.volatility,
            pricing_date=self.pricing_date,
            discount_curve=self.csr,
            dividend_yield=self.dividend_yield,
        )

        # Market data and GBM for MCS
        self.market_data = MarketData(self.pricing_date, self.csr, currency="USD")
        sim_config = SimulationConfig(
            paths=200000,
            frequency="W",
            day_count_convention=365,
            final_date=self.maturity,
        )
        process_params = GBMParams(
            initial_value=self.spot,
            volatility=self.volatility,
            dividend_yield=self.dividend_yield,
        )
        self.gbm = GeometricBrownianMotion("gbm", self.market_data, process_params, sim_config)

        # GBM with dividend
        process_params_div = GBMParams(
            initial_value=self.spot,
            volatility=self.volatility,
            dividend_yield=0.03,
        )
        self.gbm_div = GeometricBrownianMotion(
            "gbm_div", self.market_data, process_params_div, sim_config
        )

    def _create_underlying_data(self, initial_value=None, volatility=None, dividend_yield=None):
        """Factory method to create UnderlyingData with sensible defaults.

        Args:
            initial_value: Override spot price, defaults to self.spot
            volatility: Override volatility, defaults to self.volatility
            dividend_yield: Override dividend yield, defaults to self.dividend_yield

        Returns:
            UnderlyingData instance with specified parameters
        """
        return UnderlyingData(
            initial_value=initial_value if initial_value is not None else self.spot,
            volatility=volatility if volatility is not None else self.volatility,
            pricing_date=self.pricing_date,
            discount_curve=self.csr,
            dividend_yield=dividend_yield if dividend_yield is not None else self.dividend_yield,
        )


class TestDeltaBasicProperties(TestGreeksSetup):
    """Test basic properties of delta values."""

    def test_call_delta_positive(self):
        """Test that call option delta is positive."""
        spec = OptionSpec(
            option_type=OptionType.CALL,
            exercise_type=ExerciseType.EUROPEAN,
            strike=self.strike,
            maturity=self.maturity,
            currency="USD",
        )
        valuation = OptionValuation("call_atm", self.ud, spec, PricingMethod.BSM_CONTINUOUS)
        delta = valuation.delta()
        assert delta > 0, "Call delta should be positive"

    def test_put_delta_negative(self):
        """Test that put option delta is negative."""
        spec = OptionSpec(
            option_type=OptionType.PUT,
            exercise_type=ExerciseType.EUROPEAN,
            strike=self.strike,
            maturity=self.maturity,
            currency="USD",
        )
        valuation = OptionValuation("put_atm", self.ud, spec, PricingMethod.BSM_CONTINUOUS)
        delta = valuation.delta()
        assert delta < 0, "Put delta should be negative"

    def test_call_delta_atm_greater_than_half(self):
        """Test that ATM call delta is > 0.5 (adjusted for positive r). Due to Ito's lemma.

        With positive risk-free rate, the forward is higher than spot, so ATM delta > 0.5.
        """
        spec = OptionSpec(
            option_type=OptionType.CALL,
            exercise_type=ExerciseType.EUROPEAN,
            strike=self.strike,  # ATM: spot == strike
            maturity=self.maturity,
            currency="USD",
        )
        valuation = OptionValuation("call_atm", self.ud, spec, PricingMethod.BSM_CONTINUOUS)
        delta = valuation.delta()
        # With positive r, delta should be > 0.5
        assert 0.5 < delta < 1.0, f"ATM call delta {delta} should be between 0.5 and 1.0"

    def test_call_delta_itm_close_to_one(self):
        """Test that deep ITM call delta approaches 1."""
        ud_itm = self._create_underlying_data(initial_value=150.0)  # Deep ITM
        spec = OptionSpec(
            option_type=OptionType.CALL,
            exercise_type=ExerciseType.EUROPEAN,
            strike=self.strike,
            maturity=self.maturity,
            currency="USD",
        )
        valuation = OptionValuation("call_itm", ud_itm, spec, PricingMethod.BSM_CONTINUOUS)
        delta = valuation.delta()
        assert delta > 0.95, f"Deep ITM call delta {delta} should be close to 1.0"

    def test_call_delta_otm_close_to_zero(self):
        """Test that deep OTM call delta approaches 0."""
        ud_otm = self._create_underlying_data(initial_value=50.0)  # Deep OTM
        spec = OptionSpec(
            option_type=OptionType.CALL,
            exercise_type=ExerciseType.EUROPEAN,
            strike=self.strike,
            maturity=self.maturity,
            currency="USD",
        )
        valuation = OptionValuation("call_otm", ud_otm, spec, PricingMethod.BSM_CONTINUOUS)
        delta = valuation.delta()
        assert delta < 0.05, f"Deep OTM call delta {delta} should be close to 0.0"

    def test_put_delta_itm_close_to_negative_one(self):
        """Test that deep ITM put delta approaches -1."""
        ud_itm = self._create_underlying_data(initial_value=50.0)  # Deep ITM for put
        spec = OptionSpec(
            option_type=OptionType.PUT,
            exercise_type=ExerciseType.EUROPEAN,
            strike=self.strike,
            maturity=self.maturity,
            currency="USD",
        )
        valuation = OptionValuation("put_itm", ud_itm, spec, PricingMethod.BSM_CONTINUOUS)
        delta = valuation.delta()
        assert delta < -0.95, f"Deep ITM put delta {delta} should be close to -1.0"

    def test_put_delta_otm_close_to_zero(self):
        """Test that deep OTM put delta approaches 0."""
        ud_otm = self._create_underlying_data(initial_value=150.0)  # Deep OTM for put
        spec = OptionSpec(
            option_type=OptionType.PUT,
            exercise_type=ExerciseType.EUROPEAN,
            strike=self.strike,
            maturity=self.maturity,
            currency="USD",
        )
        valuation = OptionValuation("put_otm", ud_otm, spec, PricingMethod.BSM_CONTINUOUS)
        delta = valuation.delta()
        assert delta > -0.05, f"Deep OTM put delta {delta} should be close to 0.0"


class TestGammaBasicProperties(TestGreeksSetup):
    """Test basic properties of gamma values."""

    def test_gamma_always_positive(self):
        """Test that gamma is always positive for calls and puts."""
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

        call_val = OptionValuation("call", self.ud, call_spec, PricingMethod.BSM_CONTINUOUS)
        put_val = OptionValuation("put", self.ud, put_spec, PricingMethod.BSM_CONTINUOUS)

        assert call_val.gamma() > 0, "Call gamma should be positive"
        assert put_val.gamma() > 0, "Put gamma should be positive"

    def test_gamma_highest_atm(self):
        """Test that gamma is highest for ATM options."""
        # ATM gamma
        ud_atm = self._create_underlying_data()
        spec_atm = OptionSpec(
            option_type=OptionType.CALL,
            exercise_type=ExerciseType.EUROPEAN,
            strike=self.strike,
            maturity=self.maturity,
            currency="USD",
        )
        val_atm = OptionValuation("call_atm", ud_atm, spec_atm, PricingMethod.BSM_CONTINUOUS)
        gamma_atm = val_atm.gamma()

        # ITM gamma
        ud_itm = self._create_underlying_data(initial_value=110.0)
        spec_itm = OptionSpec(
            option_type=OptionType.CALL,
            exercise_type=ExerciseType.EUROPEAN,
            strike=self.strike,
            maturity=self.maturity,
            currency="USD",
        )
        val_itm = OptionValuation("call_itm", ud_itm, spec_itm, PricingMethod.BSM_CONTINUOUS)
        gamma_itm = val_itm.gamma()

        # Gamma should be higher for ATM than ITM
        assert gamma_atm > gamma_itm, "ATM gamma should be higher than ITM gamma"

    def test_gamma_decreases_with_time(self):
        """Test that gamma decreases as time to maturity increases."""
        # Short time to maturity
        maturity_short = dt.datetime(2025, 3, 1)  # 2 months
        spec_short = OptionSpec(
            option_type=OptionType.CALL,
            exercise_type=ExerciseType.EUROPEAN,
            strike=self.strike,
            maturity=maturity_short,
            currency="USD",
        )
        val_short = OptionValuation("call_short", self.ud, spec_short, PricingMethod.BSM_CONTINUOUS)
        gamma_short = val_short.gamma()

        # Long time to maturity
        spec_long = OptionSpec(
            option_type=OptionType.CALL,
            exercise_type=ExerciseType.EUROPEAN,
            strike=self.strike,
            maturity=self.maturity,
            currency="USD",
        )
        val_long = OptionValuation("call_long", self.ud, spec_long, PricingMethod.BSM_CONTINUOUS)
        gamma_long = val_long.gamma()

        # Gamma should be higher for shorter time to maturity (at ATM)
        assert gamma_short > gamma_long, "Gamma should decrease as time to maturity increases"


class TestVegaBasicProperties(TestGreeksSetup):
    """Test basic properties of vega values."""

    def test_vega_always_positive(self):
        """Test that vega is always positive for calls and puts."""
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

        call_val = OptionValuation("call", self.ud, call_spec, PricingMethod.BSM_CONTINUOUS)
        put_val = OptionValuation("put", self.ud, put_spec, PricingMethod.BSM_CONTINUOUS)

        assert call_val.vega() > 0, "Call vega should be positive"
        assert put_val.vega() > 0, "Put vega should be positive"

    def test_vega_highest_atm(self):
        """Test that vega is highest for ATM options."""
        # ATM vega
        ud_atm = self._create_underlying_data()
        spec_atm = OptionSpec(
            option_type=OptionType.CALL,
            exercise_type=ExerciseType.EUROPEAN,
            strike=self.strike,
            maturity=self.maturity,
            currency="USD",
        )
        val_atm = OptionValuation("call_atm", ud_atm, spec_atm, PricingMethod.BSM_CONTINUOUS)
        vega_atm = val_atm.vega()

        # ITM vega
        ud_itm = self._create_underlying_data(initial_value=110.0)
        spec_itm = OptionSpec(
            option_type=OptionType.CALL,
            exercise_type=ExerciseType.EUROPEAN,
            strike=self.strike,
            maturity=self.maturity,
            currency="USD",
        )
        val_itm = OptionValuation("call_itm", ud_itm, spec_itm, PricingMethod.BSM_CONTINUOUS)
        vega_itm = val_itm.vega()

        # Vega should be higher for ATM than ITM
        assert vega_atm > vega_itm, "ATM vega should be higher than ITM vega"

    def test_vega_increases_with_time(self):
        """Test that vega increases as time to maturity increases."""
        # Short time to maturity
        maturity_short = dt.datetime(2025, 3, 1)  # 2 months
        spec_short = OptionSpec(
            option_type=OptionType.CALL,
            exercise_type=ExerciseType.EUROPEAN,
            strike=self.strike,
            maturity=maturity_short,
            currency="USD",
        )
        val_short = OptionValuation("call_short", self.ud, spec_short, PricingMethod.BSM_CONTINUOUS)
        vega_short = val_short.vega()

        # Long time to maturity
        spec_long = OptionSpec(
            option_type=OptionType.CALL,
            exercise_type=ExerciseType.EUROPEAN,
            strike=self.strike,
            maturity=self.maturity,
            currency="USD",
        )
        val_long = OptionValuation("call_long", self.ud, spec_long, PricingMethod.BSM_CONTINUOUS)
        vega_long = val_long.vega()

        # Vega should be higher for longer time to maturity (at ATM)
        assert vega_long > vega_short, "Vega should increase as time to maturity increases"


class TestGreekCalculationMethods(TestGreeksSetup):
    """Test analytical vs numerical greek calculation methods."""

    def test_bsm_analytical_vs_numerical_delta(self):
        """Test that analytical and numerical delta are close for BSM."""
        spec = OptionSpec(
            option_type=OptionType.CALL,
            exercise_type=ExerciseType.EUROPEAN,
            strike=self.strike,
            maturity=self.maturity,
            currency="USD",
        )
        valuation = OptionValuation("call", self.ud, spec, PricingMethod.BSM_CONTINUOUS)

        delta_analytical = valuation.delta(greek_calc_method=GreekCalculationMethod.ANALYTICAL)
        delta_numerical = valuation.delta(greek_calc_method=GreekCalculationMethod.NUMERICAL)

        # Should be very close
        assert np.isclose(
            delta_analytical, delta_numerical, rtol=1e-3
        ), f"Analytical {delta_analytical} vs numerical {delta_numerical}"

    def test_bsm_analytical_vs_numerical_gamma(self):
        """Test that analytical and numerical gamma are close for BSM."""
        spec = OptionSpec(
            option_type=OptionType.CALL,
            exercise_type=ExerciseType.EUROPEAN,
            strike=self.strike,
            maturity=self.maturity,
            currency="USD",
        )
        valuation = OptionValuation("call", self.ud, spec, PricingMethod.BSM_CONTINUOUS)

        gamma_analytical = valuation.gamma(greek_calc_method=GreekCalculationMethod.ANALYTICAL)
        gamma_numerical = valuation.gamma(greek_calc_method=GreekCalculationMethod.NUMERICAL)

        # Should be very close
        assert np.isclose(
            gamma_analytical, gamma_numerical, rtol=1e-3
        ), f"Analytical {gamma_analytical} vs numerical {gamma_numerical}"

    def test_bsm_analytical_vs_numerical_vega(self):
        """Test that analytical and numerical vega are close for BSM."""
        spec = OptionSpec(
            option_type=OptionType.CALL,
            exercise_type=ExerciseType.EUROPEAN,
            strike=self.strike,
            maturity=self.maturity,
            currency="USD",
        )
        valuation = OptionValuation("call", self.ud, spec, PricingMethod.BSM_CONTINUOUS)

        vega_analytical = valuation.vega(greek_calc_method=GreekCalculationMethod.ANALYTICAL)
        vega_numerical = valuation.vega(greek_calc_method=GreekCalculationMethod.NUMERICAL)

        # Should be very close
        assert np.isclose(
            vega_analytical, vega_numerical, rtol=1e-3
        ), f"Analytical {vega_analytical} vs numerical {vega_numerical}"


class TestGreekConsistencyAcrossPricingMethods(TestGreeksSetup):
    """Test that greeks are consistent across different pricing methods."""

    def test_call_delta_consistency_bsm_binomial(self):
        """Test call delta consistency between BSM and Binomial."""
        spec = OptionSpec(
            option_type=OptionType.CALL,
            exercise_type=ExerciseType.EUROPEAN,
            strike=self.strike,
            maturity=self.maturity,
            currency="USD",
        )

        bsm_val = OptionValuation("call_bsm", self.ud, spec, PricingMethod.BSM_CONTINUOUS)
        delta_bsm = bsm_val.delta()

        binomial_val = OptionValuation("call_binomial", self.ud, spec, PricingMethod.BINOMIAL)
        delta_binomial = binomial_val.delta(num_steps=2500)

        # Should be close within tolerance
        assert np.isclose(
            delta_bsm, delta_binomial, atol=0.01
        ), f"BSM delta {delta_bsm} vs Binomial {delta_binomial}"

    def test_call_delta_consistency_bsm_mcs(self):
        """Test call delta consistency between BSM and Monte Carlo."""
        spec = OptionSpec(
            option_type=OptionType.CALL,
            exercise_type=ExerciseType.EUROPEAN,
            strike=self.strike,
            maturity=self.maturity,
            currency="USD",
        )

        bsm_val = OptionValuation("call_bsm", self.ud, spec, PricingMethod.BSM_CONTINUOUS)
        delta_bsm = bsm_val.delta()

        mcs_val = OptionValuation("call_mcs", self.gbm, spec, PricingMethod.MONTE_CARLO)
        delta_mcs = mcs_val.delta(random_seed=42)

        # Should be close within tolerance (larger tolerance for MCS)
        assert np.isclose(
            delta_bsm, delta_mcs, atol=0.03
        ), f"BSM delta {delta_bsm} vs MCS {delta_mcs}"

    def test_gamma_consistency_bsm_binomial(self):
        """Test gamma consistency between BSM and Binomial."""
        spec = OptionSpec(
            option_type=OptionType.CALL,
            exercise_type=ExerciseType.EUROPEAN,
            strike=self.strike,
            maturity=self.maturity,
            currency="USD",
        )

        bsm_val = OptionValuation("call_bsm", self.ud, spec, PricingMethod.BSM_CONTINUOUS)
        gamma_bsm = bsm_val.gamma()

        binomial_val = OptionValuation("call_binomial", self.ud, spec, PricingMethod.BINOMIAL)
        gamma_binomial = binomial_val.gamma(num_steps=2500)

        # Should be close within tolerance (binomial is numerical so tolerance is larger)
        assert np.isclose(
            gamma_bsm, gamma_binomial, atol=0.005
        ), f"BSM gamma {gamma_bsm} vs Binomial {gamma_binomial}"

    def test_vega_consistency_bsm_binomial(self):
        """Test vega consistency between BSM and Binomial."""
        spec = OptionSpec(
            option_type=OptionType.CALL,
            exercise_type=ExerciseType.EUROPEAN,
            strike=self.strike,
            maturity=self.maturity,
            currency="USD",
        )

        bsm_val = OptionValuation("call_bsm", self.ud, spec, PricingMethod.BSM_CONTINUOUS)
        vega_bsm = bsm_val.vega()

        binomial_val = OptionValuation("call_binomial", self.ud, spec, PricingMethod.BINOMIAL)
        vega_binomial = binomial_val.vega(num_steps=2500)

        # Should be close within tolerance
        assert np.isclose(
            vega_bsm, vega_binomial, atol=0.1
        ), f"BSM vega {vega_bsm} vs Binomial {vega_binomial}"


class TestGreeksDividendYieldEffect(TestGreeksSetup):
    """Test the effect of dividend yield on greeks."""

    def test_call_delta_lower_with_dividend_yield(self):
        """Test that call delta is lower with dividend yield."""
        spec = OptionSpec(
            option_type=OptionType.CALL,
            exercise_type=ExerciseType.EUROPEAN,
            strike=self.strike,
            maturity=self.maturity,
            currency="USD",
        )

        # No dividend yield
        val_no_div = OptionValuation("call_no_div", self.ud, spec, PricingMethod.BSM_CONTINUOUS)
        delta_no_div = val_no_div.delta()

        # With dividend yield
        ud_with_div = self._create_underlying_data(dividend_yield=0.03)
        val_with_div = OptionValuation(
            "call_with_div", ud_with_div, spec, PricingMethod.BSM_CONTINUOUS
        )
        delta_with_div = val_with_div.delta()

        # Delta should be lower with dividend yield (dividend reduces expected spot growth)
        assert delta_with_div < delta_no_div, (
            f"Call delta with dividend {delta_with_div} should be "
            f"lower than without dividend {delta_no_div}"
        )

    def test_put_delta_more_negative_with_dividend_yield(self):
        """Test that put delta becomes more negative with dividend yield.

        Higher dividend yield reduces expected spot growth, making puts more valuable
        (and thus more negative in delta).
        """
        spec = OptionSpec(
            option_type=OptionType.PUT,
            exercise_type=ExerciseType.EUROPEAN,
            strike=self.strike,
            maturity=self.maturity,
            currency="USD",
        )

        # No dividend yield
        val_no_div = OptionValuation("put_no_div", self.ud, spec, PricingMethod.BSM_CONTINUOUS)
        delta_no_div = val_no_div.delta()

        # With dividend yield
        ud_with_div = self._create_underlying_data(dividend_yield=0.03)
        val_with_div = OptionValuation(
            "put_with_div", ud_with_div, spec, PricingMethod.BSM_CONTINUOUS
        )
        delta_with_div = val_with_div.delta()

        # Put delta should be more negative (further from 0) with dividend yield
        assert delta_with_div < delta_no_div, (
            f"Put delta with dividend {delta_with_div} should be "
            f"lower (more negative) than without dividend {delta_no_div}"
        )


class TestGreekErrorHandling(TestGreeksSetup):
    """Test error handling for greek calculations."""

    def test_analytical_greek_with_non_bsm_raises_error(self):
        """Test that requesting analytical greeks with non-BSM method raises ValueError."""
        spec = OptionSpec(
            option_type=OptionType.CALL,
            exercise_type=ExerciseType.EUROPEAN,
            strike=self.strike,
            maturity=self.maturity,
            currency="USD",
        )

        valuation = OptionValuation("call_binomial", self.ud, spec, PricingMethod.BINOMIAL)

        with pytest.raises(
            ValueError, match="Analytical greeks are only available for BSM_CONTINUOUS"
        ):
            valuation.delta(greek_calc_method=GreekCalculationMethod.ANALYTICAL)

    def test_invalid_greek_calc_method_type_raises_error(self):
        """Test that invalid greek_calc_method type raises TypeError."""
        spec = OptionSpec(
            option_type=OptionType.CALL,
            exercise_type=ExerciseType.EUROPEAN,
            strike=self.strike,
            maturity=self.maturity,
            currency="USD",
        )

        valuation = OptionValuation("call_bsm", self.ud, spec, PricingMethod.BSM_CONTINUOUS)

        with pytest.raises(
            TypeError, match="greek_calc_method must be GreekCalculationMethod enum"
        ):
            valuation.delta(greek_calc_method="analytical")  # Should be enum, not string
