"""Comprehensive tests for valuation modules (BSM, Binomial, MCS)."""

import pytest
import datetime as dt
import numpy as np
from portfolio_analytics.valuation import (
    OptionSpec,
    UnderlyingConfig,
    UnderlyingData,
    OptionValuation,
)
from portfolio_analytics.valuation_bsm import _BSMEuropeanValuation
from portfolio_analytics.valuation_binomial import (
    _BinomialEuropeanValuation,
)
from portfolio_analytics.valuation_mcs import _MCEuropeanValuation
from portfolio_analytics.enums import OptionType, ExerciseType, PricingMethod
from portfolio_analytics.stochastic_processes import (
    GeometricBrownianMotion,
    GBMParams,
    SimulationConfig,
)
from portfolio_analytics.market_environment import MarketData
from portfolio_analytics.rates import ConstantShortRate


class TestOptionSpec:
    """Tests for OptionSpec dataclass."""

    def test_valid_option_spec_creation(self):
        """Test successful creation of valid OptionSpec."""
        spec = OptionSpec(
            option_type=OptionType.CALL,
            exercise_type=ExerciseType.EUROPEAN,
            strike=100.0,
            maturity=dt.datetime(2026, 12, 31),
            currency="USD",
            contract_size=100,
        )
        assert spec.option_type == OptionType.CALL
        assert spec.exercise_type == ExerciseType.EUROPEAN
        assert spec.strike == 100.0
        assert spec.contract_size == 100

    def test_option_spec_with_none_strike(self):
        """Test OptionSpec with None strike for strike-less products."""
        spec = OptionSpec(
            option_type=OptionType.CALL,
            exercise_type=ExerciseType.EUROPEAN,
            strike=None,
            maturity=dt.datetime(2026, 12, 31),
            currency="EUR",
        )
        assert spec.strike is None

    def test_option_spec_invalid_option_type(self):
        """Test that invalid option_type raises TypeError."""
        with pytest.raises(TypeError, match="option_type must be OptionType enum"):
            OptionSpec(
                option_type="CALL",  # Invalid: string instead of enum
                exercise_type=ExerciseType.EUROPEAN,
                strike=100.0,
                maturity=dt.datetime(2026, 12, 31),
                currency="USD",
            )

    def test_option_spec_invalid_exercise_type(self):
        """Test that invalid exercise_type raises TypeError."""
        with pytest.raises(TypeError, match="exercise_type must be ExerciseType enum"):
            OptionSpec(
                option_type=OptionType.PUT,
                exercise_type="EUROPEAN",  # Invalid: string instead of enum
                strike=100.0,
                maturity=dt.datetime(2026, 12, 31),
                currency="USD",
            )

    def test_option_spec_frozen(self):
        """Test that OptionSpec is frozen (immutable)."""
        spec = OptionSpec(
            option_type=OptionType.CALL,
            exercise_type=ExerciseType.EUROPEAN,
            strike=100.0,
            maturity=dt.datetime(2026, 12, 31),
            currency="USD",
        )
        with pytest.raises(AttributeError):
            spec.strike = 105.0


class TestUnderlyingConfig:
    """Tests for UnderlyingConfig dataclass."""

    def test_gbm_config_creation(self):
        """Test UnderlyingConfig for GBM model."""
        config = UnderlyingConfig(
            name="STOCK",
            model="gbm",
            initial_value=100.0,
            volatility=0.2,
        )
        assert config.name == "STOCK"
        assert config.model == "gbm"
        assert config.initial_value == 100.0
        assert config.jump_intensity is None
        assert config.kappa is None

    def test_jd_config_creation(self):
        """Test UnderlyingConfig for Jump Diffusion model."""
        config = UnderlyingConfig(
            name="STOCK_JD",
            model="jd",
            initial_value=100.0,
            volatility=0.2,
            jump_intensity=0.5,
            jump_mean=-0.1,
            jump_std=0.3,
        )
        assert config.model == "jd"
        assert config.jump_intensity == 0.5
        assert config.jump_mean == -0.1
        assert config.jump_std == 0.3

    def test_srd_config_creation(self):
        """Test UnderlyingConfig for Square Root Diffusion model."""
        config = UnderlyingConfig(
            name="RATE",
            model="srd",
            initial_value=0.03,
            volatility=0.015,
            kappa=0.2,
            theta=0.05,
        )
        assert config.model == "srd"
        assert config.kappa == 0.2
        assert config.theta == 0.05


class TestUnderlyingData:
    """Tests for UnderlyingData class."""

    def setup_method(self):
        """Set up market environment for tests."""
        self.csr = ConstantShortRate("csr", 0.05)
        self.pricing_date = dt.datetime(2025, 1, 1)
        self.market_data = MarketData(self.pricing_date, self.csr, currency="USD")

    def test_underlying_data_creation(self):
        """Test successful creation of UnderlyingData."""
        ud = UnderlyingData(
            initial_value=100.0,
            volatility=0.2,
            pricing_date=self.pricing_date,
            discount_curve=self.csr,
        )
        assert ud.initial_value == 100.0
        assert ud.volatility == 0.2
        assert ud.pricing_date == self.pricing_date

    def test_underlying_data_attributes_mutable(self):
        """Test that UnderlyingData attributes can be modified (unlike frozen dataclasses)."""
        ud = UnderlyingData(
            initial_value=100.0,
            volatility=0.2,
            pricing_date=self.pricing_date,
            discount_curve=self.csr,
        )
        # Should be able to modify attributes
        ud.initial_value = 105.0
        assert ud.initial_value == 105.0


class TestOptionValuation:
    """Tests for OptionValuation dispatcher class."""

    def setup_method(self):
        """Set up market environment and valuation specifications for tests."""
        self.pricing_date = dt.datetime(2025, 1, 1)
        self.maturity = dt.datetime(2026, 1, 1)
        self.strike = 100.0
        self.csr = ConstantShortRate("csr", 0.05)
        self.market_data = MarketData(self.pricing_date, self.csr, currency="USD")

        # Standard option spec
        self.call_spec = OptionSpec(
            option_type=OptionType.CALL,
            exercise_type=ExerciseType.EUROPEAN,
            strike=self.strike,
            maturity=self.maturity,
            currency="USD",
        )

        self.put_spec = OptionSpec(
            option_type=OptionType.PUT,
            exercise_type=ExerciseType.EUROPEAN,
            strike=self.strike,
            maturity=self.maturity,
            currency="USD",
        )

    def test_option_valuation_with_underlying_data_bsm(self):
        """Test OptionValuation creation with UnderlyingData and BSM pricing."""
        ud = UnderlyingData(
            initial_value=100.0,
            volatility=0.2,
            pricing_date=self.pricing_date,
            discount_curve=self.csr,
        )

        valuation = OptionValuation(
            name="CALL_BSM",
            underlying=ud,
            spec=self.call_spec,
            pricing_method=PricingMethod.BSM_CONTINUOUS,
        )
        assert valuation.name == "CALL_BSM"
        assert valuation.strike == self.strike
        assert isinstance(valuation._impl, _BSMEuropeanValuation)

    def test_option_valuation_with_underlying_data_binomial(self):
        """Test OptionValuation creation with UnderlyingData and Binomial pricing."""
        ud = UnderlyingData(
            initial_value=100.0,
            volatility=0.2,
            pricing_date=self.pricing_date,
            discount_curve=self.csr,
        )

        valuation = OptionValuation(
            name="CALL_BINOMIAL",
            underlying=ud,
            spec=self.call_spec,
            pricing_method=PricingMethod.BINOMIAL,
        )
        assert isinstance(valuation._impl, _BinomialEuropeanValuation)

    def test_option_valuation_with_path_simulation_mcs(self):
        """Test OptionValuation creation with PathSimulation and MC pricing."""
        gbm_params = GBMParams(initial_value=100.0, volatility=0.2)
        sim_config = SimulationConfig(
            paths=1000,
            frequency="D",
            final_date=self.maturity,
        )
        gbm = GeometricBrownianMotion(
            "gbm_test",
            self.market_data,
            gbm_params,
            sim_config,
        )

        valuation = OptionValuation(
            name="CALL_MCS",
            underlying=gbm,
            spec=self.call_spec,
            pricing_method=PricingMethod.MONTE_CARLO,
        )
        assert isinstance(valuation._impl, _MCEuropeanValuation)

    def test_option_valuation_invalid_maturity(self):
        """Test that OptionValuation raises error if maturity <= pricing_date."""
        ud = UnderlyingData(
            initial_value=100.0,
            volatility=0.2,
            pricing_date=self.pricing_date,
            discount_curve=self.csr,
        )

        invalid_spec = OptionSpec(
            option_type=OptionType.CALL,
            exercise_type=ExerciseType.EUROPEAN,
            strike=self.strike,
            maturity=self.pricing_date,  # Invalid: maturity == pricing_date
            currency="USD",
        )

        with pytest.raises(ValueError, match="Option maturity must be after pricing_date"):
            OptionValuation(
                name="INVALID",
                underlying=ud,
                spec=invalid_spec,
                pricing_method=PricingMethod.BSM_CONTINUOUS,
            )

    def test_option_valuation_invalid_pricing_method_type(self):
        """Test that OptionValuation validates pricing_method is PricingMethod enum."""
        ud = UnderlyingData(
            initial_value=100.0,
            volatility=0.2,
            pricing_date=self.pricing_date,
            discount_curve=self.csr,
        )

        with pytest.raises(TypeError, match="pricing_method must be PricingMethod enum"):
            OptionValuation(
                name="INVALID",
                underlying=ud,
                spec=self.call_spec,
                pricing_method="BSM",  # Invalid: string instead of enum
            )

    def test_option_valuation_mc_requires_path_simulation(self):
        """Test that Monte Carlo pricing requires PathSimulation, not UnderlyingData."""
        ud = UnderlyingData(
            initial_value=100.0,
            volatility=0.2,
            pricing_date=self.pricing_date,
            discount_curve=self.csr,
        )

        with pytest.raises(
            TypeError, match="Monte Carlo pricing requires underlying to be a PathSimulation"
        ):
            OptionValuation(
                name="INVALID",
                underlying=ud,
                spec=self.call_spec,
                pricing_method=PricingMethod.MONTE_CARLO,
            )

    def test_option_valuation_american_bsm_not_implemented(self):
        """Test that American option BSM pricing raises NotImplementedError."""
        ud = UnderlyingData(
            initial_value=100.0,
            volatility=0.2,
            pricing_date=self.pricing_date,
            discount_curve=self.csr,
        )

        american_spec = OptionSpec(
            option_type=OptionType.CALL,
            exercise_type=ExerciseType.AMERICAN,
            strike=self.strike,
            maturity=self.maturity,
            currency="USD",
        )

        # Error is raised during initialization, not present_value
        with pytest.raises(NotImplementedError, match="BSM is only applicable to European"):
            OptionValuation(
                name="CALL_AMERICAN_BSM",
                underlying=ud,
                spec=american_spec,
                pricing_method=PricingMethod.BSM_CONTINUOUS,
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

        self.csr = ConstantShortRate("csr", self.rate)
        self.market_data = MarketData(self.pricing_date, self.csr, currency="USD")

        underlying_params = {
            "initial_value": self.spot,
            "volatility": self.volatility,
            "pricing_date": self.pricing_date,
            "discount_curve": self.csr,
        }

        self.ud = UnderlyingData(**underlying_params)

        self.ud_div = UnderlyingData(**{**underlying_params, "dividend_yield": 0.03})

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
            name="CALL_ATM",
            underlying=self.ud,
            spec=call_spec,
            pricing_method=PricingMethod.BSM_CONTINUOUS,
        )

        pv = valuation.present_value()

        # ATM call should have positive value
        assert pv > 0
        # ATM call value should be approx 10.45 for these parameters
        assert np.isclose(pv, 10.45, rtol=0.01)

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
            name="PUT_ATM",
            underlying=self.ud,
            spec=put_spec,
            pricing_method=PricingMethod.BSM_CONTINUOUS,
        )

        pv = valuation.present_value()

        # ATM put should have positive value
        assert pv > 0

        assert np.isclose(pv, 5.57, rtol=0.01)

    def test_bsm_call_itm(self):
        """Test BSM call option in-the-money."""
        self.ud.initial_value = 110.0  # ITM

        call_spec = OptionSpec(
            option_type=OptionType.CALL,
            exercise_type=ExerciseType.EUROPEAN,
            strike=self.strike,
            maturity=self.maturity,
            currency="USD",
        )

        valuation = OptionValuation(
            name="CALL_ITM",
            underlying=self.ud,
            spec=call_spec,
            pricing_method=PricingMethod.BSM_CONTINUOUS,
        )

        pv = valuation.present_value()

        # ITM call should be worth at least intrinsic value (discounted)
        intrinsic = (self.ud.initial_value - self.strike) * np.exp(-self.rate * 1.0)
        assert pv >= intrinsic * 0.95  # allow small tolerance for numerical issues

    def test_bsm_put_otm(self):
        """Test BSM put option out-of-the-money."""
        self.ud.initial_value = 110.0  # OTM for put

        put_spec = OptionSpec(
            option_type=OptionType.PUT,
            exercise_type=ExerciseType.EUROPEAN,
            strike=self.strike,
            maturity=self.maturity,
            currency="USD",
        )

        valuation = OptionValuation(
            name="PUT_OTM",
            underlying=self.ud,
            spec=put_spec,
            pricing_method=PricingMethod.BSM_CONTINUOUS,
        )

        pv = valuation.present_value()

        # OTM put should have some value due to time value
        assert pv > 0
        # For OTM put, intrinsic is 0 (spot > strike), so time value should be small
        # Put value for OTM should be less than the case when ITM
        assert pv < 5.0

    def test_bsm_with_dividend_yield(self):
        """Test BSM pricing with dividend yield."""
        call_spec = OptionSpec(
            option_type=OptionType.CALL,
            exercise_type=ExerciseType.EUROPEAN,
            strike=self.strike,
            maturity=self.maturity,
            currency="USD",
        )

        valuation = OptionValuation(
            name="ZERO_DIVIDEND",
            underlying=self.ud,
            spec=call_spec,
            pricing_method=PricingMethod.BSM_CONTINUOUS,
        )

        valuation_div = OptionValuation(
            name="POSITIVE_DIVIDEND",
            underlying=self.ud_div,
            spec=call_spec,
            pricing_method=PricingMethod.BSM_CONTINUOUS,
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
            name="CALL",
            underlying=self.ud,
            spec=call_spec,
            pricing_method=PricingMethod.BSM_CONTINUOUS,
        )

        # Create new underlying for put to avoid state pollution
        ud_put = UnderlyingData(
            initial_value=self.spot,
            volatility=self.volatility,
            pricing_date=self.pricing_date,
            discount_curve=self.csr,
        )

        put_val = OptionValuation(
            name="PUT",
            underlying=ud_put,
            spec=put_spec,
            pricing_method=PricingMethod.BSM_CONTINUOUS,
        )

        call_price = call_val.present_value()
        put_price = put_val.present_value()

        # Put-call parity with no dividend
        T = 1.0
        parity_rhs = self.spot * np.exp(-0.0 * T) - self.strike * np.exp(-self.rate * T)

        assert np.isclose(call_price - put_price, parity_rhs, rtol=1e-10)

    def test_bsm_full_return(self):
        """Test BSM with full=True returns tuple."""
        call_spec = OptionSpec(
            option_type=OptionType.CALL,
            exercise_type=ExerciseType.EUROPEAN,
            strike=self.strike,
            maturity=self.maturity,
            currency="USD",
        )

        valuation = OptionValuation(
            name="CALL",
            underlying=self.ud,
            spec=call_spec,
            pricing_method=PricingMethod.BSM_CONTINUOUS,
        )

        result = valuation.present_value(full=True)
        assert isinstance(result, tuple)
        assert len(result) == 2
        # Both values should be the same for BSM
        assert result[0] == result[1]


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

        self.csr = ConstantShortRate("csr", self.rate)

        self.ud = UnderlyingData(
            initial_value=self.spot,
            volatility=self.volatility,
            pricing_date=self.pricing_date,
            discount_curve=self.csr,
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
        )

        pv = valuation.present_value(num_steps=500)
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

        ud_eu = UnderlyingData(
            initial_value=self.spot,
            volatility=self.volatility,
            pricing_date=self.pricing_date,
            discount_curve=self.csr,
        )

        ud_am = UnderlyingData(
            initial_value=self.spot,
            volatility=self.volatility,
            pricing_date=self.pricing_date,
            discount_curve=self.csr,
        )

        eu_val = OptionValuation(
            name="CALL_EU",
            underlying=ud_eu,
            spec=eu_spec,
            pricing_method=PricingMethod.BINOMIAL,
        )

        am_val = OptionValuation(
            name="CALL_AM",
            underlying=ud_am,
            spec=am_spec,
            pricing_method=PricingMethod.BINOMIAL,
        )

        eu_price = eu_val.present_value(num_steps=500)
        am_price = am_val.present_value(num_steps=500)

        # American should be >= European (early exercise premium)
        assert np.isclose(am_price, eu_price, rtol=0.005)

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

        ud_eu = UnderlyingData(
            initial_value=self.spot,
            volatility=self.volatility,
            pricing_date=self.pricing_date,
            discount_curve=self.csr,
        )

        ud_am = UnderlyingData(
            initial_value=self.spot,
            volatility=self.volatility,
            pricing_date=self.pricing_date,
            discount_curve=self.csr,
        )

        eu_val = OptionValuation(
            name="PUT_EU",
            underlying=ud_eu,
            spec=eu_spec,
            pricing_method=PricingMethod.BINOMIAL,
        )

        am_val = OptionValuation(
            name="PUT_AM",
            underlying=ud_am,
            spec=am_spec,
            pricing_method=PricingMethod.BINOMIAL,
        )

        eu_price = eu_val.present_value(num_steps=100)
        am_price = am_val.present_value(num_steps=100)

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

        ud1 = UnderlyingData(
            initial_value=self.spot,
            volatility=self.volatility,
            pricing_date=self.pricing_date,
            discount_curve=self.csr,
        )

        ud2 = UnderlyingData(
            initial_value=self.spot,
            volatility=self.volatility,
            pricing_date=self.pricing_date,
            discount_curve=self.csr,
        )

        val1 = OptionValuation(
            name="CALL_100",
            underlying=ud1,
            spec=call_spec,
            pricing_method=PricingMethod.BINOMIAL,
        )

        val2 = OptionValuation(
            name="CALL_200",
            underlying=ud2,
            spec=call_spec,
            pricing_method=PricingMethod.BINOMIAL,
        )

        price_100 = val1.present_value(num_steps=100)
        price_200 = val2.present_value(num_steps=200)

        # Prices should be closer with more steps (convergence)
        # This is a weak test due to oscillation in binomial, but direction should be reasonable
        assert abs(price_200 - price_100) < 1.0


class TestMCSValuation:
    """Tests for Monte Carlo Simulation valuation."""

    def setup_method(self):
        """Set up market environment for MCS tests."""
        self.pricing_date = dt.datetime(2025, 1, 1)
        self.maturity = dt.datetime(2026, 1, 1)
        self.strike = 100.0
        self.spot = 100.0
        self.volatility = 0.2
        self.rate = 0.05

        self.csr = ConstantShortRate("csr", self.rate)
        self.market_data = MarketData(self.pricing_date, self.csr, currency="USD")

    def test_mcs_european_call_atm(self):
        """Test MCS European call option pricing."""
        gbm_params = GBMParams(initial_value=self.spot, volatility=self.volatility)
        sim_config = SimulationConfig(
            paths=10000,
            frequency="D",
            final_date=self.maturity,
        )

        gbm = GeometricBrownianMotion(
            "gbm_call",
            self.market_data,
            gbm_params,
            sim_config,
        )

        call_spec = OptionSpec(
            option_type=OptionType.CALL,
            exercise_type=ExerciseType.EUROPEAN,
            strike=self.strike,
            maturity=self.maturity,
            currency="USD",
        )

        valuation = OptionValuation(
            name="CALL_MCS",
            underlying=gbm,
            spec=call_spec,
            pricing_method=PricingMethod.MONTE_CARLO,
        )

        pv = valuation.present_value(random_seed=42)

        # Should be positive and in reasonable range
        assert pv > 0
        assert np.isclose(pv, 10.45, rtol=0.02)  # 2% tolerance due to MC error

    def test_mcs_european_put_atm(self):
        """Test MCS European put option pricing."""
        gbm_params = GBMParams(initial_value=self.spot, volatility=self.volatility)
        sim_config = SimulationConfig(
            paths=10000,
            frequency="D",
            final_date=self.maturity,
        )

        gbm = GeometricBrownianMotion(
            "gbm_put",
            self.market_data,
            gbm_params,
            sim_config,
        )

        put_spec = OptionSpec(
            option_type=OptionType.PUT,
            exercise_type=ExerciseType.EUROPEAN,
            strike=self.strike,
            maturity=self.maturity,
            currency="USD",
        )

        valuation = OptionValuation(
            name="PUT_MCS",
            underlying=gbm,
            spec=put_spec,
            pricing_method=PricingMethod.MONTE_CARLO,
        )

        pv = valuation.present_value(random_seed=42)

        # Should be positive
        assert pv > 0

    def test_mcs_reproducibility_with_seed(self):
        """Test that MCS with same seed produces identical results."""
        gbm_params = GBMParams(initial_value=self.spot, volatility=self.volatility)
        sim_config = SimulationConfig(
            paths=1000,  # Smaller for faster test
            frequency="D",
            final_date=self.maturity,
        )

        call_spec = OptionSpec(
            option_type=OptionType.CALL,
            exercise_type=ExerciseType.EUROPEAN,
            strike=self.strike,
            maturity=self.maturity,
            currency="USD",
        )

        # First valuation
        gbm1 = GeometricBrownianMotion(
            "gbm1",
            self.market_data,
            gbm_params,
            sim_config,
        )

        val1 = OptionValuation(
            name="CALL1",
            underlying=gbm1,
            spec=call_spec,
            pricing_method=PricingMethod.MONTE_CARLO,
        )

        pv1 = val1.present_value(random_seed=123)

        # Second valuation with same seed
        gbm2 = GeometricBrownianMotion(
            "gbm2",
            self.market_data,
            gbm_params,
            sim_config,
        )

        val2 = OptionValuation(
            name="CALL2",
            underlying=gbm2,
            spec=call_spec,
            pricing_method=PricingMethod.MONTE_CARLO,
        )

        pv2 = val2.present_value(random_seed=123)

        # Should be identical
        assert np.isclose(pv1, pv2)

    def test_mcs_american_option(self):
        """Test MCS American option using Longstaff-Schwartz."""
        gbm_params = GBMParams(initial_value=self.spot, volatility=self.volatility)
        sim_config = SimulationConfig(
            paths=5000,
            frequency="D",
            final_date=self.maturity,
        )

        gbm = GeometricBrownianMotion(
            "gbm_am",
            self.market_data,
            gbm_params,
            sim_config,
        )

        am_spec = OptionSpec(
            option_type=OptionType.PUT,
            exercise_type=ExerciseType.AMERICAN,
            strike=self.strike,
            maturity=self.maturity,
            currency="USD",
        )

        valuation = OptionValuation(
            name="PUT_AMERICAN_MCS",
            underlying=gbm,
            spec=am_spec,
            pricing_method=PricingMethod.MONTE_CARLO,
        )

        pv1 = valuation.present_value(random_seed=42, deg=2)
        pv2 = valuation.present_value(random_seed=42, deg=5)

        # Should have positive value
        assert pv1 > 0 and pv2 > 0
        # Values with different polynomial degrees should be close
        assert np.isclose(pv2, pv1, rtol=0.02)  # 2% tolerance

        # binomial should also be close
        ud_bin = UnderlyingData(
            initial_value=self.spot,
            volatility=self.volatility,
            pricing_date=self.pricing_date,
            discount_curve=self.csr,
        )

        binom_valuation = OptionValuation(
            name="PUT_AMERICAN_BINOMIAL",
            underlying=ud_bin,
            spec=am_spec,
            pricing_method=PricingMethod.BINOMIAL,
        )

        pv_binom = binom_valuation.present_value(num_steps=500)

        assert np.isclose(pv1, pv_binom, rtol=0.02)

    def test_mcs_full_return(self):
        """Test MCS with full=True returns tuple of (pv, pathwise_pvs)."""
        gbm_params = GBMParams(initial_value=self.spot, volatility=self.volatility)
        sim_config = SimulationConfig(
            paths=1000,
            frequency="D",
            final_date=self.maturity,
        )

        gbm = GeometricBrownianMotion(
            "gbm_full",
            self.market_data,
            gbm_params,
            sim_config,
        )

        call_spec = OptionSpec(
            option_type=OptionType.CALL,
            exercise_type=ExerciseType.EUROPEAN,
            strike=self.strike,
            maturity=self.maturity,
            currency="USD",
        )

        valuation = OptionValuation(
            name="CALL_FULL",
            underlying=gbm,
            spec=call_spec,
            pricing_method=PricingMethod.MONTE_CARLO,
        )

        pv, pv_pathwise = valuation.present_value(full=True, random_seed=42)

        assert isinstance(pv, (float, np.floating))
        assert isinstance(pv_pathwise, np.ndarray)
        assert pv_pathwise.shape[0] == sim_config.paths
        # Mean of pathwise should equal pv
        assert np.isclose(np.mean(pv_pathwise), pv)


class TestGreeks:
    """Tests for greek calculations (delta, vega)."""

    def setup_method(self):
        """Set up market environment for greek tests."""
        self.pricing_date = dt.datetime(2025, 1, 1)
        self.maturity = dt.datetime(2026, 1, 1)
        self.strike = 100.0
        self.spot = 100.0
        self.volatility = 0.2
        self.rate = 0.05

        self.csr = ConstantShortRate("csr", self.rate)

        self.ud = UnderlyingData(
            initial_value=self.spot,
            volatility=self.volatility,
            pricing_date=self.pricing_date,
            discount_curve=self.csr,
        )

    def test_call_delta_positive(self):
        """Test that call option delta is positive."""
        call_spec = OptionSpec(
            option_type=OptionType.CALL,
            exercise_type=ExerciseType.EUROPEAN,
            strike=self.strike,
            maturity=self.maturity,
            currency="USD",
        )

        valuation = OptionValuation(
            name="CALL",
            underlying=self.ud,
            spec=call_spec,
            pricing_method=PricingMethod.BSM_CONTINUOUS,
        )

        delta = valuation.delta()
        assert 0 < delta < 1

    def test_put_delta_negative(self):
        """Test that put option delta is negative."""
        ud = UnderlyingData(
            initial_value=self.spot,
            volatility=self.volatility,
            pricing_date=self.pricing_date,
            discount_curve=self.csr,
        )

        put_spec = OptionSpec(
            option_type=OptionType.PUT,
            exercise_type=ExerciseType.EUROPEAN,
            strike=self.strike,
            maturity=self.maturity,
            currency="USD",
        )

        valuation = OptionValuation(
            name="PUT",
            underlying=ud,
            spec=put_spec,
            pricing_method=PricingMethod.BSM_CONTINUOUS,
        )

        delta = valuation.delta()
        assert -1 < delta < 0

    def test_vega_positive(self):
        """Test that vega is positive (option value increases with volatility)."""
        call_spec = OptionSpec(
            option_type=OptionType.CALL,
            exercise_type=ExerciseType.EUROPEAN,
            strike=self.strike,
            maturity=self.maturity,
            currency="USD",
        )

        valuation = OptionValuation(
            name="CALL",
            underlying=self.ud,
            spec=call_spec,
            pricing_method=PricingMethod.BSM_CONTINUOUS,
        )

        vega = valuation.vega()
        # Vega should be positive (more volatility = higher option value)
        assert vega > 0

    def test_delta_custom_epsilon(self):
        """Test delta calculation with custom epsilon."""
        call_spec = OptionSpec(
            option_type=OptionType.CALL,
            exercise_type=ExerciseType.EUROPEAN,
            strike=self.strike,
            maturity=self.maturity,
            currency="USD",
        )

        valuation = OptionValuation(
            name="CALL",
            underlying=self.ud,
            spec=call_spec,
            pricing_method=PricingMethod.BSM_CONTINUOUS,
        )

        delta = valuation.delta(epsilon=0.5)
        assert 0 < delta < 1

    def test_vega_custom_epsilon(self):
        """Test vega calculation with custom epsilon."""
        call_spec = OptionSpec(
            option_type=OptionType.CALL,
            exercise_type=ExerciseType.EUROPEAN,
            strike=self.strike,
            maturity=self.maturity,
            currency="USD",
        )

        valuation = OptionValuation(
            name="CALL",
            underlying=self.ud,
            spec=call_spec,
            pricing_method=PricingMethod.BSM_CONTINUOUS,
        )

        vega = valuation.vega(epsilon=0.02)
        assert vega > 0
