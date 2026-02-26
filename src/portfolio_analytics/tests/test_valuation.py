"""Comprehensive tests for valuation dispatcher, specs, and integration."""

import pytest
import datetime as dt
import numpy as np
from portfolio_analytics.exceptions import (
    ConfigurationError,
    UnsupportedFeatureError,
    ValidationError,
)
from portfolio_analytics.valuation import (
    OptionSpec,
    PayoffSpec,
    UnderlyingPricingData,
    OptionValuation,
    BinomialParams,
    MonteCarloParams,
    PDEParams,
)
from portfolio_analytics.strategies import CondorSpec
from portfolio_analytics.valuation.bsm import _BSMEuropeanValuation
from portfolio_analytics.valuation.binomial import (
    _BinomialEuropeanValuation,
)
from portfolio_analytics.valuation.monte_carlo import _MCEuropeanValuation
from portfolio_analytics.enums import (
    DayCountConvention,
    OptionType,
    ExerciseType,
    PricingMethod,
    PositionSide,
)
from portfolio_analytics.stochastic_processes import (
    GBMProcess,
    GBMParams,
    SimulationConfig,
)
from portfolio_analytics.market_environment import MarketData
from portfolio_analytics.rates import DiscountCurve
from portfolio_analytics.valuation.pde import _FDAmericanValuation
from portfolio_analytics.tests.helpers import flat_curve


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
        """Test that None strike is rejected for vanilla OptionSpec."""
        with pytest.raises(ValidationError, match="OptionSpec\\.strike must be provided"):
            OptionSpec(
                option_type=OptionType.CALL,
                exercise_type=ExerciseType.EUROPEAN,
                strike=None,
                maturity=dt.datetime(2026, 12, 31),
                currency="EUR",
            )

    def test_option_spec_invalid_option_type(self):
        """Test that invalid option_type raises TypeError."""
        with pytest.raises(ConfigurationError, match="option_type must be OptionType enum"):
            OptionSpec(
                option_type="CALL",  # Invalid: string instead of enum
                exercise_type=ExerciseType.EUROPEAN,
                strike=100.0,
                maturity=dt.datetime(2026, 12, 31),
                currency="USD",
            )

    def test_option_spec_invalid_exercise_type(self):
        """Test that invalid exercise_type raises TypeError."""
        with pytest.raises(ConfigurationError, match="exercise_type must be ExerciseType enum"):
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


class TestCondorSpec:
    def test_valid_condor_spec_creation(self):
        spec = CondorSpec(
            exercise_type=ExerciseType.EUROPEAN,
            strikes=(50.0, 90.0, 110.0, 150.0),
            maturity=dt.datetime(2026, 12, 31),
            currency="USD",
            side=PositionSide.LONG,
            contract_size=100,
        )
        assert spec.exercise_type == ExerciseType.EUROPEAN
        assert spec.strikes == (50.0, 90.0, 110.0, 150.0)

        legs = spec.leg_definitions()
        assert legs == [
            (OptionType.PUT, 50.0, -1.0),
            (OptionType.PUT, 90.0, +1.0),
            (OptionType.CALL, 110.0, +1.0),
            (OptionType.CALL, 150.0, -1.0),
        ]

    def test_payoff_matches_sum_of_leg_payoffs(self):
        spec = CondorSpec(
            exercise_type=ExerciseType.EUROPEAN,
            strikes=(50.0, 90.0, 110.0, 150.0),
            maturity=dt.datetime(2026, 12, 31),
            currency="USD",
            side=PositionSide.LONG,
        )

        spots = np.array([0.0, 40.0, 70.0, 100.0, 130.0, 200.0])
        k1, k2, k3, k4 = spec.strikes
        leg_payoff = (
            -np.maximum(k1 - spots, 0.0)
            + np.maximum(k2 - spots, 0.0)
            + np.maximum(spots - k3, 0.0)
            - np.maximum(spots - k4, 0.0)
        )
        assert np.allclose(spec.terminal_payoff(spots), leg_payoff)

    def test_invalid_strike_order_raises(self):
        with pytest.raises(ValidationError, match="K1 < K2"):
            CondorSpec(
                exercise_type=ExerciseType.EUROPEAN,
                strikes=(90.0, 50.0, 110.0, 150.0),
                maturity=dt.datetime(2026, 12, 31),
                currency="USD",
            )


class TestUnderlyingPricingData:
    """Tests for UnderlyingPricingData class."""

    def setup_method(self):
        """Set up market environment for tests."""
        self.pricing_date = dt.datetime(2025, 1, 1)
        self.maturity = dt.datetime(2026, 1, 1)
        self.curve = flat_curve(self.pricing_date, self.maturity, 0.05)
        self.market_data = MarketData(self.pricing_date, self.curve, currency="USD")

    def test_underlying_data_creation(self):
        """Test successful creation of UnderlyingPricingData."""
        ud = UnderlyingPricingData(
            initial_value=100.0,
            volatility=0.2,
            market_data=self.market_data,
        )
        assert ud.initial_value == 100.0
        assert ud.volatility == 0.2
        assert ud.pricing_date == self.pricing_date

    def test_underlying_data_is_frozen(self):
        """Test that UnderlyingPricingData is frozen (immutable)."""
        ud = UnderlyingPricingData(
            initial_value=100.0,
            volatility=0.2,
            market_data=self.market_data,
        )
        with pytest.raises(AttributeError):
            ud.initial_value = 105.0


class TestOptionValuation:
    """Tests for OptionValuation dispatcher class."""

    def setup_method(self):
        """Set up market environment and valuation specifications for tests."""
        self.pricing_date = dt.datetime(2025, 1, 1)
        self.maturity = dt.datetime(2026, 1, 1)
        self.strike = 100.0
        self.curve = flat_curve(self.pricing_date, self.maturity, 0.05)
        self.market_data = MarketData(self.pricing_date, self.curve, currency="USD")

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
        """Test OptionValuation creation with UnderlyingPricingData and BSM pricing."""
        ud = UnderlyingPricingData(
            initial_value=100.0,
            volatility=0.2,
            market_data=self.market_data,
        )

        valuation = OptionValuation(
            underlying=ud,
            spec=self.call_spec,
            pricing_method=PricingMethod.BSM,
        )
        assert valuation.strike == self.strike
        assert isinstance(valuation._impl, _BSMEuropeanValuation)

    def test_option_valuation_with_underlying_data_binomial(self):
        """Test OptionValuation creation with UnderlyingPricingData and Binomial pricing."""
        ud = UnderlyingPricingData(
            initial_value=100.0,
            volatility=0.2,
            market_data=self.market_data,
        )

        valuation = OptionValuation(
            underlying=ud,
            spec=self.call_spec,
            pricing_method=PricingMethod.BINOMIAL,
        )
        assert isinstance(valuation._impl, _BinomialEuropeanValuation)

    def test_bsm_accepts_nonflat_discount_curve(self):
        """BSM should accept time-varying discount curves."""
        nonflat_curve = DiscountCurve(
            times=np.array([0.0, 0.5, 1.0]),
            dfs=np.array([1.0, np.exp(-0.03 * 0.5), np.exp(-0.06 * 1.0)]),
        )
        market_data = MarketData(self.pricing_date, nonflat_curve, currency="USD")
        ud = UnderlyingPricingData(
            initial_value=100.0,
            volatility=0.2,
            market_data=market_data,
        )

        valuation = OptionValuation(
            underlying=ud,
            spec=self.call_spec,
            pricing_method=PricingMethod.BSM,
        )
        assert valuation.present_value() > 0.0

    def test_option_valuation_with_path_simulation_mcs(self):
        """Test OptionValuation creation with PathSimulation and MC pricing."""
        gbm_params = GBMParams(initial_value=100.0, volatility=0.2)
        sim_config = SimulationConfig(
            paths=1000,
            frequency="D",
            end_date=self.maturity,
        )
        gbm = GBMProcess(
            self.market_data,
            gbm_params,
            sim_config,
        )

        valuation = OptionValuation(
            underlying=gbm,
            spec=self.call_spec,
            pricing_method=PricingMethod.MONTE_CARLO,
        )
        assert isinstance(valuation._impl, _MCEuropeanValuation)

    def test_mcs_discrete_dividends_reduce_call_price(self):
        """Discrete dividends should reduce European call price under MC."""
        ex_date = self.pricing_date + dt.timedelta(days=180)
        time_grid = np.array([self.pricing_date, ex_date, self.maturity])

        gbm_params_no_div = GBMParams(initial_value=100.0, volatility=0.2)
        gbm_params_div = GBMParams(
            initial_value=100.0,
            volatility=0.2,
            discrete_dividends=[(ex_date, 1.0)],
        )

        sim_config = SimulationConfig(
            paths=20000,
            day_count_convention=DayCountConvention.ACT_365F,
            time_grid=time_grid,
        )

        gbm_no_div = GBMProcess(
            self.market_data,
            gbm_params_no_div,
            sim_config,
        )
        gbm_div = GBMProcess(
            self.market_data,
            gbm_params_div,
            sim_config,
        )

        val_no_div = OptionValuation(
            underlying=gbm_no_div,
            spec=self.call_spec,
            pricing_method=PricingMethod.MONTE_CARLO,
            params=MonteCarloParams(random_seed=42),
        )
        val_div = OptionValuation(
            underlying=gbm_div,
            spec=self.call_spec,
            pricing_method=PricingMethod.MONTE_CARLO,
            params=MonteCarloParams(random_seed=42),
        )

        pv_no_div = val_no_div.present_value()
        pv_div = val_div.present_value()

        assert pv_div < pv_no_div

    def test_option_valuation_invalid_maturity(self):
        """Test that OptionValuation raises error if maturity <= pricing_date."""
        ud = UnderlyingPricingData(
            initial_value=100.0,
            volatility=0.2,
            market_data=self.market_data,
        )

        invalid_spec = OptionSpec(
            option_type=OptionType.CALL,
            exercise_type=ExerciseType.EUROPEAN,
            strike=self.strike,
            maturity=self.pricing_date,  # Invalid: maturity == pricing_date
            currency="USD",
        )

        with pytest.raises(ValidationError, match="Option maturity must be after pricing_date"):
            OptionValuation(
                underlying=ud,
                spec=invalid_spec,
                pricing_method=PricingMethod.BSM,
            )

    def test_option_valuation_currency_mismatch(self):
        """Test that OptionValuation raises for cross-currency inputs."""
        ud = UnderlyingPricingData(
            initial_value=100.0,
            volatility=0.2,
            market_data=self.market_data,
        )

        eur_spec = OptionSpec(
            option_type=OptionType.CALL,
            exercise_type=ExerciseType.EUROPEAN,
            strike=self.strike,
            maturity=self.maturity,
            currency="EUR",
        )

        with pytest.raises(
            UnsupportedFeatureError, match="Cross-currency valuation is not supported"
        ):
            OptionValuation(
                underlying=ud,
                spec=eur_spec,
                pricing_method=PricingMethod.BSM,
            )

    def test_binomial_condor_equals_sum_of_legs(self):
        """A European condor payoff priced directly equals sum of vanilla legs (binomial)."""
        ud = UnderlyingPricingData(
            initial_value=90.0,
            volatility=0.2,
            market_data=self.market_data,
            dividend_curve=None,
        )

        strikes = (50.0, 90.0, 110.0, 150.0)
        condor_spec = CondorSpec(
            exercise_type=ExerciseType.EUROPEAN,
            strikes=strikes,
            maturity=self.maturity,
            currency="USD",
            side=PositionSide.LONG,
        )

        payoff_spec = PayoffSpec(
            exercise_type=ExerciseType.EUROPEAN,
            maturity=self.maturity,
            currency="USD",
            payoff_fn=condor_spec.terminal_payoff,
        )
        payoff_pv = OptionValuation(
            underlying=ud,
            spec=payoff_spec,
            pricing_method=PricingMethod.BINOMIAL,
            params=BinomialParams(num_steps=2000),
        ).present_value()

        leg_pv = 0.0
        for opt_type, k, w in condor_spec.leg_definitions():
            leg_spec = OptionSpec(
                option_type=opt_type,
                exercise_type=ExerciseType.EUROPEAN,
                strike=k,
                maturity=self.maturity,
                currency="USD",
            )
            leg_val = OptionValuation(
                underlying=ud,
                spec=leg_spec,
                pricing_method=PricingMethod.BINOMIAL,
                params=BinomialParams(num_steps=2000),
            )
            leg_pv += w * leg_val.present_value()

        assert np.isclose(payoff_pv, leg_pv, rtol=1e-3, atol=0)

    def test_mcs_condor_equals_sum_of_legs(self):
        """A European condor payoff priced directly equals sum of vanilla legs (Monte Carlo)."""

        simulation_config = SimulationConfig(
            paths=200_000,
            frequency="W",
            day_count_convention=DayCountConvention.ACT_365F,
            end_date=self.maturity,
        )
        process_params = GBMParams(initial_value=90, volatility=0.2)
        gbm = GBMProcess(self.market_data, process_params, simulation_config)

        strikes = (50.0, 90.0, 110.0, 150.0)
        condor_spec = CondorSpec(
            exercise_type=ExerciseType.EUROPEAN,
            strikes=strikes,
            maturity=self.maturity,
            currency="USD",
            side=PositionSide.LONG,
        )

        payoff_spec = PayoffSpec(
            exercise_type=ExerciseType.EUROPEAN,
            maturity=self.maturity,
            currency="USD",
            payoff_fn=condor_spec.terminal_payoff,
        )
        payoff_pv = OptionValuation(
            underlying=gbm,
            spec=payoff_spec,
            pricing_method=PricingMethod.MONTE_CARLO,
            params=MonteCarloParams(random_seed=42),
        ).present_value()

        leg_pv = 0.0
        for opt_type, k, w in condor_spec.leg_definitions():
            leg_spec = OptionSpec(
                option_type=opt_type,
                exercise_type=ExerciseType.EUROPEAN,
                strike=k,
                maturity=self.maturity,
                currency="USD",
            )
            leg_val = OptionValuation(
                underlying=gbm,
                spec=leg_spec,
                pricing_method=PricingMethod.MONTE_CARLO,
                params=MonteCarloParams(random_seed=42),
            )
            leg_pv += w * leg_val.present_value()

        assert np.isclose(payoff_pv, leg_pv, rtol=0.02, atol=0)

    def test_binomial_mcs_condor_equivalence(self):
        """Condor PV via binomial should approx equal PV via MCS under same params."""
        initial_value, volatility = 90, 0.2

        ud = UnderlyingPricingData(
            initial_value=initial_value,
            volatility=volatility,
            market_data=self.market_data,
            dividend_curve=None,
        )
        simulation_config = SimulationConfig(
            paths=200_000,
            frequency="W",
            day_count_convention=DayCountConvention.ACT_365F,
            end_date=self.maturity,
        )
        process_params = GBMParams(initial_value=initial_value, volatility=volatility)
        gbm = GBMProcess(self.market_data, process_params, simulation_config)

        strikes = (50.0, 90.0, 110.0, 150.0)
        condor_spec = CondorSpec(
            exercise_type=ExerciseType.EUROPEAN,
            strikes=strikes,
            maturity=self.maturity,
            currency="USD",
            side=PositionSide.LONG,
        )

        binomial_pv = 0.0
        for opt_type, k, w in condor_spec.leg_definitions():
            leg_spec = OptionSpec(
                option_type=opt_type,
                exercise_type=ExerciseType.EUROPEAN,
                strike=k,
                maturity=self.maturity,
                currency="USD",
            )
            binomial_pv += (
                w
                * OptionValuation(
                    underlying=ud,
                    spec=leg_spec,
                    pricing_method=PricingMethod.BINOMIAL,
                    params=BinomialParams(num_steps=2500),
                ).present_value()
            )

        mcs_pv = 0.0
        for opt_type, k, w in condor_spec.leg_definitions():
            leg_spec = OptionSpec(
                option_type=opt_type,
                exercise_type=ExerciseType.EUROPEAN,
                strike=k,
                maturity=self.maturity,
                currency="USD",
            )
            mcs_pv += (
                w
                * OptionValuation(
                    underlying=gbm,
                    spec=leg_spec,
                    pricing_method=PricingMethod.MONTE_CARLO,
                    params=MonteCarloParams(random_seed=42),
                ).present_value()
            )

        assert np.isclose(binomial_pv, mcs_pv, rtol=0.01)

    def test_custom_payoff_single_contract_american_supported(self):
        """Custom payoff should be priceable as a single American contract.

        This specifically checks that early exercise decisions compare intrinsic vs continuation
        for the *whole payoff*, not decomposed legs.
        """

        def capped_payoff(spot: np.ndarray | float) -> np.ndarray:
            s = np.asarray(spot, dtype=float)
            return np.minimum(40.0, np.maximum(90.0 - s, 0.0) + np.maximum(s - 110.0, 0.0))

        spec_am = PayoffSpec(
            exercise_type=ExerciseType.AMERICAN,
            maturity=self.maturity,
            currency="USD",
            payoff_fn=capped_payoff,
        )
        spec_eu = PayoffSpec(
            exercise_type=ExerciseType.EUROPEAN,
            maturity=self.maturity,
            currency="USD",
            payoff_fn=capped_payoff,
        )

        # Binomial (UnderlyingPricingData)
        ud = UnderlyingPricingData(
            initial_value=100.0,
            volatility=0.2,
            market_data=self.market_data,
            dividend_curve=None,
        )
        pv_binom_am = OptionValuation(
            underlying=ud,
            spec=spec_am,
            pricing_method=PricingMethod.BINOMIAL,
            params=BinomialParams(num_steps=500),
        ).present_value()
        pv_binom_eu = OptionValuation(
            underlying=ud,
            spec=spec_eu,
            pricing_method=PricingMethod.BINOMIAL,
            params=BinomialParams(num_steps=500),
        ).present_value()

        assert pv_binom_am >= pv_binom_eu
        assert 0.0 <= pv_binom_am <= 40.0 + 1e-8

        # Monte Carlo (PathSimulation) American via LSM
        gbm_params = GBMParams(initial_value=100.0, volatility=0.2)
        sim_config = SimulationConfig(paths=20000, frequency="W", end_date=self.maturity)
        gbm = GBMProcess(self.market_data, gbm_params, sim_config)
        pv_mcs_am = OptionValuation(
            underlying=gbm,
            spec=spec_am,
            pricing_method=PricingMethod.MONTE_CARLO,
            params=MonteCarloParams(random_seed=42, deg=3),
        ).present_value()
        pv_mcs_eu = OptionValuation(
            underlying=gbm,
            spec=spec_eu,
            pricing_method=PricingMethod.MONTE_CARLO,
            params=MonteCarloParams(random_seed=42),
        ).present_value()

        assert pv_mcs_am >= pv_mcs_eu
        assert 0.0 <= pv_mcs_am <= 40.0 + 1e-8

    def test_american_condor_is_sum_of_american_legs_binomial(self):
        """American CondorSpec is valued as an independently exercisable strategy (sum of legs)."""
        ud = UnderlyingPricingData(
            initial_value=90.0,
            volatility=0.2,
            market_data=self.market_data,
            dividend_curve=None,
        )

        strikes = (50.0, 90.0, 110.0, 150.0)
        condor_spec = CondorSpec(
            exercise_type=ExerciseType.AMERICAN,
            strikes=strikes,
            maturity=self.maturity,
            currency="USD",
            side=PositionSide.LONG,
        )

        condor_pv = condor_spec.present_value(
            underlying=ud,
            pricing_method=PricingMethod.BINOMIAL,
            params=BinomialParams(num_steps=500),
        )

        k1, k2, k3, k4 = strikes
        leg_specs = [
            (OptionType.PUT, k1, -1.0),
            (OptionType.PUT, k2, +1.0),
            (OptionType.CALL, k3, +1.0),
            (OptionType.CALL, k4, -1.0),
        ]
        leg_pv = 0.0
        for opt_type, k, w in leg_specs:
            leg_spec = OptionSpec(
                option_type=opt_type,
                exercise_type=ExerciseType.AMERICAN,
                strike=k,
                maturity=self.maturity,
                currency="USD",
            )
            leg_val = OptionValuation(
                underlying=ud,
                spec=leg_spec,
                pricing_method=PricingMethod.BINOMIAL,
                params=BinomialParams(num_steps=500),
            )
            leg_pv += w * leg_val.present_value()

        assert np.isclose(condor_pv, leg_pv, rtol=1e-12, atol=0)

    def test_american_condor_is_sum_of_american_legs_mcs(self):
        """Same semantics under MCS/LSM: CondorSpec AMERICAN aggregates per-leg exercise."""
        simulation_config = SimulationConfig(
            paths=50_000,
            frequency="W",
            day_count_convention=DayCountConvention.ACT_365F,
            end_date=self.maturity,
        )
        process_params = GBMParams(initial_value=90, volatility=0.2)
        gbm = GBMProcess(self.market_data, process_params, simulation_config)

        strikes = (50.0, 90.0, 110.0, 150.0)
        condor_spec = CondorSpec(
            exercise_type=ExerciseType.AMERICAN,
            strikes=strikes,
            maturity=self.maturity,
            currency="USD",
            side=PositionSide.LONG,
        )

        k1, k2, k3, k4 = strikes
        leg_specs = [
            (OptionType.PUT, k1, -1.0),
            (OptionType.PUT, k2, +1.0),
            (OptionType.CALL, k3, +1.0),
            (OptionType.CALL, k4, -1.0),
        ]
        leg_pv = 0.0
        for opt_type, k, w in leg_specs:
            leg_spec = OptionSpec(
                option_type=opt_type,
                exercise_type=ExerciseType.AMERICAN,
                strike=k,
                maturity=self.maturity,
                currency="USD",
            )
            leg_val = OptionValuation(
                underlying=gbm,
                spec=leg_spec,
                pricing_method=PricingMethod.MONTE_CARLO,
                params=MonteCarloParams(random_seed=42, deg=3),
            )
            leg_pv += w * leg_val.present_value()

        condor_pv = condor_spec.present_value(
            underlying=gbm,
            pricing_method=PricingMethod.MONTE_CARLO,
            params=MonteCarloParams(random_seed=42, deg=3),
        )
        assert np.isclose(condor_pv, leg_pv, rtol=1e-12, atol=0)

    def test_option_valuation_invalid_pricing_method_type(self):
        """Test that OptionValuation validates pricing_method is PricingMethod enum."""
        ud = UnderlyingPricingData(
            initial_value=100.0,
            volatility=0.2,
            market_data=self.market_data,
        )

        with pytest.raises(ConfigurationError, match="pricing_method must be PricingMethod enum"):
            OptionValuation(
                underlying=ud,
                spec=self.call_spec,
                pricing_method="BSM",  # Invalid: string instead of enum
            )

    def test_option_valuation_mc_requires_path_simulation(self):
        """Test that Monte Carlo pricing requires PathSimulation, not UnderlyingPricingData."""
        ud = UnderlyingPricingData(
            initial_value=100.0,
            volatility=0.2,
            market_data=self.market_data,
        )

        with pytest.raises(
            ConfigurationError,
            match="Monte Carlo pricing requires underlying to be a PathSimulation",
        ):
            OptionValuation(
                underlying=ud,
                spec=self.call_spec,
                pricing_method=PricingMethod.MONTE_CARLO,
            )

    def test_underlying_pricing_data_warns_mixed_dividends(self):
        """Both dividend_curve and discrete_dividends should emit a warning."""
        with pytest.warns(UserWarning, match="both dividend_curve and discrete_dividends"):
            UnderlyingPricingData(
                initial_value=100.0,
                volatility=0.2,
                market_data=self.market_data,
                dividend_curve=flat_curve(self.pricing_date, self.maturity, 0.02),
                discrete_dividends=[(self.pricing_date + dt.timedelta(days=90), 1.0)],
            )

    def test_option_valuation_binomial_rejects_path_simulation(self):
        """Test that Binomial pricing rejects PathSimulation (should use UnderlyingPricingData)."""
        process = GBMProcess(
            market_data=self.market_data,
            process_params=GBMParams(initial_value=100.0, volatility=0.2),
            sim=SimulationConfig(
                paths=1000,
                day_count_convention=DayCountConvention.ACT_365F,
                time_grid=np.array([self.pricing_date, self.maturity]),
            ),
        )

        with pytest.raises(
            ConfigurationError, match="BINOMIAL pricing does not use stochastic path simulation"
        ):
            OptionValuation(
                underlying=process,
                spec=self.call_spec,
                pricing_method=PricingMethod.BINOMIAL,
            )

    def test_option_valuation_bsm_rejects_path_simulation(self):
        """Test that BSM pricing rejects PathSimulation (should use UnderlyingPricingData)."""
        process = GBMProcess(
            market_data=self.market_data,
            process_params=GBMParams(initial_value=100.0, volatility=0.2),
            sim=SimulationConfig(
                paths=1000,
                day_count_convention=DayCountConvention.ACT_365F,
                time_grid=np.array([self.pricing_date, self.maturity]),
            ),
        )

        with pytest.raises(
            ConfigurationError, match="BSM pricing does not use stochastic path simulation"
        ):
            OptionValuation(
                underlying=process,
                spec=self.call_spec,
                pricing_method=PricingMethod.BSM,
            )

    def test_option_valuation_pde_rejects_path_simulation(self):
        """Test that PDE_FD pricing rejects PathSimulation (should use UnderlyingPricingData)."""
        process = GBMProcess(
            market_data=self.market_data,
            process_params=GBMParams(initial_value=100.0, volatility=0.2),
            sim=SimulationConfig(
                paths=1000,
                day_count_convention=DayCountConvention.ACT_365F,
                time_grid=np.array([self.pricing_date, self.maturity]),
            ),
        )

        with pytest.raises(
            ConfigurationError, match="PDE_FD pricing does not use stochastic path simulation"
        ):
            OptionValuation(
                underlying=process,
                spec=self.call_spec,
                pricing_method=PricingMethod.PDE_FD,
            )

    def test_option_valuation_american_bsm_not_implemented(self):
        """Test that American option BSM pricing raises NotImplementedError."""
        ud = UnderlyingPricingData(
            initial_value=100.0,
            volatility=0.2,
            market_data=self.market_data,
        )

        american_spec = OptionSpec(
            option_type=OptionType.CALL,
            exercise_type=ExerciseType.AMERICAN,
            strike=self.strike,
            maturity=self.maturity,
            currency="USD",
        )

        # Error is raised during initialization, not present_value
        with pytest.raises(UnsupportedFeatureError, match="BSM is only applicable to European"):
            OptionValuation(
                underlying=ud,
                spec=american_spec,
                pricing_method=PricingMethod.BSM,
            )

    def test_dispatcher_creates_fd_impl_for_american(self):
        ud = UnderlyingPricingData(
            initial_value=100.0,
            volatility=0.2,
            market_data=self.market_data,
            dividend_curve=None,
        )
        spec = OptionSpec(
            option_type=OptionType.PUT,
            exercise_type=ExerciseType.AMERICAN,
            strike=self.strike,
            maturity=self.maturity,
            currency="USD",
        )

        valuation = OptionValuation(ud, spec, PricingMethod.PDE_FD)
        assert isinstance(valuation._impl, _FDAmericanValuation)

    def test_american_put_fd_close_to_binomial(self):
        ud = UnderlyingPricingData(
            initial_value=100.0,
            volatility=0.2,
            market_data=self.market_data,
            dividend_curve=None,
        )
        spec = OptionSpec(
            option_type=OptionType.PUT,
            exercise_type=ExerciseType.AMERICAN,
            strike=self.strike,
            maturity=self.maturity,
            currency="USD",
        )

        fd_val = OptionValuation(
            ud,
            spec,
            PricingMethod.PDE_FD,
            params=PDEParams(spot_steps=90, time_steps=90, max_iter=20_000),
        )
        fd_pv = fd_val.present_value()

        tree_val = OptionValuation(
            ud, spec, PricingMethod.BINOMIAL, params=BinomialParams(num_steps=1200)
        )
        tree_pv = tree_val.present_value()

        # Both are numerical approximations; keep tolerance modest for test stability.
        assert np.isclose(fd_pv, tree_pv, rtol=0.01)

    def test_american_call_fd_close_to_binomial(self):
        ud = UnderlyingPricingData(
            initial_value=100.0,
            volatility=0.2,
            market_data=self.market_data,
            dividend_curve=flat_curve(self.pricing_date, self.maturity, 0.03),
        )
        spec = OptionSpec(
            option_type=OptionType.CALL,
            exercise_type=ExerciseType.AMERICAN,
            strike=self.strike,
            maturity=self.maturity,
            currency="USD",
        )

        fd_val = OptionValuation(
            ud,
            spec,
            PricingMethod.PDE_FD,
            params=PDEParams(spot_steps=90, time_steps=90, max_iter=20_000),
        )
        fd_pv = fd_val.present_value()

        tree_val = OptionValuation(
            ud, spec, PricingMethod.BINOMIAL, params=BinomialParams(num_steps=1200)
        )
        tree_pv = tree_val.present_value()

        # Both are numerical approximations; keep tolerance modest for test stability.
        assert np.isclose(fd_pv, tree_pv, rtol=0.01)

    def test_american_call_no_dividend_equals_european_fd(self):
        """American CALL with q=0 has no early exercise premium.

        Regression test for the PDE fast-path that prices this case using the
        European CN solver (skipping PSOR).
        """
        ud = UnderlyingPricingData(
            initial_value=100.0,
            volatility=0.2,
            market_data=self.market_data,
            dividend_curve=None,
        )
        spec_am = OptionSpec(
            option_type=OptionType.CALL,
            exercise_type=ExerciseType.AMERICAN,
            strike=self.strike,
            maturity=self.maturity,
            currency="USD",
        )
        spec_eu = OptionSpec(
            option_type=OptionType.CALL,
            exercise_type=ExerciseType.EUROPEAN,
            strike=self.strike,
            maturity=self.maturity,
            currency="USD",
        )

        params = PDEParams(spot_steps=90, time_steps=90, max_iter=20_000)

        am_val = OptionValuation(ud, spec_am, PricingMethod.PDE_FD, params=params)
        eu_val = OptionValuation(ud, spec_eu, PricingMethod.PDE_FD, params=params)

        am_pv = am_val.present_value()
        eu_pv = eu_val.present_value()

        assert np.isclose(am_pv, eu_pv, rtol=1e-12, atol=1e-12)

    def test_american_call_no_dividend_close_to_bsm_european(self):
        ud = UnderlyingPricingData(
            initial_value=100.0,
            volatility=0.2,
            market_data=self.market_data,
            dividend_curve=None,
        )
        spec_am = OptionSpec(
            option_type=OptionType.CALL,
            exercise_type=ExerciseType.AMERICAN,
            strike=self.strike,
            maturity=self.maturity,
            currency="USD",
        )
        spec_eu = OptionSpec(
            option_type=OptionType.CALL,
            exercise_type=ExerciseType.EUROPEAN,
            strike=self.strike,
            maturity=self.maturity,
            currency="USD",
        )

        fd_val = OptionValuation(
            ud,
            spec_am,
            PricingMethod.PDE_FD,
            params=PDEParams(spot_steps=90, time_steps=90, max_iter=20_000),
        )
        fd_pv = fd_val.present_value()

        bsm_val = OptionValuation(ud, spec_eu, PricingMethod.BSM)
        bsm_pv = bsm_val.present_value()

        # American call with no dividends should be near European (no early exercise benefit).
        assert np.isclose(fd_pv, bsm_pv, atol=1.0)

    def test_binomial_discrete_dividends_close_to_bsm(self):
        """Binomial with discrete dividends should be close to BSM with dividend-adjusted spot."""
        spec = OptionSpec(
            option_type=OptionType.CALL,
            exercise_type=ExerciseType.EUROPEAN,
            strike=self.strike,
            maturity=self.maturity,
            currency="USD",
        )

        discrete_divs = [
            (self.pricing_date + dt.timedelta(days=90), 0.6),
            (self.pricing_date + dt.timedelta(days=180), 0.6),
        ]

        ud = UnderlyingPricingData(
            initial_value=100.0,
            volatility=0.2,
            market_data=self.market_data,
            dividend_curve=None,
            discrete_dividends=discrete_divs,
        )

        binom_val = OptionValuation(
            ud,
            spec,
            PricingMethod.BINOMIAL,
            params=BinomialParams(num_steps=1200),
        )
        bsm_val = OptionValuation(ud, spec, PricingMethod.BSM)

        binom_pv = binom_val.present_value()
        bsm_pv = bsm_val.present_value()

        assert np.isclose(binom_pv, bsm_pv, rtol=0.02)
