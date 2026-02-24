"""Tests for Monte Carlo Simulation valuation."""

import datetime as dt

import numpy as np

from portfolio_analytics.enums import ExerciseType, OptionType, PricingMethod
from portfolio_analytics.market_environment import MarketData
from portfolio_analytics.stochastic_processes import (
    GBMParams,
    GBMProcess,
    SimulationConfig,
)
from portfolio_analytics.tests.helpers import flat_curve
from portfolio_analytics.valuation import (
    BinomialParams,
    MonteCarloParams,
    OptionSpec,
    OptionValuation,
    UnderlyingPricingData,
)


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

        self.curve = flat_curve(self.pricing_date, self.maturity, self.rate, name="csr")
        self.market_data = MarketData(self.pricing_date, self.curve, currency="USD")

    def test_mcs_european_call_atm(self):
        """Test MCS European call option pricing."""
        gbm_params = GBMParams(initial_value=self.spot, volatility=self.volatility)
        sim_config = SimulationConfig(
            paths=10000,
            frequency="D",
            end_date=self.maturity,
        )

        gbm = GBMProcess(
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
            params=MonteCarloParams(random_seed=42),
        )

        pv = valuation.present_value()

        # Should be positive and in reasonable range
        assert pv > 0
        assert np.isclose(pv, 10.45, rtol=0.02)  # 2% tolerance due to MC error

    def test_mcs_european_put_atm(self):
        """Test MCS European put option pricing."""
        gbm_params = GBMParams(initial_value=self.spot, volatility=self.volatility)
        sim_config = SimulationConfig(
            paths=10000,
            frequency="D",
            end_date=self.maturity,
        )

        gbm = GBMProcess(
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
            params=MonteCarloParams(random_seed=42),
        )

        pv = valuation.present_value()

        # Should be positive
        assert pv > 0

    def test_mcs_reproducibility_with_seed(self):
        """Test that MCS with same seed produces identical results."""
        gbm_params = GBMParams(initial_value=self.spot, volatility=self.volatility)
        sim_config = SimulationConfig(
            paths=1000,  # Smaller for faster test
            frequency="D",
            end_date=self.maturity,
        )

        call_spec = OptionSpec(
            option_type=OptionType.CALL,
            exercise_type=ExerciseType.EUROPEAN,
            strike=self.strike,
            maturity=self.maturity,
            currency="USD",
        )

        # First valuation
        gbm1 = GBMProcess(
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
            params=MonteCarloParams(random_seed=123),
        )

        pv1 = val1.present_value()

        # Second valuation with same seed
        gbm2 = GBMProcess(
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
            params=MonteCarloParams(random_seed=123),
        )

        pv2 = val2.present_value()

        # Should be identical
        assert np.isclose(pv1, pv2)

    def test_mcs_american_option(self):
        """Test MCS American option using Longstaff-Schwartz."""
        gbm_params = GBMParams(initial_value=self.spot, volatility=self.volatility)
        sim_config = SimulationConfig(
            paths=5000,
            frequency="D",
            end_date=self.maturity,
        )

        gbm = GBMProcess(
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

        valuation_deg2 = OptionValuation(
            name="PUT_AMERICAN_MCS_DEG2",
            underlying=gbm,
            spec=am_spec,
            pricing_method=PricingMethod.MONTE_CARLO,
            params=MonteCarloParams(random_seed=42, deg=2),
        )
        valuation_deg5 = OptionValuation(
            name="PUT_AMERICAN_MCS_DEG5",
            underlying=gbm,
            spec=am_spec,
            pricing_method=PricingMethod.MONTE_CARLO,
            params=MonteCarloParams(random_seed=42, deg=5),
        )

        pv1 = valuation_deg2.present_value()
        pv2 = valuation_deg5.present_value()

        # Should have positive value
        assert pv1 > 0 and pv2 > 0
        # Values with different polynomial degrees should be close
        assert np.isclose(pv2, pv1, rtol=0.02)  # 2% tolerance

        # binomial should also be close
        ud_bin = UnderlyingPricingData(
            initial_value=self.spot,
            volatility=self.volatility,
            market_data=self.market_data,
        )

        binom_valuation = OptionValuation(
            name="PUT_AMERICAN_BINOMIAL",
            underlying=ud_bin,
            spec=am_spec,
            pricing_method=PricingMethod.BINOMIAL,
            params=BinomialParams(num_steps=500),
        )

        pv_binom = binom_valuation.present_value()

        assert np.isclose(pv1, pv_binom, rtol=0.02)

    def test_mcs_pathwise_return(self):
        """Test MCS present_value_pathwise returns discounted PVs per path."""
        gbm_params = GBMParams(initial_value=self.spot, volatility=self.volatility)
        sim_config = SimulationConfig(
            paths=1000,
            frequency="D",
            end_date=self.maturity,
        )

        gbm = GBMProcess(
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
            params=MonteCarloParams(random_seed=42),
        )

        pv = valuation.present_value()
        pv_pathwise = valuation.present_value_pathwise()

        assert isinstance(pv, (float, np.floating))
        assert isinstance(pv_pathwise, np.ndarray)
        assert pv_pathwise.shape[0] == sim_config.paths
        # Mean of pathwise should equal pv
        assert np.isclose(np.mean(pv_pathwise), pv)
