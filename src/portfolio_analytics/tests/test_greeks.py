"""Comprehensive tests for greek calculations (delta, gamma, vega)."""

import datetime as dt

import numpy as np
import pytest

from portfolio_analytics.enums import (
    ExerciseType,
    GreekCalculationMethod,
    OptionType,
    PricingMethod,
    DayCountConvention,
)
from portfolio_analytics.market_environment import MarketData
from portfolio_analytics.rates import ConstantShortRate
from portfolio_analytics.stochastic_processes import (
    GBMParams,
    PathSimulation,
    GeometricBrownianMotion,
    SimulationConfig,
)
from portfolio_analytics.valuation import (
    OptionSpec,
    OptionValuation,
    UnderlyingPricingData,
)
from portfolio_analytics.valuation import BinomialParams, MonteCarloParams


class TestGreeksSetup:
    """Base setup for greek tests with common parameters + factory helpers.

    Notes
    -----
    - setup fixture is function-scoped (default), so each test method gets fresh state.
    - factories avoid mutating shared UnderlyingPricingData instances within a test.
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        self.pricing_date = dt.datetime(2025, 1, 1)
        self.maturity = dt.datetime(2026, 1, 1)
        self.spot = 100.0
        self.strike = 100.0
        self.rate = 0.05
        self.volatility = 0.20
        self.currency = "USD"

        # Discount curve
        self.csr = ConstantShortRate("csr", self.rate)

        # Market data + sim config for Monte Carlo
        self.market_data = MarketData(self.pricing_date, self.csr, currency=self.currency)
        self.sim_config = SimulationConfig(
            paths=200_000,
            frequency="W",
            day_count_convention=DayCountConvention.ACT_365F,
            end_date=self.maturity,
        )

    # -------- factories (preferred over mutating shared instances) --------

    def _make_ud(
        self,
        *,
        spot: float | None = None,
        vol: float | None = None,
        dividend_yield: float = 0.0,
        pricing_date: dt.datetime | None = None,
    ) -> UnderlyingPricingData:
        md = (
            self.market_data
            if pricing_date is None
            else MarketData(pricing_date, self.csr, currency=self.currency)
        )
        return UnderlyingPricingData(
            initial_value=self.spot if spot is None else spot,
            volatility=self.volatility if vol is None else vol,
            market_data=md,
            dividend_yield=dividend_yield,
        )

    def _make_spec(
        self,
        *,
        option_type: OptionType,
        exercise_type: ExerciseType = ExerciseType.EUROPEAN,
        strike: float | None = None,
        maturity: dt.datetime | None = None,
        currency: str | None = None,
    ) -> OptionSpec:
        return OptionSpec(
            option_type=option_type,
            exercise_type=exercise_type,
            strike=self.strike if strike is None else strike,
            maturity=self.maturity if maturity is None else maturity,
            currency=self.currency if currency is None else currency,
        )

    def _make_val(
        self,
        name: str,
        underlying: PathSimulation | UnderlyingPricingData,
        spec: OptionSpec,
        method: PricingMethod,
        params=None,
    ) -> OptionValuation:
        return OptionValuation(name, underlying, spec, method, params=params)

    def _make_gbm(
        self,
        *,
        spot: float | None = None,
        vol: float | None = None,
        dividend_yield: float = 0.0,
    ) -> GeometricBrownianMotion:
        params = GBMParams(
            initial_value=self.spot if spot is None else spot,
            volatility=self.volatility if vol is None else vol,
            dividend_yield=dividend_yield,
        )
        return GeometricBrownianMotion("gbm", self.market_data, params, self.sim_config)


class TestDeltaBasicProperties(TestGreeksSetup):
    """Test basic properties of delta values."""

    def test_call_delta_positive(self):
        spec = self._make_spec(option_type=OptionType.CALL)
        ud = self._make_ud()
        valuation = self._make_val("call_atm", ud, spec, PricingMethod.BSM)
        assert valuation.delta() > 0

    def test_put_delta_negative(self):
        spec = self._make_spec(option_type=OptionType.PUT)
        ud = self._make_ud()
        valuation = self._make_val("put_atm", ud, spec, PricingMethod.BSM)
        assert valuation.delta() < 0

    def test_call_delta_atm_greater_than_half(self):
        """With positive r, forward > spot so ATM call delta > 0.5."""
        spec = self._make_spec(option_type=OptionType.CALL, strike=self.strike)
        ud = self._make_ud(spot=self.spot)
        valuation = self._make_val("call_atm", ud, spec, PricingMethod.BSM)
        delta = valuation.delta()
        assert 0.5 < delta < 1.0

    def test_call_delta_itm_close_to_one(self):
        spec = self._make_spec(option_type=OptionType.CALL)
        ud = self._make_ud(spot=150.0)
        valuation = self._make_val("call_itm", ud, spec, PricingMethod.BSM)
        assert valuation.delta() > 0.95

    def test_call_delta_otm_close_to_zero(self):
        spec = self._make_spec(option_type=OptionType.CALL)
        ud = self._make_ud(spot=50.0)
        valuation = self._make_val("call_otm", ud, spec, PricingMethod.BSM)
        assert valuation.delta() < 0.05

    def test_put_delta_itm_close_to_negative_one(self):
        spec = self._make_spec(option_type=OptionType.PUT)
        ud = self._make_ud(spot=50.0)
        valuation = self._make_val("put_itm", ud, spec, PricingMethod.BSM)
        assert valuation.delta() < -0.95

    def test_put_delta_otm_close_to_zero(self):
        spec = self._make_spec(option_type=OptionType.PUT)
        ud = self._make_ud(spot=150.0)
        valuation = self._make_val("put_otm", ud, spec, PricingMethod.BSM)
        assert valuation.delta() > -0.05


class TestGammaBasicProperties(TestGreeksSetup):
    """Test basic properties of gamma values."""

    def test_gamma_always_positive(self):
        call_spec = self._make_spec(option_type=OptionType.CALL)
        put_spec = self._make_spec(option_type=OptionType.PUT)
        ud_call = self._make_ud()
        ud_put = self._make_ud()

        call_val = self._make_val("call", ud_call, call_spec, PricingMethod.BSM)
        put_val = self._make_val("put", ud_put, put_spec, PricingMethod.BSM)

        assert call_val.gamma() > 0
        assert put_val.gamma() > 0

    def test_gamma_highest_atm(self):
        spec = self._make_spec(option_type=OptionType.CALL)

        # ATM gamma
        ud_atm = self._make_ud(spot=self.spot)
        val_atm = self._make_val("call_atm", ud_atm, spec, PricingMethod.BSM)
        gamma_atm = val_atm.gamma()

        # ITM gamma (separate underlying instance)
        ud_itm = self._make_ud(spot=110.0)
        val_itm = self._make_val("call_itm", ud_itm, spec, PricingMethod.BSM)
        gamma_itm = val_itm.gamma()

        assert gamma_atm > gamma_itm

    def test_gamma_decreases_with_time(self):
        maturity_short = dt.datetime(2025, 3, 1)

        ud_short = self._make_ud(pricing_date=self.pricing_date)
        spec_short = self._make_spec(option_type=OptionType.CALL, maturity=maturity_short)
        val_short = self._make_val("call_short", ud_short, spec_short, PricingMethod.BSM)
        gamma_short = val_short.gamma()

        ud_long = self._make_ud(pricing_date=self.pricing_date)
        spec_long = self._make_spec(option_type=OptionType.CALL, maturity=self.maturity)
        val_long = self._make_val("call_long", ud_long, spec_long, PricingMethod.BSM)
        gamma_long = val_long.gamma()

        assert gamma_short > gamma_long


class TestVegaBasicProperties(TestGreeksSetup):
    """Test basic properties of vega values."""

    def test_vega_always_positive(self):
        call_spec = self._make_spec(option_type=OptionType.CALL)
        put_spec = self._make_spec(option_type=OptionType.PUT)
        ud_call = self._make_ud()
        ud_put = self._make_ud()

        call_val = self._make_val("call", ud_call, call_spec, PricingMethod.BSM)
        put_val = self._make_val("put", ud_put, put_spec, PricingMethod.BSM)

        assert call_val.vega() > 0
        assert put_val.vega() > 0

    def test_vega_highest_atm(self):
        spec = self._make_spec(option_type=OptionType.CALL)

        ud_atm = self._make_ud(spot=self.spot)
        val_atm = self._make_val("call_atm", ud_atm, spec, PricingMethod.BSM)
        vega_atm = val_atm.vega()

        ud_itm = self._make_ud(spot=110.0)
        val_itm = self._make_val("call_itm", ud_itm, spec, PricingMethod.BSM)
        vega_itm = val_itm.vega()

        assert vega_atm > vega_itm

    def test_vega_increases_with_time(self):
        maturity_short = dt.datetime(2025, 3, 1)
        spec_short = self._make_spec(option_type=OptionType.CALL, maturity=maturity_short)
        ud_short = self._make_ud()
        val_short = self._make_val("call_short", ud_short, spec_short, PricingMethod.BSM)
        vega_short = val_short.vega()

        spec_long = self._make_spec(option_type=OptionType.CALL, maturity=self.maturity)
        ud_long = self._make_ud()
        val_long = self._make_val("call_long", ud_long, spec_long, PricingMethod.BSM)
        vega_long = val_long.vega()

        assert vega_long > vega_short


class TestGreekCalculationMethods(TestGreeksSetup):
    """Test analytical vs numerical greek calculation methods."""

    def test_bsm_analytical_vs_numerical_delta(self):
        spec = self._make_spec(option_type=OptionType.CALL)
        ud = self._make_ud()
        valuation = self._make_val("call", ud, spec, PricingMethod.BSM)

        delta_analytical = valuation.delta(greek_calc_method=GreekCalculationMethod.ANALYTICAL)
        delta_numerical = valuation.delta(greek_calc_method=GreekCalculationMethod.NUMERICAL)

        assert np.isclose(delta_analytical, delta_numerical, rtol=1e-3)

    def test_bsm_analytical_vs_numerical_gamma(self):
        spec = self._make_spec(option_type=OptionType.CALL)
        ud = self._make_ud()
        valuation = self._make_val("call", ud, spec, PricingMethod.BSM)

        gamma_analytical = valuation.gamma(greek_calc_method=GreekCalculationMethod.ANALYTICAL)
        gamma_numerical = valuation.gamma(greek_calc_method=GreekCalculationMethod.NUMERICAL)

        assert np.isclose(gamma_analytical, gamma_numerical, rtol=1e-3)

    def test_bsm_analytical_vs_numerical_vega(self):
        spec = self._make_spec(option_type=OptionType.CALL)
        ud = self._make_ud()
        valuation = self._make_val("call", ud, spec, PricingMethod.BSM)

        vega_analytical = valuation.vega(greek_calc_method=GreekCalculationMethod.ANALYTICAL)
        vega_numerical = valuation.vega(greek_calc_method=GreekCalculationMethod.NUMERICAL)

        assert np.isclose(vega_analytical, vega_numerical, rtol=1e-3)

    def test_bsm_analytical_vs_numerical_rho(self):
        spec = self._make_spec(option_type=OptionType.CALL)
        ud = self._make_ud()
        valuation = self._make_val("call", ud, spec, PricingMethod.BSM)

        rho_analytical = valuation.rho(greek_calc_method=GreekCalculationMethod.ANALYTICAL)
        rho_numerical = valuation.rho(greek_calc_method=GreekCalculationMethod.NUMERICAL)

        assert np.isclose(rho_analytical, rho_numerical, rtol=1e-3)


class TestGreekConsistencyAcrossPricingMethods(TestGreeksSetup):
    """Test that greeks are consistent across different pricing methods."""

    def test_call_delta_consistency_bsm_binomial(self):
        spec = self._make_spec(option_type=OptionType.CALL)

        bsm_val = self._make_val("call_bsm", self._make_ud(), spec, PricingMethod.BSM)
        delta_bsm = bsm_val.delta()

        binomial_val = self._make_val(
            "call_binomial",
            self._make_ud(),
            spec,
            PricingMethod.BINOMIAL,
            params=BinomialParams(num_steps=2500),
        )
        delta_binomial = binomial_val.delta()

        assert np.isclose(delta_bsm, delta_binomial, atol=0.01)

    def test_call_delta_consistency_bsm_mcs(self):
        spec = self._make_spec(option_type=OptionType.CALL)

        bsm_val = self._make_val("call_bsm", self._make_ud(), spec, PricingMethod.BSM)
        delta_bsm = bsm_val.delta()

        gbm = self._make_gbm(dividend_yield=0.0)
        mcs_val = self._make_val(
            "call_mcs",
            gbm,
            spec,
            PricingMethod.MONTE_CARLO,
            params=MonteCarloParams(random_seed=42),
        )
        delta_mcs = mcs_val.delta()

        assert np.isclose(delta_bsm, delta_mcs, atol=0.03)

    def test_gamma_consistency_bsm_binomial(self):
        spec = self._make_spec(option_type=OptionType.CALL)

        bsm_val = self._make_val("call_bsm", self._make_ud(), spec, PricingMethod.BSM)
        gamma_bsm = bsm_val.gamma()

        binomial_val = self._make_val(
            "call_binomial",
            self._make_ud(),
            spec,
            PricingMethod.BINOMIAL,
            params=BinomialParams(num_steps=2500),
        )
        gamma_binomial = binomial_val.gamma()

        assert np.isclose(gamma_bsm, gamma_binomial, atol=0.005)

    def test_vega_consistency_bsm_binomial(self):
        spec = self._make_spec(option_type=OptionType.CALL)

        bsm_val = self._make_val("call_bsm", self._make_ud(), spec, PricingMethod.BSM)
        vega_bsm = bsm_val.vega()

        binomial_val = self._make_val(
            "call_binomial",
            self._make_ud(),
            spec,
            PricingMethod.BINOMIAL,
            params=BinomialParams(num_steps=2500),
        )
        vega_binomial = binomial_val.vega()

        assert np.isclose(vega_bsm, vega_binomial, atol=0.1)


class TestGreeksDividendYieldEffect(TestGreeksSetup):
    """Test the effect of dividend yield on greeks."""

    def test_call_delta_lower_with_dividend_yield(self):
        spec = self._make_spec(option_type=OptionType.CALL)

        val_no_div = self._make_val(
            "call_no_div", self._make_ud(dividend_yield=0.0), spec, PricingMethod.BSM
        )
        delta_no_div = val_no_div.delta()

        val_with_div = self._make_val(
            "call_with_div", self._make_ud(dividend_yield=0.03), spec, PricingMethod.BSM
        )
        delta_with_div = val_with_div.delta()

        assert delta_with_div < delta_no_div

    def test_put_delta_more_negative_with_dividend_yield(self):
        spec = self._make_spec(option_type=OptionType.PUT)

        val_no_div = self._make_val(
            "put_no_div", self._make_ud(dividend_yield=0.0), spec, PricingMethod.BSM
        )
        delta_no_div = val_no_div.delta()

        val_with_div = self._make_val(
            "put_with_div", self._make_ud(dividend_yield=0.03), spec, PricingMethod.BSM
        )
        delta_with_div = val_with_div.delta()

        assert delta_with_div < delta_no_div


class TestGreekErrorHandling(TestGreeksSetup):
    """Test error handling for greek calculations."""

    def test_analytical_greek_with_non_bsm_raises_error(self):
        spec = self._make_spec(option_type=OptionType.CALL)

        valuation = self._make_val("call_binomial", self._make_ud(), spec, PricingMethod.BINOMIAL)

        with pytest.raises(ValueError, match="Analytical greeks are only available for BSM"):
            valuation.delta(greek_calc_method=GreekCalculationMethod.ANALYTICAL)

    def test_invalid_greek_calc_method_type_raises_error(self):
        spec = self._make_spec(option_type=OptionType.CALL)

        valuation = self._make_val("call_bsm", self._make_ud(), spec, PricingMethod.BSM)

        with pytest.raises(
            TypeError, match="greek_calc_method must be GreekCalculationMethod enum"
        ):
            valuation.delta(greek_calc_method="analytical")  # type: ignore[arg-type]


class TestGreekImmutability(TestGreeksSetup):
    """Test that greek calculations don't mutate underlying state (thread-safety)."""

    def test_delta_does_not_mutate_underlying_pricing_data(self):
        """Verify delta calculation doesn't mutate UnderlyingPricingData.initial_value."""
        ud = self._make_ud()
        spec = self._make_spec(option_type=OptionType.CALL)
        valuation = self._make_val("call_bsm", ud, spec, PricingMethod.BSM)

        original_spot = ud.initial_value
        original_vol = ud.volatility

        # Calculate delta with numerical method
        delta = valuation.delta(greek_calc_method=GreekCalculationMethod.NUMERICAL)

        # Verify underlying state is unchanged
        assert ud.initial_value == original_spot, "Delta calculation mutated initial_value"
        assert ud.volatility == original_vol, "Delta calculation mutated volatility"
        assert delta is not None  # sanity check

    def test_gamma_does_not_mutate_underlying_pricing_data(self):
        """Verify gamma calculation doesn't mutate UnderlyingPricingData.initial_value."""
        ud = self._make_ud()
        spec = self._make_spec(option_type=OptionType.CALL)
        valuation = self._make_val(
            "call_binomial",
            ud,
            spec,
            PricingMethod.BINOMIAL,
            params=BinomialParams(num_steps=100),
        )

        original_spot = ud.initial_value
        original_vol = ud.volatility

        # Calculate gamma
        gamma = valuation.gamma()

        # Verify underlying state is unchanged
        assert ud.initial_value == original_spot, "Gamma calculation mutated initial_value"
        assert ud.volatility == original_vol, "Gamma calculation mutated volatility"
        assert gamma is not None  # sanity check

    def test_vega_does_not_mutate_underlying_pricing_data(self):
        """Verify vega calculation doesn't mutate UnderlyingPricingData.volatility."""
        ud = self._make_ud()
        spec = self._make_spec(option_type=OptionType.CALL)
        valuation = self._make_val("call_bsm", ud, spec, PricingMethod.BSM)

        original_spot = ud.initial_value
        original_vol = ud.volatility

        # Calculate vega with numerical method
        vega = valuation.vega(greek_calc_method=GreekCalculationMethod.NUMERICAL)

        # Verify underlying state is unchanged
        assert ud.initial_value == original_spot, "Vega calculation mutated initial_value"
        assert ud.volatility == original_vol, "Vega calculation mutated volatility"
        assert vega is not None  # sanity check

    def test_delta_does_not_mutate_path_simulation(self):
        """Verify delta calculation doesn't mutate PathSimulation.initial_value."""
        # Create PathSimulation
        process = GeometricBrownianMotion(
            name="STOCK",
            market_data=self.market_data,
            process_params=GBMParams(initial_value=self.spot, volatility=self.volatility),
            sim=SimulationConfig(
                paths=1000,
                day_count_convention=DayCountConvention.ACT_365F,
                time_grid=np.array([self.pricing_date, self.maturity]),
            ),
        )

        spec = self._make_spec(option_type=OptionType.CALL)
        valuation = OptionValuation(
            name="call_mc",
            underlying=process,
            spec=spec,
            pricing_method=PricingMethod.MONTE_CARLO,
            params=MonteCarloParams(random_seed=42),
        )

        original_spot = process.initial_value
        original_vol = process.volatility

        # Calculate delta
        delta = valuation.delta()

        # Verify PathSimulation state is unchanged
        assert (
            process.initial_value == original_spot
        ), "Delta calculation mutated PathSimulation.initial_value"
        assert (
            process.volatility == original_vol
        ), "Delta calculation mutated PathSimulation.volatility"
        assert delta is not None  # sanity check

    def test_multiple_concurrent_greeks_with_shared_underlying(self):
        """Verify multiple valuations can share underlying without interference.

        This simulates thread-safety scenario where multiple portfolios
        might calculate greeks for the same underlying simultaneously.
        """
        # Shared underlying
        ud = self._make_ud()
        original_spot = ud.initial_value
        original_vol = ud.volatility

        # Create multiple valuations with different strikes sharing same underlying
        spec1 = self._make_spec(option_type=OptionType.CALL, strike=90.0)
        spec2 = self._make_spec(option_type=OptionType.CALL, strike=100.0)
        spec3 = self._make_spec(option_type=OptionType.PUT, strike=110.0)

        val1 = self._make_val("call_90", ud, spec1, PricingMethod.BSM)
        val2 = self._make_val("call_100", ud, spec2, PricingMethod.BSM)
        val3 = self._make_val("put_110", ud, spec3, PricingMethod.BSM)

        # Calculate greeks for all valuations (simulating concurrent access)
        delta1 = val1.delta(greek_calc_method=GreekCalculationMethod.NUMERICAL)
        gamma2 = val2.gamma(greek_calc_method=GreekCalculationMethod.NUMERICAL)
        vega3 = val3.vega(greek_calc_method=GreekCalculationMethod.NUMERICAL)

        # Verify underlying state is unchanged
        assert ud.initial_value == original_spot, "Shared underlying was mutated"
        assert ud.volatility == original_vol, "Shared underlying volatility was mutated"

        # Verify results are reasonable
        assert 0 < delta1 < 1.0  # ITM call delta
        assert gamma2 > 0  # ATM gamma is positive
        assert vega3 > 0  # vega always positive
