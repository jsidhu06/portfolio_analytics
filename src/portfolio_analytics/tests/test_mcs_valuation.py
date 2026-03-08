"""Tests for Monte Carlo Simulation valuation."""

import numpy as np

from portfolio_analytics.enums import ExerciseType, OptionType, PricingMethod
from portfolio_analytics.stochastic_processes import (
    GBMParams,
    GBMProcess,
    SimulationConfig,
)
from portfolio_analytics.tests.conftest import (
    BINOM_PARAMS,
    MC_PARAMS,
    MATURITY,
    SPOT,
    VOL,
)
from portfolio_analytics.tests.helpers import market_data, pv, spec
from portfolio_analytics.valuation import (
    MonteCarloParams,
    OptionValuation,
    UnderlyingData,
)

# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

_MD = market_data()


def _gbm(
    paths: int = 10_000,
    frequency: str = "D",
    spot: float = SPOT,
    vol: float = VOL,
) -> GBMProcess:
    params = GBMParams(initial_value=spot, volatility=vol)
    sim_config = SimulationConfig(paths=paths, frequency=frequency, end_date=MATURITY)
    return GBMProcess(_MD, params, sim_config)


class TestMCSValuation:
    """Tests for Monte Carlo Simulation valuation."""

    def test_mcs_european_call_atm(self):
        """Test MCS European call option pricing."""
        result = pv(_gbm(), spec(), PricingMethod.MONTE_CARLO, params=MC_PARAMS)
        assert result > 0
        assert np.isclose(result, 10.45, rtol=0.02)

    def test_mcs_european_put_atm(self):
        """Test MCS European put option pricing."""
        result = pv(_gbm(), spec(OptionType.PUT), PricingMethod.MONTE_CARLO, params=MC_PARAMS)
        assert result > 0

    def test_mcs_reproducibility_with_seed(self):
        """Test that MCS with same seed produces identical results."""
        sp = spec()
        mc_seed = MonteCarloParams(random_seed=123)
        pv1 = pv(_gbm(paths=1000), sp, PricingMethod.MONTE_CARLO, params=mc_seed)
        pv2 = pv(_gbm(paths=1000), sp, PricingMethod.MONTE_CARLO, params=mc_seed)
        assert np.isclose(pv1, pv2)

    def test_mcs_american_option(self):
        """Test MCS American option using Longstaff-Schwartz."""
        gbm = _gbm(paths=5000)
        am_spec = spec(OptionType.PUT, exercise=ExerciseType.AMERICAN)

        pv_deg2 = pv(
            gbm, am_spec, PricingMethod.MONTE_CARLO, params=MonteCarloParams(random_seed=42, deg=2)
        )
        pv_deg5 = pv(
            gbm, am_spec, PricingMethod.MONTE_CARLO, params=MonteCarloParams(random_seed=42, deg=5)
        )

        assert pv_deg2 > 0 and pv_deg5 > 0
        assert np.isclose(pv_deg5, pv_deg2, rtol=0.02)

        # binomial should also be close
        ud_bin = UnderlyingData(initial_value=SPOT, volatility=VOL, market_data=_MD)
        pv_binom = pv(ud_bin, am_spec, PricingMethod.BINOMIAL, params=BINOM_PARAMS)
        assert np.isclose(pv_deg2, pv_binom, rtol=0.02)

    def test_mcs_pathwise_return(self):
        """Test MCS present_value_pathwise returns discounted PVs per path."""
        gbm = _gbm(paths=1000)
        valuation = OptionValuation(gbm, spec(), PricingMethod.MONTE_CARLO, params=MC_PARAMS)

        pv_scalar = valuation.present_value()
        pv_pathwise = valuation.present_value_pathwise()

        assert isinstance(pv_scalar, (float, np.floating))
        assert isinstance(pv_pathwise, np.ndarray)
        assert pv_pathwise.shape[0] == 1000
        assert np.isclose(np.mean(pv_pathwise), pv_scalar)
