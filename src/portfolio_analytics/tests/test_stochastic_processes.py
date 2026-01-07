import pytest
import datetime as dt
import numpy as np
from portfolio_analytics.stochastic_processes import (
    PathSimulation,
    GeometricBrownianMotion,
    GBMParams,
    SimulationConfig,
)
from portfolio_analytics.market_environment import MarketData
from portfolio_analytics.rates import ConstantShortRate


class TestPathSimulation:
    """Tests for the abstract PathSimulation class"""

    def test_cannot_instantiate_abstract_class(self):
        """Test that PathSimulation cannot be instantiated directly due to abstract method"""
        csr = ConstantShortRate("csr", 0.05)
        market_data = MarketData(dt.datetime(2025, 1, 1), csr, currency="EUR")
        process_params = GBMParams(initial_value=100.0, volatility=0.2)
        sim = SimulationConfig(
            paths=10000,
            frequency="ME",
            final_date=dt.datetime(2025, 12, 31),
        )

        with pytest.raises(
            TypeError,
            match="Can't instantiate abstract class PathSimulation without an implementation for abstract method 'generate_paths'",
        ):
            PathSimulation("test_name", market_data, process_params, sim, corr=None)


class TestGeometricBrownianMotion:
    """Tests for the GeometricBrownianMotion class"""

    def setup_method(self):
        """Set up market environment for GBM tests"""
        self.csr = ConstantShortRate("csr", 0.05)
        self.market_data = MarketData(dt.datetime(2025, 1, 1), self.csr, currency="EUR")
        self.process_params = GBMParams(initial_value=36.0, volatility=0.2)
        self.sim = SimulationConfig(
            paths=10000,
            frequency="ME",
            final_date=dt.datetime(2025, 12, 31),
        )
        self.gbm = GeometricBrownianMotion(
            "gbm_test",
            self.market_data,
            self.process_params,
            self.sim,
            corr=None,
        )

    def test_generate_paths_reproducibility_with_seed(self):
        """Test that generate_paths produces identical results with the same random_seed"""
        # Generate paths with seed 1000
        paths_1 = self.gbm.generate_paths(random_seed=1000).copy()

        # Create a new instance and generate again with same seed
        gbm_2 = GeometricBrownianMotion(
            "gbm_test_2",
            self.market_data,
            self.process_params,
            self.sim,
            corr=None,
        )
        paths_2 = gbm_2.generate_paths(random_seed=1000).copy()

        # Should be identical
        np.testing.assert_array_equal(paths_1, paths_2)

    def test_generate_paths_uses_correct_random_numbers(self):
        """Test that generate_paths uses random numbers equivalent to sn_random_numbers with same seed"""

        # Generate paths
        paths = self.gbm.generate_paths(random_seed=1000)

        assert np.allclose(
            paths[:5, 0],
            np.array([36.0, 37.37221481, 39.45866146, 42.51433276, 41.76443271]),
        )
        assert np.allclose(
            paths[-5:, 0],
            np.array([40.71879366, 40.56504051, 40.15717404, 42.0974104, 43.33170027]),
        )

    def test_generate_paths_basic_properties(self):
        """Test basic properties of generated paths"""
        paths = self.gbm.generate_paths(random_seed=1000)

        # Check shape: (M, num_paths) where M is time steps, num_paths is paths
        M = len(self.gbm.time_grid)
        num_paths = self.sim.paths
        assert paths.shape == (M, num_paths)

        # First row should be initial value
        np.testing.assert_array_equal(paths[0], self.process_params.initial_value)

        # All values should be positive (GBM property)
        assert np.all(paths > 0)

    def test_instrument_values_set_after_generate_paths(self):
        """Test that instrument_values is set after calling generate_paths"""
        assert self.gbm.instrument_values is None

        paths = self.gbm.generate_paths(random_seed=1000)

        # Both direct return and instance variable should be set
        assert self.gbm.instrument_values is not None
        np.testing.assert_array_equal(paths, self.gbm.instrument_values)
