import pytest
import datetime as dt
import numpy as np
from portfolio_analytics.stochastic_processes import (
    PathSimulation,
    GeometricBrownianMotion,
)
from portfolio_analytics.utils import MarketEnvironment, ConstantShortRate


class TestPathSimulation:
    """Tests for the abstract PathSimulation class"""

    def test_cannot_instantiate_abstract_class(self):
        """Test that PathSimulation cannot be instantiated directly due to abstract method"""
        me = MarketEnvironment("test_me", dt.datetime(2025, 1, 1))
        me.add_constant("initial_value", 100.0)
        me.add_constant("volatility", 0.2)
        me.add_constant("final_date", dt.datetime(2025, 12, 31))
        me.add_constant("currency", "EUR")
        me.add_constant("frequency", "ME")
        me.add_constant("paths", 10000)
        csr = ConstantShortRate("csr", 0.05)
        me.add_curve("discount_curve", csr)

        with pytest.raises(
            TypeError,
            match="Can't instantiate abstract class PathSimulation without an implementation for abstract method 'generate_paths'",
        ):
            PathSimulation("test_name", me, corr=False)


class TestGeometricBrownianMotion:
    """Tests for the GeometricBrownianMotion class"""

    def setup_method(self):
        """Set up market environment for GBM tests"""
        self.me = MarketEnvironment("me_gbm", dt.datetime(2025, 1, 1))
        self.me.add_constant("initial_value", 36.0)
        self.me.add_constant("volatility", 0.2)
        self.me.add_constant("final_date", dt.datetime(2025, 12, 31))
        self.me.add_constant("currency", "EUR")
        self.me.add_constant("frequency", "ME")
        self.me.add_constant("paths", 10000)
        csr = ConstantShortRate("csr", 0.05)
        self.me.add_curve("discount_curve", csr)
        self.gbm = GeometricBrownianMotion("gbm_test", self.me, corr=False)

    def test_generate_paths_reproducibility_with_seed(self):
        """Test that generate_paths produces identical results with the same random_seed"""
        # Generate paths with seed 1000
        self.gbm.generate_paths(random_seed=1000)
        paths_1 = self.gbm.instrument_values.copy()

        # Reset and generate again with same seed
        self.gbm.instrument_values = None
        self.gbm.generate_paths(random_seed=1000)
        paths_2 = self.gbm.instrument_values.copy()

        # Should be identical
        np.testing.assert_array_equal(paths_1, paths_2)

    def test_generate_paths_uses_correct_random_numbers(self):
        """Test that generate_paths uses random numbers equivalent to sn_random_numbers with same seed"""

        # Generate paths
        self.gbm.generate_paths(random_seed=1000)

        assert np.allclose(
            self.gbm.instrument_values[:5, 0],
            np.array([36.0, 37.37221481, 39.45866146, 42.51433276, 41.76443271]),
        )
        assert np.allclose(
            self.gbm.instrument_values[-5:, 0],
            np.array([40.71879366, 40.56504051, 40.15717404, 42.0974104, 43.33170027]),
        )

    def test_generate_paths_basic_properties(self):
        """Test basic properties of generated paths"""
        self.gbm.generate_paths(random_seed=1000)
        paths = self.gbm.instrument_values

        # Check shape: (M, num_paths) where M is time steps, num_paths is paths
        M = len(self.gbm.time_grid)
        num_paths = self.me.get_constant("paths")
        assert paths.shape == (M, num_paths)

        # First row should be initial value
        np.testing.assert_array_equal(paths[0], self.me.get_constant("initial_value"))

        # All values should be positive (GBM property)
        assert np.all(paths > 0)  # check this

    def test_update_method(self):
        """Test the update method modifies parameters correctly"""

        self.gbm.update(volatility=0.3, initial_value=50.0)

        assert self.gbm.volatility == 0.3
        assert self.gbm.initial_value == 50.0
        # instrument_values should be reset to None
        assert self.gbm.instrument_values is None
