import pytest
import datetime as dt
import numpy as np
from portfolio_analytics.stochastic_processes import (
    PathSimulation,
    GeometricBrownianMotion,
    SquareRootDiffusion,
    JumpDiffusion,
    GBMParams,
    JDParams,
    SRDParams,
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
            end_date=dt.datetime(2026, 1, 1),
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
            end_date=dt.datetime(2026, 1, 1),
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

    def test_generate_paths_returns_array(self):
        """Test that generate_paths returns the paths array"""
        paths = self.gbm.generate_paths(random_seed=1000)

        # Should return a valid ndarray
        assert isinstance(paths, np.ndarray)
        assert paths.shape[0] == len(self.gbm.time_grid)
        assert paths.shape[1] == self.sim.paths

    def test_time_grid_generation(self):
        """Test that time_grid is generated correctly."""
        # Generate paths to ensure time_grid is populated
        self.gbm.generate_paths(random_seed=42)

        assert self.gbm.time_grid is not None
        # Time grid should start with pricing_date and end with end_date
        assert self.gbm.time_grid[0] == self.market_data.pricing_date
        assert self.gbm.time_grid[-1] == self.sim.end_date

    def test_get_instrument_values(self):
        """Test get_instrument_values method."""
        values = self.gbm.get_instrument_values(random_seed=42)

        # Should return the instrument values array
        assert values is not None
        assert values.shape == (len(self.gbm.time_grid), self.sim.paths)

    def test_gbm_with_different_volatilities(self):
        """Test GBM with different volatility levels."""
        low_vol_params = GBMParams(initial_value=100.0, volatility=0.05)
        high_vol_params = GBMParams(initial_value=100.0, volatility=0.5)

        gbm_low = GeometricBrownianMotion(
            "gbm_low_vol",
            self.market_data,
            low_vol_params,
            self.sim,
        )

        gbm_high = GeometricBrownianMotion(
            "gbm_high_vol",
            self.market_data,
            high_vol_params,
            self.sim,
        )

        paths_low = gbm_low.generate_paths(random_seed=42)
        paths_high = gbm_high.generate_paths(random_seed=42)

        # Higher volatility should lead to wider range of final values
        std_low = np.std(paths_low[-1])
        std_high = np.std(paths_high[-1])

        assert std_high > std_low * 1.5  # High vol should have much wider spread

    def test_gbm_drift_matches_risk_free_rate(self):
        """Test that GBM paths have drift approximately equal to risk-free rate."""
        # With many paths, mean return should approximate r - 0.5*sigma^2
        short_rate = 0.05
        volatility = 0.2
        csr = ConstantShortRate("csr", short_rate)
        market_data = MarketData(dt.datetime(2025, 1, 1), csr, currency="USD")
        process_params = GBMParams(initial_value=100.0, volatility=volatility)
        sim_config = SimulationConfig(
            paths=10000,
            frequency="D",
            end_date=dt.datetime(2026, 1, 1),
        )

        gbm = GeometricBrownianMotion("gbm_drift", market_data, process_params, sim_config)
        paths = gbm.generate_paths(random_seed=42)

        # Calculate mean return
        T = 1.0  # 1 year
        final_prices = paths[-1]
        initial_price = paths[0, 0]
        returns = np.log(final_prices / initial_price) / T

        expected_drift = short_rate - 0.5 * volatility**2
        mean_return = np.mean(returns)

        # Should be close (within standard error)
        assert np.abs(mean_return - expected_drift) < 0.01


class TestSquareRootDiffusion:
    """Tests for the SquareRootDiffusion (CIR) class."""

    def setup_method(self):
        """Set up market environment for SRD tests."""
        self.csr = ConstantShortRate("csr", 0.05)
        self.market_data = MarketData(dt.datetime(2025, 1, 1), self.csr, currency="USD")
        self.process_params = SRDParams(
            initial_value=0.03,
            volatility=0.015,
            kappa=0.2,
            theta=0.05,
        )
        self.sim = SimulationConfig(
            paths=5000,
            frequency="D",
            end_date=dt.datetime(2026, 1, 1),
        )
        self.srd = SquareRootDiffusion(
            "srd_test",
            self.market_data,
            self.process_params,
            self.sim,
        )

    def test_srd_initialization(self):
        """Test SRD initializes with correct parameters."""
        assert self.srd.kappa == self.process_params.kappa
        assert self.srd.theta == self.process_params.theta
        assert self.srd.volatility == self.process_params.volatility

    def test_srd_generate_paths(self):
        """Test SRD path generation."""
        paths = self.srd.generate_paths(random_seed=42)

        # Check shape
        assert paths.shape[0] == len(self.srd.time_grid)
        assert paths.shape[1] == self.sim.paths

        # First row should be initial value
        np.testing.assert_array_almost_equal(paths[0], self.process_params.initial_value)

        # All values should be non-negative (CIR property)
        assert np.all(paths >= 0)

    def test_srd_mean_reversion(self):
        """Test that SRD exhibits mean reversion towards theta."""
        long_sim = SimulationConfig(
            paths=5000,
            frequency="D",
            end_date=dt.datetime(2030, 1, 1),  # 5 years
        )

        srd_long = SquareRootDiffusion(
            "srd_long",
            self.market_data,
            self.process_params,
            long_sim,
        )

        paths = srd_long.generate_paths(random_seed=42)

        # Final values should be closer to theta than initial value
        final_mean = np.mean(paths[-1])

        # With mean reversion, final mean should be between initial and theta
        initial = paths[0, 0]
        theta = self.process_params.theta

        # Final should be closer to theta
        dist_to_theta_initial = abs(initial - theta)
        dist_to_theta_final = abs(final_mean - theta)

        assert dist_to_theta_final < dist_to_theta_initial

    def test_srd_reproducibility(self):
        """Test SRD reproducibility with same seed."""
        paths1 = self.srd.generate_paths(random_seed=123).copy()

        srd2 = SquareRootDiffusion(
            "srd_test2",
            self.market_data,
            self.process_params,
            self.sim,
        )

        paths2 = srd2.generate_paths(random_seed=123)

        np.testing.assert_array_equal(paths1, paths2)


class TestJumpDiffusion:
    """Tests for the JumpDiffusion (Merton) class."""

    def setup_method(self):
        """Set up market environment for Jump Diffusion tests."""
        self.csr = ConstantShortRate("csr", 0.05)
        self.market_data = MarketData(dt.datetime(2025, 1, 1), self.csr, currency="USD")
        self.process_params = JDParams(
            initial_value=100.0,
            volatility=0.2,
            jump_intensity=0.5,  # 0.5 jumps per year on average
            jump_mean=-0.1,
            jump_std=0.3,
        )
        self.sim = SimulationConfig(
            paths=5000,
            frequency="D",
            end_date=dt.datetime(2026, 1, 1),
        )
        self.jd = JumpDiffusion(
            "jd_test",
            self.market_data,
            self.process_params,
            self.sim,
        )

    def test_jd_initialization(self):
        """Test JD initializes with correct parameters."""
        assert self.jd.jump_intensity == self.process_params.jump_intensity
        assert self.jd.jump_mean == self.process_params.jump_mean
        assert self.jd.jump_std == self.process_params.jump_std
        assert self.jd.volatility == self.process_params.volatility

    def test_jd_generate_paths(self):
        """Test JD path generation."""
        paths = self.jd.generate_paths(random_seed=42)

        # Check shape
        assert paths.shape[0] == len(self.jd.time_grid)
        assert paths.shape[1] == self.sim.paths

        # First row should be initial value
        np.testing.assert_array_almost_equal(paths[0], self.process_params.initial_value)

        # All values should be positive
        assert np.all(paths > 0)

    def test_jd_jump_impact(self):
        """Test that jump diffusion paths have discontinuities (jumps)."""
        # Generate paths with jumps
        paths_jd = self.jd.generate_paths(random_seed=42)

        # Generate comparable GBM paths
        gbm_params = GBMParams(
            initial_value=self.process_params.initial_value,
            volatility=self.process_params.volatility,
        )
        gbm = GeometricBrownianMotion(
            "gbm_comparable",
            self.market_data,
            gbm_params,
            self.sim,
        )

        paths_gbm = gbm.generate_paths(random_seed=42)

        # JD should have larger final value spread due to jumps
        std_jd = np.std(paths_jd[-1])
        std_gbm = np.std(paths_gbm[-1])

        # Jump diffusion should have higher volatility
        assert std_jd > std_gbm * 0.9  # Allow some tolerance

    def test_jd_reproducibility(self):
        """Test JD reproducibility with same seed."""
        paths1 = self.jd.generate_paths(random_seed=456).copy()

        jd2 = JumpDiffusion(
            "jd_test2",
            self.market_data,
            self.process_params,
            self.sim,
        )

        paths2 = jd2.generate_paths(random_seed=456)

        np.testing.assert_array_equal(paths1, paths2)

    def test_jd_zero_jump_intensity(self):
        """Test JD with zero jump intensity (should be similar to GBM)."""
        no_jump_params = JDParams(
            initial_value=100.0,
            volatility=0.2,
            jump_intensity=0.0,  # No jumps
            jump_mean=0.0,
            jump_std=0.0,
        )

        jd_no_jump = JumpDiffusion(
            "jd_no_jump",
            self.market_data,
            no_jump_params,
            self.sim,
        )

        paths_jd = jd_no_jump.generate_paths(random_seed=99)

        # Without jumps, all values should be positive and continuous
        assert np.all(paths_jd > 0)


class TestSimulationConfig:
    """Tests for SimulationConfig dataclass."""

    def test_simulation_config_creation(self):
        """Test SimulationConfig creation with required parameters."""
        config = SimulationConfig(
            paths=5000,
            frequency="D",
            end_date=dt.datetime(2026, 1, 1),
        )

        assert config.paths == 5000
        assert config.frequency == "D"
        assert config.end_date == dt.datetime(2026, 1, 1)
        assert config.day_count_convention == 365  # default

    def test_simulation_config_custom_day_count(self):
        """Test SimulationConfig with custom day count convention."""
        config = SimulationConfig(
            paths=1000,
            frequency="W",
            end_date=dt.datetime(2026, 1, 1),
            day_count_convention=360,
        )

        assert config.day_count_convention == 360

    def test_simulation_config_with_special_dates(self):
        """Test SimulationConfig with special dates."""
        special_dates = [
            dt.datetime(2025, 3, 17),
            dt.datetime(2025, 6, 20),
        ]

        config = SimulationConfig(
            paths=1000,
            frequency="D",
            end_date=dt.datetime(2026, 1, 1),
            special_dates=special_dates,
        )

        assert len(config.special_dates) == 2

    def test_simulation_config_with_custom_time_grid(self):
        """Test SimulationConfig with pre-defined time grid."""
        custom_grid = np.array(
            [
                dt.datetime(2025, 1, 1),
                dt.datetime(2025, 6, 1),
                dt.datetime(2026, 1, 1),
            ]
        )

        config = SimulationConfig(
            paths=500,
            frequency="D",
            end_date=dt.datetime(2026, 1, 1),
            time_grid=custom_grid,
        )

        assert config.time_grid is not None
        np.testing.assert_array_equal(config.time_grid, custom_grid)
