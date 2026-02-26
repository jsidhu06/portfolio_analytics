"""Tests for DiscountCurve construction, interpolation, and derived rates."""

import numpy as np
import pytest

from portfolio_analytics.exceptions import ValidationError
from portfolio_analytics.rates import DiscountCurve


# ---------------------------------------------------------------------------
# Construction / Validation
# ---------------------------------------------------------------------------


class TestDiscountCurveConstruction:
    """Validate DiscountCurve constructor guards."""

    def test_flat_curve_basic(self):
        curve = DiscountCurve.flat(rate=0.05, end_time=1.0)
        assert curve.flat_rate == 0.05
        assert float(curve.df(0.0)) == pytest.approx(1.0)
        assert float(curve.df(1.0)) == pytest.approx(np.exp(-0.05))

    def test_flat_curve_multiple_steps(self):
        curve = DiscountCurve.flat(rate=0.05, end_time=2.0, steps=10)
        assert curve.times.size == 11
        assert float(curve.df(1.0)) == pytest.approx(np.exp(-0.05))

    def test_flat_curve_zero_rate(self):
        curve = DiscountCurve.flat(rate=0.0, end_time=1.0)
        assert float(curve.df(0.5)) == pytest.approx(1.0)

    def test_flat_curve_negative_end_time_raises(self):
        with pytest.raises(ValidationError, match="end_time must be positive"):
            DiscountCurve.flat(rate=0.05, end_time=-1.0)

    def test_flat_curve_zero_end_time_raises(self):
        with pytest.raises(ValidationError, match="end_time must be positive"):
            DiscountCurve.flat(rate=0.05, end_time=0.0)

    def test_flat_curve_zero_steps_raises(self):
        with pytest.raises(ValidationError, match="steps must be >= 1"):
            DiscountCurve.flat(rate=0.05, end_time=1.0, steps=0)

    def test_non_increasing_times_raises(self):
        with pytest.raises(ValidationError, match="strictly increasing"):
            DiscountCurve(times=np.array([0.0, 0.5, 0.5]), dfs=np.array([1.0, 0.98, 0.96]))

    def test_negative_df_raises(self):
        with pytest.raises(ValidationError, match="discount factors must be positive"):
            DiscountCurve(times=np.array([0.0, 1.0]), dfs=np.array([1.0, -0.5]))

    def test_df_greater_than_one_warns(self):
        """Discount factors > 1 (negative rates) are allowed but warn."""
        with pytest.warns(match="Discount factors > 1 detected"):
            DiscountCurve(times=np.array([0.0, 1.0]), dfs=np.array([1.0, 1.5]))

    def test_mismatched_arrays_raises(self):
        with pytest.raises(ValidationError, match="same length"):
            DiscountCurve(times=np.array([0.0, 1.0]), dfs=np.array([1.0]))

    def test_inconsistent_flat_rate_raises(self):
        """flat_rate provided but dfs don't match exp(-r*t)."""
        with pytest.raises(ValidationError, match="flat_rate is only allowed"):
            DiscountCurve(
                times=np.array([0.0, 1.0]),
                dfs=np.array([1.0, 0.90]),
                flat_rate=0.05,
            )

    def test_consistent_flat_rate_accepted(self):
        """flat_rate consistent with dfs should be accepted."""
        t = np.array([0.0, 1.0])
        rate = 0.03
        dfs = np.exp(-rate * t)
        curve = DiscountCurve(times=t, dfs=dfs, flat_rate=rate)
        assert curve.flat_rate == rate

    def test_frozen_dataclass(self):
        curve = DiscountCurve.flat(rate=0.05, end_time=1.0)
        with pytest.raises(AttributeError):
            curve.flat_rate = 0.10  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Interpolation (df method)
# ---------------------------------------------------------------------------


class TestDiscountCurveDf:
    """Test df() interpolation behavior."""

    @pytest.fixture()
    def curve(self) -> DiscountCurve:
        """Non-flat curve for interpolation testing."""
        times = np.array([0.0, 0.5, 1.0, 2.0])
        dfs = np.array([1.0, 0.975, 0.95, 0.90])
        return DiscountCurve(times=times, dfs=dfs)

    def test_exact_node(self, curve: DiscountCurve):
        assert float(curve.df(0.5)) == pytest.approx(0.975)

    def test_interpolated_value(self, curve: DiscountCurve):
        """Mid-point interpolation in log-space."""
        df_interp = float(curve.df(0.25))
        # Log-linear: log(df) at 0.25 should be midpoint of log(1.0) and log(0.975)
        expected = np.exp(0.5 * (np.log(1.0) + np.log(0.975)))
        assert df_interp == pytest.approx(expected, rel=1e-10)

    def test_vectorized_input(self, curve: DiscountCurve):
        """df() should accept arrays and return arrays."""
        t = np.array([0.0, 0.5, 1.0])
        result = curve.df(t)
        assert result.shape == (3,)
        assert float(result[0]) == pytest.approx(1.0)
        assert float(result[1]) == pytest.approx(0.975)
        assert float(result[2]) == pytest.approx(0.95)

    def test_extrapolation_flat(self, curve: DiscountCurve):
        """Extrapolation beyond the curve should use flat log-df."""
        df_beyond = float(curve.df(3.0))
        df_end = float(curve.df(2.0))
        # np.interp uses right=log_df[-1] for extrapolation, so df stays at the last value
        assert df_beyond == pytest.approx(df_end)

    def test_scalar_input(self, curve: DiscountCurve):
        """df() called with a scalar should return a scalar-shaped array."""
        result = curve.df(0.5)
        assert result.ndim == 0  # 0-d array


# ---------------------------------------------------------------------------
# Forward rates
# ---------------------------------------------------------------------------


class TestDiscountCurveForwardRate:
    """Test forward_rate() and step_forward_rates()."""

    def test_flat_curve_forward_rate_equals_flat_rate(self):
        curve = DiscountCurve.flat(rate=0.05, end_time=2.0, steps=4)
        fwd = curve.forward_rate(0.25, 0.75)
        assert fwd == pytest.approx(0.05, rel=1e-10)

    def test_forward_rate_t1_le_t0_raises(self):
        curve = DiscountCurve.flat(rate=0.05, end_time=1.0)
        with pytest.raises(ValidationError, match="Need t1 > t0"):
            curve.forward_rate(0.5, 0.5)

    def test_forward_rate_consistency_with_dfs(self):
        """f(t0,t1) should satisfy df(t1) = df(t0) * exp(-f*(t1-t0))."""
        times = np.array([0.0, 0.5, 1.0, 2.0])
        dfs = np.array([1.0, 0.975, 0.95, 0.90])
        curve = DiscountCurve(times=times, dfs=dfs)

        t0, t1 = 0.25, 1.5
        fwd = curve.forward_rate(t0, t1)
        df_t0 = float(curve.df(t0))
        df_t1 = float(curve.df(t1))
        # Verify: df(t1) = df(t0) * exp(-f * (t1-t0))
        reconstructed = df_t0 * np.exp(-fwd * (t1 - t0))
        assert reconstructed == pytest.approx(df_t1, rel=1e-10)

    def test_step_forward_rates_flat(self):
        curve = DiscountCurve.flat(rate=0.05, end_time=2.0, steps=4)
        grid = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
        fwds = curve.step_forward_rates(grid)
        assert fwds.shape == (4,)
        np.testing.assert_allclose(fwds, 0.05, rtol=1e-10)

    def test_step_forward_rates_non_increasing_raises(self):
        curve = DiscountCurve.flat(rate=0.05, end_time=1.0)
        with pytest.raises(ValidationError, match="strictly increasing"):
            curve.step_forward_rates(np.array([0.0, 0.5, 0.5]))

    def test_step_forward_rates_non_flat(self):
        """Forward rates from a non-flat curve should vary."""
        times = np.array([0.0, 0.5, 1.0, 2.0])
        # Different forward rates in each segment
        dfs = np.array(
            [
                1.0,
                np.exp(-0.03 * 0.5),
                np.exp(-0.03 * 0.5 - 0.05 * 0.5),
                np.exp(-0.03 * 0.5 - 0.05 * 0.5 - 0.04 * 1.0),
            ]
        )
        curve = DiscountCurve(times=times, dfs=dfs)
        fwds = curve.step_forward_rates(times)
        np.testing.assert_allclose(fwds, [0.03, 0.05, 0.04], rtol=1e-10)
