"""Tests for MarketData and CorrelationContext validation."""

import datetime as dt

import numpy as np
import pytest

from portfolio_analytics.enums import DayCountConvention
from portfolio_analytics.exceptions import ValidationError
from portfolio_analytics.market_environment import CorrelationContext, MarketData
from portfolio_analytics.rates import DiscountCurve


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PRICING_DATE = dt.datetime(2025, 1, 1)
CURVE = DiscountCurve.flat(0.05, end_time=1.0)


# ---------------------------------------------------------------------------
# MarketData
# ---------------------------------------------------------------------------


class TestMarketDataValidation:
    def test_valid_construction(self):
        md = MarketData(PRICING_DATE, CURVE, currency="USD")
        assert md.pricing_date == PRICING_DATE
        assert md.currency == "USD"
        assert md.day_count_convention is DayCountConvention.ACT_365F

    def test_custom_day_count(self):
        md = MarketData(
            PRICING_DATE, CURVE, currency="USD", day_count_convention=DayCountConvention.ACT_360
        )
        assert md.day_count_convention is DayCountConvention.ACT_360

    def test_rejects_bad_pricing_date(self):
        with pytest.raises(ValidationError, match="pricing_date must be a datetime"):
            MarketData("2025-01-01", CURVE, currency="USD")

    def test_rejects_bad_discount_curve(self):
        with pytest.raises(ValidationError, match="discount_curve must be a DiscountCurve"):
            MarketData(PRICING_DATE, 0.05, currency="USD")

    def test_rejects_empty_currency(self):
        with pytest.raises(ValidationError, match="currency must be a non-empty string"):
            MarketData(PRICING_DATE, CURVE, currency="")

    def test_rejects_non_string_currency(self):
        with pytest.raises(ValidationError, match="currency must be a non-empty string"):
            MarketData(PRICING_DATE, CURVE, currency=123)

    def test_rejects_bad_day_count_convention(self):
        with pytest.raises(ValidationError, match="day_count_convention must be"):
            MarketData(PRICING_DATE, CURVE, currency="USD", day_count_convention="ACT_365F")


# ---------------------------------------------------------------------------
# CorrelationContext
# ---------------------------------------------------------------------------


def _valid_corr_context(**overrides):
    """Build a valid 2-asset CorrelationContext, with optional overrides."""
    defaults = dict(
        correlation_matrix=np.array([[1.0, 0.5], [0.5, 1.0]]),
        random_numbers=np.random.default_rng(0).standard_normal((2, 10, 100)),
        asset_names=["A", "B"],
    )
    defaults.update(overrides)
    return CorrelationContext(**defaults)


class TestCorrelationContextValid:
    def test_valid_construction(self):
        ctx = _valid_corr_context()
        assert ctx.cholesky_matrix.shape == (2, 2)
        assert ctx.asset_index("A") == 0
        assert ctx.asset_index("B") == 1

    def test_cholesky_reconstructs_correlation(self):
        C = np.array([[1.0, 0.5], [0.5, 1.0]])
        ctx = _valid_corr_context(correlation_matrix=C)
        np.testing.assert_allclose(ctx.cholesky_matrix @ ctx.cholesky_matrix.T, C)


class TestCorrelationContextValidation:
    def test_rejects_non_square_matrix(self):
        with pytest.raises(ValidationError, match="must be square"):
            _valid_corr_context(correlation_matrix=np.ones((2, 3)))

    def test_rejects_1d_matrix(self):
        with pytest.raises(ValidationError, match="must be square"):
            _valid_corr_context(correlation_matrix=np.ones(4))

    def test_rejects_mismatched_asset_names_length(self):
        with pytest.raises(ValidationError, match="asset_names length"):
            _valid_corr_context(asset_names=["A", "B", "C"])

    def test_rejects_duplicate_asset_names(self):
        with pytest.raises(ValidationError, match="unique entries"):
            _valid_corr_context(asset_names=["A", "A"])

    def test_rejects_asymmetric_matrix(self):
        C = np.array([[1.0, 0.5], [0.3, 1.0]])
        with pytest.raises(ValidationError, match="symmetric"):
            _valid_corr_context(correlation_matrix=C)

    def test_rejects_non_unit_diagonal(self):
        C = np.array([[2.0, 0.5], [0.5, 1.0]])
        with pytest.raises(ValidationError, match="diagonal must be 1"):
            _valid_corr_context(correlation_matrix=C)

    def test_rejects_non_positive_definite(self):
        # Correlation > 1 in off-diag makes it not positive-definite
        C = np.array([[1.0, 1.5], [1.5, 1.0]])
        with pytest.raises(ValidationError, match="not positive-definite"):
            _valid_corr_context(correlation_matrix=C)

    def test_rejects_2d_random_numbers(self):
        rn = np.ones((2, 10))
        with pytest.raises(ValidationError, match="must be 3-D"):
            _valid_corr_context(random_numbers=rn)

    def test_rejects_wrong_random_numbers_axis0(self):
        rn = np.ones((3, 10, 100))  # 3 != 2 assets
        with pytest.raises(ValidationError, match="axis-0"):
            _valid_corr_context(random_numbers=rn)

    def test_unknown_asset_name_raises(self):
        ctx = _valid_corr_context()
        with pytest.raises(ValidationError, match="not found in asset_names"):
            ctx.asset_index("C")
