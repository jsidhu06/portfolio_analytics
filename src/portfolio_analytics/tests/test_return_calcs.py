import pandas as pd
import pytest
from portfolio_analytics.return_calcs import (
    assert_price_dividend_series_aligned,
    calculate_total_return_index_ts,
    calculate_daily_total_return_gross_dividends_ts,
)


class TestAssertPriceDividendSeriesAligned:
    """Test suite for assert_price_dividend_series_aligned function"""

    def test_aligned_series_with_same_length_and_index(self):
        """Test that aligned series with same length and index pass without error"""
        date_range = pd.date_range("2025-01-01", "2025-01-10", freq="D")
        price_series = pd.Series(
            [100, 101, 102, 103, 104, 105, 106, 107, 108, 109], index=date_range
        )
        dividend_series = pd.Series([0, 0.5, 0, 0.5, 0, 0.5, 0, 0.5, 0, 0.5], index=date_range)

        # Should not raise any exception
        assert_price_dividend_series_aligned(price_series, dividend_series)

    def test_misaligned_series_different_lengths(self):
        """Test that series with different lengths raise AssertionError"""
        date_range = pd.date_range("2025-01-01", "2025-01-10", freq="D")
        price_series = pd.Series(
            [100, 101, 102, 103, 104, 105, 106, 107, 108, 109], index=date_range
        )
        dividend_series = pd.Series([0, 0.5, 0, 0.5, 0], index=date_range[:5])

        with pytest.raises(AssertionError, match="price and dividend series are misaligned"):
            assert_price_dividend_series_aligned(price_series, dividend_series)

    def test_misaligned_series_different_index(self):
        """Test that series with same length but different index raise AssertionError"""
        date_range_1 = pd.date_range("2025-01-01", "2025-01-10", freq="D")
        date_range_2 = pd.date_range("2025-01-02", "2025-01-11", freq="D")

        price_series = pd.Series(
            [100, 101, 102, 103, 104, 105, 106, 107, 108, 109], index=date_range_1
        )
        dividend_series = pd.Series([0, 0.5, 0, 0.5, 0, 0.5, 0, 0.5, 0, 0.5], index=date_range_2)

        with pytest.raises(AssertionError, match="price and dividend series are misaligned"):
            assert_price_dividend_series_aligned(price_series, dividend_series)

    def test_misaligned_series_partial_index_overlap(self):
        """Test that series with partial index overlap raise AssertionError"""
        date_range_1 = pd.date_range("2025-01-01", "2025-01-10", freq="D")
        date_range_2 = pd.date_range("2025-01-05", "2025-01-14", freq="D")

        price_series = pd.Series(range(100, 110), index=date_range_1)
        dividend_series = pd.Series(range(0, 10), index=date_range_2)

        with pytest.raises(AssertionError, match="price and dividend series are misaligned"):
            assert_price_dividend_series_aligned(price_series, dividend_series)

    def test_aligned_series_with_integer_index(self):
        """Test that aligned series with integer index pass without error"""
        price_series = pd.Series([100, 101, 102, 103, 104])
        dividend_series = pd.Series([0, 0.5, 0, 0.5, 0])

        # Should not raise any exception
        assert_price_dividend_series_aligned(price_series, dividend_series)

    def test_misaligned_series_different_integer_index(self):
        """Test that series with different integer indices raise AssertionError"""
        price_series = pd.Series([100, 101, 102, 103, 104], index=[0, 1, 2, 3, 4])
        dividend_series = pd.Series([0, 0.5, 0, 0.5, 0], index=[1, 2, 3, 4, 5])

        with pytest.raises(AssertionError, match="price and dividend series are misaligned"):
            assert_price_dividend_series_aligned(price_series, dividend_series)

    def test_aligned_single_element_series(self):
        """Test that aligned single-element series pass without error"""
        price_series = pd.Series([100])
        dividend_series = pd.Series([0.5])

        # Should not raise any exception
        assert_price_dividend_series_aligned(price_series, dividend_series)


class TestCalculateTotalReturnIndex:
    """Test suite for calculate_total_return_index_ts and calculate_daily_total_return_gross_dividends_ts"""

    def test_pct_change_total_return_index_equals_daily_gross_dividends(self):
        """
        Test that pct_change(periods=1) of total_return_index equals daily_total_return_gross_dividends.

        By mathematical proof:
        If TR_t = P_t * N_t (total return index)
        And N_t = N_{t-1} * (1 + D_t / P_t) (share count)
        Then TR_t / TR_{t-1} = (P_t * N_t) / (P_{t-1} * N_{t-1})
                             = (P_t / P_{t-1}) * (N_t / N_{t-1})
                             = (P_t / P_{t-1}) * (1 + D_t / P_t)
                             = (P_t + D_t) / P_{t-1}
        Which is exactly what calculate_daily_total_return_gross_dividends_ts computes.
        """
        date_range = pd.date_range("2025-01-01", "2025-01-15", freq="D")
        price_series = pd.Series(
            [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114],
            index=date_range,
        )
        dividend_series = pd.Series(
            [0, 0.5, 0, 1.0, 0, 0.5, 0, 0.75, 0, 0.5, 0, 1.0, 0, 0.5, 0], index=date_range
        )

        # Calculate total return index and take pct_change
        total_return_index = calculate_total_return_index_ts(price_series, dividend_series)
        pct_change_result = total_return_index.pct_change(periods=1)

        # Calculate daily total return with gross dividends
        daily_return_result = calculate_daily_total_return_gross_dividends_ts(
            price_series, dividend_series
        )

        # Both should be equal (ignoring the first NaN value from pct_change)
        pd.testing.assert_series_equal(
            pct_change_result.iloc[1:].reset_index(drop=True),
            daily_return_result.iloc[1:].reset_index(drop=True),
            atol=1e-10,
            check_names=False,
        )
