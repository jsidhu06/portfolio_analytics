import numpy as np
import pandas as pd
import pytest
import logging
from portfolio_analytics.return_calcs import (
    assert_price_dividend_series_aligned,
    calculate_total_return_index_ts,
    calculate_daily_total_return_gross_dividends_ts,
    calculate_total_return_over_period,
    generate_returns_df,
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


class TestCalculateTotalReturnOverPeriod:
    def test_total_return_over_period_with_dividends(self):
        date_range = pd.date_range("2025-01-01", "2025-01-10", freq="D")
        price_series = pd.Series(
            [100, 102, 101, 103, 104, 106, 105, 107, 108, 110], index=date_range
        )
        dividend_series = pd.Series([0, 1.0, 0, 0.5, 0, 1.0, 0, 0.5, 0, 1.0], index=date_range)

        total_return = calculate_total_return_over_period(price_series, dividend_series)

        # Manually calculate expected total return
        expected_return = (110 / 100) * (1 + 1.0 / 102) * (1 + 0.5 / 103) * (1 + 1.0 / 106) * (
            1 + 0.5 / 107
        ) * (1 + 1.0 / 110) - 1

        assert np.isclose(total_return, expected_return)

    def test_total_return_equivalence(self):
        date_range = pd.date_range("2025-01-01", "2025-01-10", freq="D")
        price_series = pd.Series(
            [100, 102, 101, 103, 104, 106, 105, 107, 108, 110], index=date_range
        )
        dividend_series = pd.Series([0, 1.0, 0, 0.5, 0, 1.0, 0, 0.5, 0, 1.0], index=date_range)

        total_return = calculate_total_return_over_period(price_series, dividend_series)

        # Calculate total return using pct change of total return index and geometric compounding
        # of daily return methods. All 3 should yield the same result.
        total_return_index = calculate_total_return_index_ts(price_series, dividend_series)
        total_return_via_index = (total_return_index.iloc[-1] / total_return_index.iloc[0]) - 1

        daily_total_return_ts = calculate_daily_total_return_gross_dividends_ts(
            price_series, dividend_series
        )

        total_return_via_daily = (daily_total_return_ts + 1).prod() - 1

        assert np.allclose(
            np.stack([total_return, total_return_via_index, total_return_via_daily]), total_return
        )


class TestGenerateReturnsDf:
    """Test suite for generate_returns_df function"""

    def test_generate_returns_df_basic_structure(self):
        """Test that output has correct structure: single-level columns (tickers)"""
        # Create multi-indexed price DataFrame
        dates = pd.date_range("2025-01-01", periods=5, freq="D")
        cols = pd.MultiIndex.from_product(
            [["AAPL", "MSFT"], ["Close", "Dividends"]], names=["Ticker", "Field"]
        )
        data = np.array(
            [
                [100.0, 0.0, 50.0, 0.0],  # Day 1
                [101.0, 0.0, 51.0, 0.5],  # Day 2 (MSFT dividend)
                [102.0, 0.5, 50.5, 0.0],  # Day 3 (AAPL dividend)
                [103.0, 0.0, 51.5, 0.0],  # Day 4
                [104.0, 0.0, 52.0, 0.0],  # Day 5
            ]
        )
        price_df = pd.DataFrame(data, index=dates, columns=cols)

        returns_df = generate_returns_df(price_df)

        # Output should have single-level columns (tickers only)
        assert isinstance(returns_df.columns, pd.Index)
        assert not isinstance(returns_df.columns, pd.MultiIndex)
        assert set(returns_df.columns) == {"AAPL", "MSFT"}

    def test_generate_returns_df_first_row_dropped(self):
        """Test that first row (with NaN) is dropped"""
        dates = pd.date_range("2025-01-01", periods=5, freq="D")
        cols = pd.MultiIndex.from_product(
            [["AAPL", "MSFT"], ["Close", "Dividends"]], names=["Ticker", "Field"]
        )
        data = np.array(
            [
                [100.0, 0.0, 50.0, 0.0],
                [101.0, 0.0, 51.0, 0.0],
                [102.0, 0.0, 50.0, 0.0],
                [103.0, 0.0, 51.0, 0.0],
                [104.0, 0.0, 52.0, 0.0],
            ]
        )
        price_df = pd.DataFrame(data, index=dates, columns=cols)

        returns_df = generate_returns_df(price_df)

        # First row should be dropped; output should have 4 rows
        assert len(returns_df) == 4
        assert returns_df.index[0] == dates[1]  # First remaining row is second date

    def test_generate_returns_df_no_nans_except_first_row(self):
        """Test that no NaN values remain after first row is dropped"""
        dates = pd.date_range("2025-01-01", periods=5, freq="D")
        cols = pd.MultiIndex.from_product(
            [["AAPL"], ["Close", "Dividends"]], names=["Ticker", "Field"]
        )
        data = np.array(
            [
                [100.0, 0.0],
                [101.0, 0.0],
                [102.0, 0.5],
                [103.0, 0.0],
                [104.0, 0.0],
            ]
        )
        price_df = pd.DataFrame(data, index=dates, columns=cols)

        returns_df = generate_returns_df(price_df)

        # No NaN values should remain
        assert returns_df.notna().all(axis=None)

    def test_generate_returns_df_with_dividends(self):
        """Test return calculation includes dividends"""
        dates = pd.date_range("2025-01-01", periods=3, freq="D")
        cols = pd.MultiIndex.from_product(
            [["STOCK"], ["Close", "Dividends"]], names=["Ticker", "Field"]
        )
        # Day 1: price 100, no dividend
        # Day 2: price 101, no dividend -> return = (101 + 0) / 100 - 1 = 0.01
        # Day 3: price 102, dividend 1 -> return = (102 + 1) / 101 - 1 ≈ 0.00990
        data = np.array(
            [
                [100.0, 0.0],
                [101.0, 0.0],
                [102.0, 1.0],
            ]
        )
        price_df = pd.DataFrame(data, index=dates, columns=cols)

        returns_df = generate_returns_df(price_df)

        # Check day 2 return
        assert np.isclose(returns_df.loc[dates[1], "STOCK"], 0.01)
        # Check day 3 return (includes dividend)
        expected_day3 = (102.0 + 1.0) / 101.0 - 1.0
        assert np.isclose(returns_df.loc[dates[2], "STOCK"], expected_day3)

    def test_generate_returns_df_multiple_stocks(self):
        """Test with multiple stocks"""
        dates = pd.date_range("2025-01-01", periods=4, freq="D")
        cols = pd.MultiIndex.from_product(
            [["AAPL", "MSFT", "GOOGL"], ["Close", "Dividends"]], names=["Ticker", "Field"]
        )
        data = np.array(
            [
                [100.0, 0.0, 50.0, 0.0, 75.0, 0.0],
                [101.0, 0.0, 51.0, 0.0, 74.0, 0.0],
                [102.0, 0.0, 50.5, 0.0, 76.0, 0.0],
                [103.0, 0.0, 51.5, 0.0, 75.5, 0.0],
            ]
        )
        price_df = pd.DataFrame(data, index=dates, columns=cols)

        returns_df = generate_returns_df(price_df)

        assert set(returns_df.columns) == {"AAPL", "MSFT", "GOOGL"}
        assert len(returns_df) == 3  # First row dropped

    def test_generate_returns_df_zero_returns(self):
        """Test when prices don't change (zero returns)"""
        dates = pd.date_range("2025-01-01", periods=4, freq="D")
        cols = pd.MultiIndex.from_product(
            [["STOCK"], ["Close", "Dividends"]], names=["Ticker", "Field"]
        )
        data = np.array(
            [
                [100.0, 0.0],
                [100.0, 0.0],
                [100.0, 0.0],
                [100.0, 0.0],
            ]
        )
        price_df = pd.DataFrame(data, index=dates, columns=cols)

        returns_df = generate_returns_df(price_df)

        # All returns should be zero
        assert np.allclose(returns_df.values, 0.0)

    def test_generate_returns_df_negative_returns(self):
        """Test when prices decline (negative returns)"""
        dates = pd.date_range("2025-01-01", periods=4, freq="D")
        cols = pd.MultiIndex.from_product(
            [["STOCK"], ["Close", "Dividends"]], names=["Ticker", "Field"]
        )
        data = np.array(
            [
                [100.0, 0.0],
                [99.0, 0.0],
                [98.0, 0.0],
                [97.0, 0.0],
            ]
        )
        price_df = pd.DataFrame(data, index=dates, columns=cols)

        returns_df = generate_returns_df(price_df)

        # All returns should be negative
        assert (returns_df < 0).all(axis=None)
        # Check specific value: -1/100 = -0.01
        assert np.isclose(returns_df.iloc[0, 0], -0.01)

    def test_generate_returns_df_large_price_change(self):
        """Test with large price changes"""
        dates = pd.date_range("2025-01-01", periods=3, freq="D")
        cols = pd.MultiIndex.from_product(
            [["STOCK"], ["Close", "Dividends"]], names=["Ticker", "Field"]
        )
        data = np.array(
            [
                [100.0, 0.0],
                [200.0, 0.0],  # 100% return
                [50.0, 0.0],  # -75% return
            ]
        )
        price_df = pd.DataFrame(data, index=dates, columns=cols)

        returns_df = generate_returns_df(price_df)

        # Day 2: 100% return
        assert np.isclose(returns_df.iloc[0, 0], 1.0)
        # Day 3: -75% return
        assert np.isclose(returns_df.iloc[1, 0], -0.75)

    def test_generate_returns_df_index_preservation(self):
        """Test that date index is preserved (except first row)"""
        dates = pd.date_range("2025-01-01", periods=5, freq="D")
        cols = pd.MultiIndex.from_product(
            [["AAPL"], ["Close", "Dividends"]], names=["Ticker", "Field"]
        )
        data = np.ones((5, 2)) * 100.0
        price_df = pd.DataFrame(data, index=dates, columns=cols)

        returns_df = generate_returns_df(price_df)

        # Index should be all dates except the first
        assert returns_df.index.equals(dates[1:])

    def test_generate_returns_df_with_missing_dividends_logs_warning(self, caplog):
        """Test that missing Dividends field logs warning and uses Close only"""

        dates = pd.date_range("2025-01-01", periods=3, freq="D")
        # Missing Dividends field
        cols = pd.MultiIndex.from_product([["AAPL"], ["Close", "Open"]], names=["Ticker", "Field"])
        data = np.array(
            [
                [100.0, 99.0],
                [101.0, 100.0],
                [102.0, 101.0],
            ]
        )
        price_df = pd.DataFrame(data, index=dates, columns=cols)

        with caplog.at_level(logging.WARNING):
            returns_df = generate_returns_df(price_df)

        # Check that warning was logged
        assert "No dividend data for AAPL" in caplog.text
        assert len(returns_df) == 2  # First row dropped
        # Check returns match pct_change: (101-100)/100 = 0.01, (102-101)/101 ≈ 0.0099
        assert np.isclose(returns_df.iloc[0, 0], 0.01)
        assert np.isclose(returns_df.iloc[1, 0], (102.0 - 101.0) / 101.0)

    def test_generate_returns_df_fillna_zero_dividends(self):
        """Test that NaN dividends are treated as zero"""
        dates = pd.date_range("2025-01-01", periods=4, freq="D")
        cols = pd.MultiIndex.from_product(
            [["STOCK"], ["Close", "Dividends"]], names=["Ticker", "Field"]
        )
        data = np.array(
            [
                [100.0, np.nan],
                [101.0, np.nan],
                [102.0, 0.5],
                [103.0, np.nan],
            ]
        )
        price_df = pd.DataFrame(data, index=dates, columns=cols)

        returns_df = generate_returns_df(price_df)

        # Should not raise and should produce valid returns
        assert returns_df.notna().all(axis=None)
        # Day 3 should account for dividend
        expected_day3 = (102.0 + 0.5) / 101.0 - 1.0
        assert np.isclose(returns_df.iloc[1, 0], expected_day3)
