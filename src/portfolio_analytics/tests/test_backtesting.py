import numpy as np
import pandas as pd
from portfolio_analytics.backtesting import (
    get_portfolio_returns_series,
    get_daily_quantile_portfolio_returns,
    get_quantile_portfolio_returns,
    get_quantile_portfolio_returns_df,
)


class TestGetPortfolioReturnsSeries:
    """Test suite for get_portfolio_returns_series function"""

    def test_portfolio_returns_equal_weights(self):
        """Test portfolio returns calculation with equal weights"""
        # Create sample weights for 3 stocks
        weights = pd.Series(
            {
                "AAPL": 0.33,
                "MSFT": 0.33,
                "GOOGL": 0.34,
            }
        )

        # Create sample daily returns for 10 days and 3 stocks
        dates = pd.date_range("2025-01-01", periods=10, freq="D")
        returns_df = pd.DataFrame(
            {
                "AAPL": [0.01, 0.02, -0.01, 0.015, 0.005, 0.02, -0.005, 0.01, 0.015, 0.02],
                "MSFT": [0.015, 0.01, 0.02, -0.01, 0.025, 0.01, 0.015, 0.02, -0.005, 0.01],
                "GOOGL": [0.02, -0.01, 0.015, 0.02, 0.01, -0.005, 0.02, 0.015, 0.01, 0.025],
            },
            index=dates,
        )

        # Calculate portfolio returns
        portfolio_returns = get_portfolio_returns_series(weights, returns_df)

        # Assertions
        assert isinstance(portfolio_returns, pd.Series)
        assert len(portfolio_returns) == len(returns_df)
        assert portfolio_returns.index.equals(returns_df.index)

        # For first day, portfolio return should be weighted average of individual returns
        expected_first_return = 0.01 * 0.33 + 0.015 * 0.33 + 0.02 * 0.34
        assert np.isclose(portfolio_returns.iloc[0], expected_first_return)

    def test_portfolio_returns_with_subset_of_stocks(self):
        """Test portfolio returns when weights include stocks not in returns_df"""
        # Create weights for 4 stocks
        weights = pd.Series(
            {
                "AAPL": 0.25,
                "MSFT": 0.25,
                "GOOGL": 0.25,
                "AMZN": 0.25,  # This stock is not in returns_df
            }
        )

        # Create returns for only 3 stocks
        dates = pd.date_range("2025-01-01", periods=5, freq="D")
        returns_df = pd.DataFrame(
            {
                "AAPL": [0.01, 0.02, -0.01, 0.015, 0.005],
                "MSFT": [0.015, 0.01, 0.02, -0.01, 0.025],
                "GOOGL": [0.02, -0.01, 0.015, 0.02, 0.01],
            },
            index=dates,
        )

        # Calculate portfolio returns
        portfolio_returns = get_portfolio_returns_series(weights, returns_df)

        # Assertions
        assert isinstance(portfolio_returns, pd.Series)
        assert len(portfolio_returns) == len(returns_df)
        assert not portfolio_returns.isna().any()

    def test_portfolio_returns_single_stock(self):
        """Test portfolio returns with a single stock (weight = 1.0)"""
        weights = pd.Series(
            {
                "AAPL": 1.0,
            }
        )

        dates = pd.date_range("2025-01-01", periods=5, freq="D")
        returns_df = pd.DataFrame(
            {
                "AAPL": [0.01, 0.02, -0.01, 0.015, 0.005],
            },
            index=dates,
        )

        portfolio_returns = get_portfolio_returns_series(weights, returns_df)

        # Portfolio returns should match AAPL returns exactly
        assert np.allclose(portfolio_returns.values, returns_df["AAPL"].values)

    def test_portfolio_returns_zero_returns(self):
        """Test portfolio returns with zero daily returns"""
        weights = pd.Series(
            {
                "AAPL": 0.5,
                "MSFT": 0.5,
            }
        )

        dates = pd.date_range("2025-01-01", periods=5, freq="D")
        returns_df = pd.DataFrame(
            {
                "AAPL": [0.0, 0.0, 0.0, 0.0, 0.0],
                "MSFT": [0.0, 0.0, 0.0, 0.0, 0.0],
            },
            index=dates,
        )

        portfolio_returns = get_portfolio_returns_series(weights, returns_df)

        # All portfolio returns should be zero
        assert np.allclose(portfolio_returns.values, 0.0)

    def test_portfolio_returns_negative_returns(self):
        """Test portfolio returns with negative daily returns"""
        weights = pd.Series(
            {
                "AAPL": 0.4,
                "MSFT": 0.6,
            }
        )

        dates = pd.date_range("2025-01-01", periods=5, freq="D")
        returns_df = pd.DataFrame(
            {
                "AAPL": [-0.05, -0.02, -0.03, -0.01, -0.04],
                "MSFT": [-0.02, -0.03, -0.02, -0.04, -0.01],
            },
            index=dates,
        )

        portfolio_returns = get_portfolio_returns_series(weights, returns_df)

        # All portfolio returns should be negative
        assert (portfolio_returns < 0).all()

    def test_portfolio_returns_with_zero_weights(self):
        """Test that stocks with zero weight don't impact portfolio returns"""
        # Create returns for 5 stocks
        dates = pd.date_range("2025-01-01", periods=5, freq="D")
        returns_df = pd.DataFrame(
            {
                "AAPL": [0.01, 0.02, -0.01, 0.015, 0.005],
                "MSFT": [0.015, 0.01, 0.02, -0.01, 0.025],
                "GOOGL": [0.02, -0.01, 0.015, 0.02, 0.01],
                "AMZN": [0.005, 0.015, -0.02, 0.01, 0.02],
                "TSLA": [-0.01, 0.025, 0.01, -0.015, 0.03],
            },
            index=dates,
        )

        # Case 1: Weights with only 3 stocks (non-zero weights)
        weights_without_zeros = pd.Series(
            {
                "AAPL": 0.33,
                "MSFT": 0.33,
                "GOOGL": 0.34,
            }
        )

        # Case 2: Same weights but with explicit 0 weights for AMZN and TSLA
        weights_with_zeros = pd.Series(
            {
                "AAPL": 0.33,
                "MSFT": 0.33,
                "GOOGL": 0.34,
                "AMZN": 0.0,
                "TSLA": 0.0,
            }
        )

        # Calculate portfolio returns for both cases
        portfolio_returns_without_zeros = get_portfolio_returns_series(
            weights_without_zeros, returns_df
        )
        portfolio_returns_with_zeros = get_portfolio_returns_series(weights_with_zeros, returns_df)

        # Returns should be identical
        assert np.allclose(
            portfolio_returns_without_zeros.values, portfolio_returns_with_zeros.values
        )


def make_weights_multiindex_df():
    rebalance_dates = [pd.Timestamp("2024-12-31"), pd.Timestamp("2025-01-31")]
    stocks = ["AAPL", "MSFT", "GOOGL"]
    rows = []
    for d in rebalance_dates:
        for s in stocks:
            rows.append((d, s))

    mi = pd.MultiIndex.from_tuples(rows, names=["date", "stock"])

    data = []
    for d in rebalance_dates:
        if d == rebalance_dates[0]:
            data.extend(
                [
                    {"Q1": 1.0, "Q2": 0.0},  # AAPL
                    {"Q1": 0.0, "Q2": 0.5},  # MSFT
                    {"Q1": 0.0, "Q2": 0.5},  # GOOGL
                ]
            )
        else:
            data.extend(
                [
                    {"Q1": 0.0, "Q2": 0.5},  # AAPL
                    {"Q1": 1.0, "Q2": 0.0},  # MSFT
                    {"Q1": 0.0, "Q2": 0.5},  # GOOGL
                ]
            )

    return pd.DataFrame(data, index=mi)


def make_fake_price_df(stocks, start_date, end_date):
    # Create daily dates inclusive of start and end
    dates = pd.date_range(start_date, end_date, freq="D")

    # Use a reproducible RNG so tests are deterministic
    rng = np.random.RandomState(42)

    # Build MultiIndex columns (Ticker, Field)
    data = {}
    for i, s in enumerate(stocks):
        # Generate small daily returns (some positive, some negative)
        daily_ret = rng.normal(loc=0.0005, scale=0.02, size=len(dates))
        # Ensure variability across stocks by shifting mean slightly
        daily_ret += i * 0.0001

        # Build close prices via cumulative returns from a base price
        base = 100 + i * 5
        close = base * np.cumprod(1 + daily_ret)

        # small dividends series: mostly zero, occasional small dividend
        dividends = np.zeros(len(dates))
        dividends[::3] = 0.5  # pay dividend every 3rd day

        data[(s, "Close")] = close
        data[(s, "Dividends")] = dividends

    # Create MultiIndex and DataFrame
    cols = pd.MultiIndex.from_tuples(list(data.keys()), names=["Ticker", "Field"])
    df = pd.DataFrame(data=list(data.values()), index=cols).T
    df.index = dates
    return df


def test_get_quantile_portfolio_returns_df(monkeypatch):
    """Integration test for get_quantile_portfolio_returns_df using mocked price fetcher."""
    weights_mi_df = make_weights_multiindex_df()

    # The backtesting module calls fetch_historical_price_data(stocks, start_date, end_date, actions=True)
    def fake_fetch_historical_price_data(stocks, start, end, actions=True):
        return make_fake_price_df(stocks, start, end)

    # Patch the fetcher in the backtesting module
    monkeypatch.setattr(
        "portfolio_analytics.backtesting.fetch_historical_price_data",
        fake_fetch_historical_price_data,
    )

    # Run the function under test
    result_df = get_quantile_portfolio_returns_df(weights_mi_df, pd.Timestamp("2025-02-28"))

    # Basic structure checks
    assert isinstance(result_df, pd.DataFrame)
    # For two rebalance dates we expect two result rows (we now include the last rebalance period)
    assert result_df.shape[0] == 2
    assert set(result_df.columns) == {"Q1", "Q2"}
    # No NaNs and numeric
    assert result_df.notna().all(axis=None)
    assert np.isfinite(result_df.values).all()

    # Values should be reasonable (not extremely large)
    assert (np.abs(result_df.values) < 1e3).all()


class TestQuantilePortfolioReturns:
    """Tests for quantile-level daily and cumulative returns functions"""

    def test_get_daily_quantile_portfolio_returns_matches_individual(self):
        """Daily quantile returns should equal running portfolio returns per-quantile"""
        # Stocks and weights for two quantiles
        weights_df = pd.DataFrame(
            {
                "Q2": pd.Series({"A": 1.0, "B": 0.0, "C": 0.0}),
                "Q1": pd.Series({"A": 0.0, "B": 0.5, "C": 0.5}),
            }
        )

        # Simple returns for 4 days
        dates = pd.date_range("2025-01-01", periods=4, freq="D")
        returns_df = pd.DataFrame(
            {
                "A": [0.01, 0.02, -0.005, 0.01],
                "B": [0.005, -0.01, 0.02, 0.0],
                "C": [0.02, 0.0, 0.01, -0.02],
            },
            index=dates,
        )

        daily_df = get_daily_quantile_portfolio_returns(weights_df, returns_df)

        expected_q1 = get_portfolio_returns_series(weights_df["Q1"], returns_df)
        expected_q2 = get_portfolio_returns_series(weights_df["Q2"], returns_df)

        assert list(daily_df.columns) == ["Q2", "Q1"]
        assert daily_df.index.equals(returns_df.index)
        assert np.allclose(daily_df["Q1"].values, expected_q1.values)
        assert np.allclose(daily_df["Q2"].values, expected_q2.values)
        # Since Q2 is just stock A, its returns should match exactly
        assert np.allclose(daily_df["Q2"].values, returns_df["A"].values)

    def test_get_quantile_portfolio_returns_cumulative(self):
        """Quantile returns should be computed from daily returns correctly"""
        weights_df = pd.DataFrame(
            {
                "Q2": pd.Series({"A": 0.3, "B": 0.7, "C": 0.0, "D": 0.0}),
                "Q1": pd.Series({"A": 0.0, "B": 0.0, "C": 0.9, "D": 0.1}),
            }
        )

        # individual security returns for 3 days
        dates = pd.date_range("2025-01-01", periods=3, freq="D")
        returns_df = pd.DataFrame(
            {
                "A": [0.01, 0.02, -0.01],
                "B": [0.005, -0.01, 0.03],
                "C": [-0.015, -0.005, 0.1],
                "D": [0.05, -0.01, 0.07],
            },
            index=dates,
        )

        daily_df = get_daily_quantile_portfolio_returns(weights_df, returns_df)
        expected_return = (1 + daily_df).prod().sub(1)

        quantile_returns = get_quantile_portfolio_returns(weights_df, returns_df)

        # Structure checks
        assert list(quantile_returns.index) == ["Q2", "Q1"]

        # Numeric equality
        assert np.allclose(quantile_returns.values, expected_return.values)

        assert quantile_returns.shape == (2,)  # shape of the series should be (num_quantiles,)
