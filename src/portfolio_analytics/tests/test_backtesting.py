import numpy as np
import pandas as pd
from portfolio_analytics.backtesting import get_portfolio_returns_series


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
