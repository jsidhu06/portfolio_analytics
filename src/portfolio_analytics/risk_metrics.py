from typing import Union
import numpy as np
import pandas as pd


def assert_price_dividend_series_aligned(
    price_series: pd.Series, dividend_series: pd.Series
) -> None:
    """Assert that price and dividend series are aligned.

    The series are considered aligned if they have the same length and
    their indices are identical.

    Args:
        price_series (pd.Series): Series of prices.
        dividend_series (pd.Series): Series of dividends.

    Raises:
        AssertionError: If the series are not aligned (different lengths or indices).
    """
    assert (
        len(price_series) == len(dividend_series)
        and (price_series.index == dividend_series.index).all()
    ), "price and dividend series are misaligned"


def calculate_total_return_index_share_count_ts(
    price_series: pd.Series, dividend_series: pd.Series
) -> pd.Series:
    """Calculate the share count time series for total return index.

    Starting with 1 share, calculates the number of shares held over time,
    accounting for dividend reinvestment using the formula:
    shares_t = shares_(t-1) * (1 + dividend_t / price_t).

    Args:
        price_series (pd.Series): Series of prices indexed by date/time.
        dividend_series (pd.Series): Series of dividends indexed by date/time.
                                    Must be aligned with price_series.

    Returns:
        pd.Series: Share count time series with the same index as the input series.

    Raises:
        AssertionError: If price_series and dividend_series are not aligned.
    """
    assert_price_dividend_series_aligned(price_series, dividend_series)

    share_count_ts = []
    prior_share_count = 1

    for close_price, dividend in zip(price_series, dividend_series):
        num_shares = prior_share_count * (1 + dividend / close_price)  # (1+D_t/P_t)
        share_count_ts.append(num_shares)
        prior_share_count = num_shares

    return pd.Series(share_count_ts, index=price_series.index)


def calculate_total_return_index_ts(
    price_series: pd.Series, dividend_series: pd.Series
) -> pd.Series:
    """Calculate the total return index time series.

    The total return index accounts for both price appreciation and dividend
    reinvestment, calculated as: total_return_index_t = price_t * share_count_t.

    Args:
        price_series (pd.Series): Series of prices indexed by date/time.
        dividend_series (pd.Series): Series of dividends indexed by date/time.
                                    Must be aligned with price_series.

    Returns:
        pd.Series: Total return index time series with the same index as the input series.

    Raises:
        AssertionError: If price_series and dividend_series are not aligned.
    """
    assert_price_dividend_series_aligned(price_series, dividend_series)
    share_count_ts = calculate_total_return_index_share_count_ts(price_series, dividend_series)
    return price_series * share_count_ts


def calculate_daily_total_return_gross_dividends_ts(
    price_series: pd.Series, dividend_series: pd.Series
) -> pd.Series:
    """Calculate daily total returns including gross dividends.

    Calculates the daily percentage returns that account for both price changes
    and dividend payments using the formula: return_t = ((price_t + dividend_t) / price_(t-1)) - 1.
    Results are expressed as percentages. This is RT112 on the Bloomberg terminal.

    Args:
        price_series (pd.Series): Series of prices indexed by date/time.
        dividend_series (pd.Series): Series of dividends indexed by date/time.
                                    Must be aligned with price_series.

    Returns:
        pd.Series: Daily total returns (in percentages) with the same index as the input series.

    Raises:
        AssertionError: If price_series and dividend_series are not aligned.
    """
    assert_price_dividend_series_aligned(price_series, dividend_series)
    return_series = (
        ((price_series.ffill() + dividend_series.fillna(0)) / price_series.ffill().shift(1))
        .sub(1)
        .rename("return_series")
    )

    return return_series


def get_annualization_dict() -> dict:
    """Get dictionary mapping frequency strings to annualization factors.

    Returns:
        dict: Dictionary with keys ('daily', 'weekly', 'monthly', 'quarterly', 'yearly')
              and their corresponding annualization factors.
    """
    return {
        "daily": 252,
        "weekly": 52,
        "monthly": 12,
        "quarterly": 4,
        "yearly": 1,
    }


def calculate_portfolio_vol(returns_series: pd.Series, frequency="daily") -> float:
    """Calculate annualized portfolio volatility (standard deviation of returns).

    Args:
        returns_series (pd.Series): Series of returns.
        frequency (str): Frequency of returns. Must be one of 'daily', 'weekly',
                        'monthly', 'quarterly', or 'yearly'. Defaults to 'daily'.

    Returns:
        float: Annualized volatility (standard deviation of returns).

    Raises:
        ValueError: If frequency is not one of the supported values.
    """

    annualization_dict = get_annualization_dict()
    annualization_factor = annualization_dict.get(frequency)
    if annualization_factor is None:
        raise ValueError(
            "frequency param must be one of 'daily','weekly','monthly','quarterly' or 'yearly'"
        )

    return returns_series.std() * np.sqrt(annualization_factor)


def calculate_sharpe_ratio(
    returns_series: pd.Series,
    risk_free_rate: Union[int, float] = 0,
    returns_frequency: str = "daily",
) -> float:
    """Calculate the Sharpe ratio of a returns series.

    Sharpe Ratio = (Mean portfolio return - risk free rate) / portfolio volatility

    Args:
        returns_series (pd.Series): Series of returns.
        risk_free_rate (Union[int, float]): Risk-free rate. Defaults to 0.
        returns_frequency (str): Frequency of returns. Must be one of 'daily', 'weekly',
                               'monthly', 'quarterly', or 'yearly'. Defaults to 'daily'.

    Returns:
        float: Sharpe ratio.

    Raises:
        ValueError: If returns_frequency is not one of the supported values.
    """

    portfolio_vol = calculate_portfolio_vol(returns_series, returns_frequency)
    annualization_dict = get_annualization_dict()
    annualization_factor = annualization_dict.get(returns_frequency)
    if annualization_factor is None:
        raise ValueError(
            "frequency param must be one of 'daily','weekly','monthly','quarterly' or 'yearly'"
        )

    return (returns_series.mean() * annualization_factor - risk_free_rate) / portfolio_vol
