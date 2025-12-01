import logging
import pandas as pd

logging.basicConfig(level=logging.WARNING, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

idx = pd.IndexSlice


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
    Results are expressed in decimal.

    Args:
        price_series (pd.Series): Series of prices indexed by date/time.
        dividend_series (pd.Series): Series of dividends indexed by date/time.
                                    Must be aligned with price_series.

    Returns:
        pd.Series: Daily total returns (in decimal format) with the same index as the input series.

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


def calculate_total_return_over_period(
    price_series: pd.Series, dividend_series: pd.Series
) -> float:
    """Calculate the total return over the entire period.

    This uses the canonical standard total return calculation:
    \mathrm{TRA}(t, t+n)
    = \frac{P_{t+n}}{P_t}
      \prod_{i=1}^{n}
      \left( 1 + \frac{D_{t+i}}{P_{t+i}} \right)


    Args:
        price_series (pd.Series): Series of prices indexed by date/time.
        dividend_series (pd.Series): Series of dividends indexed by date/time.
                                    Must be aligned with price_series.

    Returns:
        float: Total return over the period in decimal format.

    Raises:
        AssertionError: If price_series and dividend_series are not aligned.
    """
    assert_price_dividend_series_aligned(price_series, dividend_series)
    multiplier = (1 + dividend_series.fillna(0) / price_series.ffill()).prod()
    total_return = (price_series.iloc[-1] / price_series.iloc[0]) * multiplier - 1
    return total_return


def generate_returns_df(price_df: pd.DataFrame) -> pd.DataFrame:
    """Generate daily returns DataFrame from price DataFrame.

    Args:
        price_df (pd.DataFrame): DataFrame with index as date and a multi-indexed column (stock, field).

    Returns:
        pd.DataFrame: DataFrame of daily returns with the same index and columns as input.
    """

    def has_dividends(df: pd.DataFrame) -> bool:
        return "Dividends" in df.index.get_level_values(1)

    def calculate_returns_for_group(df: pd.DataFrame) -> pd.Series:
        if has_dividends(df):
            return calculate_daily_total_return_gross_dividends_ts(
                df.T.droplevel(axis=1, level=0)["Close"],
                df.T.droplevel(axis=1, level=0)["Dividends"],
            )

        logger.warning(
            f"No dividend data for {df.index.get_level_values(0).unique()[0]}. "
            "Calculating returns using price data only."
        )
        return (
            df.T.droplevel(axis=1, level=0)["Close"]
            .pct_change(periods=1, fill_method=None)
            .rename("return_series")
        )

    return (
        price_df.T.groupby("Ticker", group_keys=True)
        .apply(calculate_returns_for_group)
        .T.iloc[1:]
        .fillna(0)
    )  # Drop the first row with NaN values due to pct_change
