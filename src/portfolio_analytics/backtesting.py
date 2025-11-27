import pandas as pd
from .yfinance_utils import fetch_historical_price_data
from .return_calcs import generate_returns_df


def bucket_stocks(factor_series, num_quantiles=5):
    label_mapping = {3: "T", 5: "Q", 10: "D"}
    if num_quantiles not in label_mapping:
        raise ValueError("num_quantiles must be one of [3, 5, 10]")

    label = label_mapping[num_quantiles]
    labels = [f"{label}{i}" for i in reversed(range(1, num_quantiles + 1))]
    bucket = pd.qcut(
        factor_series, q=num_quantiles, labels=labels, retbins=False, duplicates="drop"
    )
    return bucket


def get_weights_df(
    buckets: pd.Series, weighting_schema: str = "equal", market_caps: pd.Series = None
) -> pd.DataFrame:
    if weighting_schema not in ["equal", "market_cap"]:
        raise ValueError("weighting_schema must be one of ['equal','market_cap']")

    weights_df = pd.DataFrame(data=0, index=buckets.index, columns=buckets.cat.categories)
    for bucket in buckets.cat.categories:
        stocks_in_bucket = buckets[buckets == bucket].index
        if weighting_schema == "equal":
            weight = 1.0 / len(stocks_in_bucket)
            weights_df.loc[stocks_in_bucket, bucket] = weight
        elif weighting_schema == "market_cap":
            if market_caps is None:
                raise ValueError("market_caps must be provided for market_cap weighting")
            total_market_cap = market_caps.loc[stocks_in_bucket].sum()
            weights_df.loc[stocks_in_bucket, bucket] = (
                market_caps.loc[stocks_in_bucket] / total_market_cap
            )

    return weights_df


def get_portfolio_returns_series(
    weights_series: pd.DataFrame, returns_df: pd.DataFrame
) -> pd.Series:
    """
    Calculate portfolio returns given weights and stock returns.

    Args:
        weights_series (pd.Series): Series with stocks as index and weights as values.
        returns_df (pd.DataFrame): DataFrame with index as date and stocks as columns.

    Returns:
        pd.Series: Series with portfolio returns for each portfolio.
    """
    # Align indices
    common_stocks = weights_series.index.intersection(returns_df.columns)
    weights_aligned = weights_series.loc[common_stocks]
    returns_aligned = returns_df[common_stocks]

    # Calculate portfolio returns
    portfolio_returns = []
    for date in returns_df.index:
        portfolio_return = (returns_aligned.loc[date] * weights_aligned).sum()
        portfolio_returns.append(portfolio_return)
        weights_aligned = weights_aligned * (1 + returns_aligned.loc[date])
        weights_aligned /= weights_aligned.sum()

    return pd.Series(portfolio_returns, index=returns_df.index)


def get_daily_quantile_portfolio_returns(
    weights_df: pd.DataFrame, returns_df: pd.DataFrame
) -> pd.DataFrame:
    """For each quantile portfolio, calculate the (daily) returns series."""
    quantile_portfolio_returns = {}
    for quantile in weights_df.columns:
        weights_series = weights_df[quantile]
        portfolio_returns = get_portfolio_returns_series(weights_series, returns_df)
        quantile_portfolio_returns[quantile] = portfolio_returns
    return pd.DataFrame(quantile_portfolio_returns, index=returns_df.index)


def get_quantile_portfolio_returns(weights_df: pd.DataFrame, returns_df: pd.DataFrame) -> pd.Series:
    """For each quantile portfolio, calculate the total return over the period"""
    daily_returns_df = get_daily_quantile_portfolio_returns(weights_df, returns_df)
    quantile_total_returns = (1 + daily_returns_df).prod().sub(1)
    return quantile_total_returns


# Now for each rebalance date, we need to do the following:
# 1. Get the weights for each quantile portfolio (if not already done)
# 2. Calculate the daily returns for each quantile portfolio until the next rebalance date
# 3. Aggregate the daily returns to get the total return for each quantile portfolio over the period
# 4. Store the results in a DataFrame with rebalance dates as rows and quantiles as columns
# Let's assume we already have a multi-indexed DataFrame of factor loadings (date, stock) -> factor value
# From that, we can derive the weights matrix (again multi-indexed)
# Now for each rebalance date, we can obtain the list of stocks in the weights matrix, fetch historical prices
# and calculate returns for those stocks over the period until the next rebalance date.
# Finally, we can use the get_quantile_portfolio_returns function to compute the returns for each quantile portfolio.
# We can loop through each rebalance date to build the final results DataFrame.


def get_quantile_portfolio_returns_df(
    weights_multiindex_df: pd.DataFrame,
    last_date: pd.Timestamp = None,
) -> pd.DataFrame:
    """Calculate quantile portfolio returns over multiple rebalance periods.

    Args:
        weights_multiindex_df (pd.DataFrame): Multi-indexed DataFrame with (date, stock) index and quantile weights as columns.

    Returns:
        pd.DataFrame: DataFrame with rebalance dates as index and quantile portfolio returns as columns.
    """

    rebalance_dates = sorted(weights_multiindex_df.index.get_level_values(0).unique())
    if last_date is not None:
        rebalance_dates.append(last_date)
    else:
        rebalance_dates.append(pd.Timestamp.today().normalize())
    results = []
    for i in range(len(rebalance_dates) - 1):
        start_date = rebalance_dates[i]
        end_date = rebalance_dates[i + 1]

        # Get weights for the current rebalance date
        weights_df = weights_multiindex_df.xs(start_date, level=0)

        # Fetch price data for the stocks in the weights DataFrame
        stocks = weights_df.index.tolist()
        price_df = fetch_historical_price_data(stocks, start_date, end_date, actions=True)

        # Calculate daily returns
        returns_df = generate_returns_df(price_df)

        # Calculate quantile portfolio returns
        quantile_returns = get_quantile_portfolio_returns(weights_df, returns_df)
        quantile_returns.name = end_date  # we set .name to end_date because these are
        # returns attributed to the period ending on end_date
        results.append(quantile_returns)

    return pd.DataFrame(results)
