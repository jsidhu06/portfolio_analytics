import pandas as pd


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
            for stock in stocks_in_bucket:
                weights_df.loc[stock, bucket] = market_caps.loc[stock] / total_market_cap

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
