import pandas as pd


def calculate_total_return_index_share_count_ts(df):
    if {"Close", "Dividends"}.difference(df.columns):
        raise ValueError("'Close' and 'Dividends' columns must be present in the dataframe")

    share_count_ts = []
    prior_share_count = 1

    for _, row in df.iterrows():
        num_shares = prior_share_count * (1 + row["Dividends"] / row["Close"])  # (1+D_t/P_t)
        share_count_ts.append(num_shares)
        prior_share_count = num_shares

    assert len(share_count_ts) == len(df), (
        "length of share count time series does not match "
        "length of Close and Dividend time series"
    )

    return pd.Series(share_count_ts, index=df.index)


def calculate_total_return_index_ts(price_ts: pd.Series, share_count_ts: pd.Series) -> pd.Series:
    assert len(price_ts) == len(
        share_count_ts
    ), "Price time series is of different length to share count time series"
    return price_ts * share_count_ts


def calculate_daily_total_return_gross_dividends_ts(
    price_series: pd.Series, dividend_series: pd.Series
):
    assert (
        len(price_series) == len(dividend_series)
        and (price_series.index == dividend_series.index).all()
    ), "price and dividend series are misaligned"

    return_series = (
        ((price_series.ffill() + dividend_series.fillna(0)) / price_series.ffill().shift(1))
        .sub(1)
        .mul(100)
        .rename("return_series")
    )

    return return_series
