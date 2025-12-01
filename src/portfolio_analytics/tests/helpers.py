import numpy as np
import pandas as pd


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
