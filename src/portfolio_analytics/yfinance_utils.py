from typing import Dict, List, Union
import logging
import datetime as dt
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


def fetch_historical_price_data(
    tickers: List[str],
    start_date: Union[str, dt.date, dt.datetime, pd.Timestamp],
    end_date: Union[str, dt.date, dt.datetime, pd.Timestamp],
    interval="1d",
    actions: bool = False,
) -> pd.DataFrame:
    "Fetch historical price data from yfinance"
    # EOD data only for now - not implementing intraday atm

    return yf.download(
        tickers,
        start_date,
        end_date,
        interval=interval,
        actions=actions,
        group_by="ticker",
        auto_adjust=True,
        threads=True,
    )


def fetch_latest_price_data(tickers: List[str]) -> Dict[str, Dict[pd.Timestamp, float]]:
    """
    Fetch latest price data from yfinance

    Returns a dictionary mapping each ticker to a dict that maps the stock price timestamp
    to the stock price
    """

    latest_prices_dict = {}
    ticker_data = yf.Tickers(tickers)

    for ticker, single_stock_data in ticker_data.tickers.items():
        latest_date_timestamp = single_stock_data.get_info().get("regularMarketTime")
        if latest_date_timestamp is None:
            logger.warning("No timestamp data for %s. Skipping.", ticker)
            continue

        latest_date = dt.datetime.fromtimestamp(latest_date_timestamp).date()
        if hasattr(single_stock_data, "fast_info") and hasattr(
            single_stock_data.fast_info, "last_price"
        ):
            latest_prices_dict.update(
                {ticker: {latest_date: single_stock_data.fast_info.last_price}}
            )
        else:
            logger.warning("No price data for %s. Skipping.", ticker)
            continue

    return latest_prices_dict
