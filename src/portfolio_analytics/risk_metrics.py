from typing import Union
import numpy as np
import pandas as pd


def get_annualization_dict() -> dict:
    return {
        "daily": 252,
        "weekly": 52,
        "monthly": 12,
        "quarterly": 4,
        "yearly": 1,
    }


def calculate_portfolio_vol(returns_series: pd.Series, frequency="daily"):
    """Calculate portfolio volatility. This is the annualized standard deviation of returns

    Handled frequencies are daily, weekly, monthly, quarterly and yearly

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
):
    """
    Calculate sharpe ratio
    Sharpe  = (Mean portfolio return  -  risk free rate) / portfolio volatility"
    """

    portfolio_vol = calculate_portfolio_vol(returns_series, returns_frequency)
    annualization_dict = get_annualization_dict()
    annualization_factor = annualization_dict.get(returns_frequency)
    if annualization_factor is None:
        raise ValueError(
            "frequency param must be one of 'daily','weekly','monthly','quarterly' or 'yearly'"
        )

    return (returns_series.mean() * annualization_factor - risk_free_rate) / portfolio_vol
