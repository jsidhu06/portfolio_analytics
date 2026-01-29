"""Helper functions for Jupyter notebooks."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta


def create_sample_option_spec(
    option_type: str = "call",
    strike: float = 100.0,
    maturity_days: int = 30,
    currency: str = "USD",
) -> dict:
    """Create a sample option specification dictionary.

    Parameters
    ----------
    option_type : str, optional
        'call' or 'put' (default: 'call')
    strike : float, optional
        Strike price (default: 100.0)
    maturity_days : int, optional
        Days to maturity (default: 30)
    currency : str, optional
        Currency code (default: 'USD')

    Returns
    -------
    dict
        Option specification dictionary
    """
    from portfolio_analytics.enums import OptionType, ExerciseType
    from portfolio_analytics.valuation.core import OptionSpec

    maturity = datetime.now() + timedelta(days=maturity_days)
    return {
        "option_type": OptionType.CALL if option_type.lower() == "call" else OptionType.PUT,
        "exercise_type": ExerciseType.EUROPEAN,
        "strike": strike,
        "maturity": maturity,
        "currency": currency,
    }


def create_sample_market_data(
    pricing_date: datetime | None = None,
    risk_free_rate: float = 0.05,
    currency: str = "USD",
) -> dict:
    """Create sample market data.

    Parameters
    ----------
    pricing_date : datetime, optional
        Pricing date (default: today)
    risk_free_rate : float, optional
        Risk-free rate (default: 0.05)
    currency : str, optional
        Currency code (default: 'USD')

    Returns
    -------
    dict
        Market data dictionary
    """
    from portfolio_analytics.market_environment import MarketData
    from portfolio_analytics.rates import ConstantShortRate

    if pricing_date is None:
        pricing_date = datetime.now()

    discount_curve = ConstantShortRate("USD", risk_free_rate)
    market_data = MarketData(pricing_date, discount_curve, currency)

    return {"market_data": market_data, "pricing_date": pricing_date}


def plot_comparison(
    x_values: np.ndarray,
    y_dict: dict[str, np.ndarray],
    title: str = "Comparison",
    xlabel: str = "X",
    ylabel: str = "Y",
    figsize: tuple[float, float] = (10, 6),
) -> None:
    """Plot multiple series for comparison.

    Parameters
    ----------
    x_values : np.ndarray
        X-axis values
    y_dict : dict[str, np.ndarray]
        Dictionary mapping label -> y values
    title : str, optional
        Plot title (default: 'Comparison')
    xlabel : str, optional
        X-axis label (default: 'X')
    ylabel : str, optional
        Y-axis label (default: 'Y')
    figsize : tuple[float, float], optional
        Figure size (default: (10, 6))
    """
    fig, ax = plt.subplots(figsize=figsize)
    for label, y_values in y_dict.items():
        ax.plot(x_values, y_values, label=label, linewidth=2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def print_option_summary(valuation) -> None:
    """Print summary of option valuation.

    Parameters
    ----------
    valuation
        OptionValuation object
    """
    print("=" * 50)
    print("Option Valuation Summary")
    print("=" * 50)
    print(f"Option Type: {valuation.option_type.value}")
    print(f"Strike: {valuation.strike:.2f}")
    print(f"Maturity: {valuation.maturity}")
    print(f"Pricing Method: {valuation.pricing_method.value}")
    print(f"\nPresent Value: {valuation.present_value():.4f}")
    print(f"\nGreeks:")
    print(f"  Delta:  {valuation.delta():.4f}")
    print(f"  Gamma:  {valuation.gamma():.6f}")
    print(f"  Vega:   {valuation.vega():.4f}")
    print(f"  Theta:  {valuation.theta():.4f}")
    print(f"  Rho:    {valuation.rho():.4f}")
    print("=" * 50)
