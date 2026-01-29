"""Plot option payoffs and strategy diagrams."""

from typing import TYPE_CHECKING
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes

if TYPE_CHECKING:
    from ..valuation.core import OptionSpec, PayoffSpec


def plot_option_payoff(
    spec: "OptionSpec | PayoffSpec",
    premium: float = 0.0,
    spot_range: tuple[float, float] | None = None,
    num_points: int = 1000,
    figsize: tuple[float, float] = (10, 6),
) -> tuple[Figure, Axes]:
    """Plot option payoff diagram.

    Parameters
    ----------
    spec : OptionSpec | PayoffSpec
        Option specification
    premium : float, optional
        Option premium paid (default: 0.0)
    spot_range : tuple[float, float], optional
        (min_spot, max_spot) range for plotting. If None, uses strike-based range.
    num_points : int, optional
        Number of points to plot (default: 1000)
    figsize : tuple[float, float], optional
        Figure size (default: (10, 6))

    Returns
    -------
    tuple[Figure, Axes]
        Matplotlib figure and axes objects
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Determine spot range
    if spot_range is None:
        if hasattr(spec, "strike") and spec.strike is not None:
            strike = spec.strike
            spot_min = strike * 0.5
            spot_max = strike * 1.5
        else:
            spot_min = 50.0
            spot_max = 150.0
    else:
        spot_min, spot_max = spot_range

    spot_values = np.linspace(spot_min, spot_max, num_points)

    # Calculate payoffs
    if hasattr(spec, "payoff_fn"):
        # PayoffSpec
        payoffs = spec.payoff_fn(spot_values)
    elif spec.option_type.value == "call":
        payoffs = np.maximum(spot_values - spec.strike, 0.0)
    elif spec.option_type.value == "put":
        payoffs = np.maximum(spec.strike - spot_values, 0.0)
    else:
        raise ValueError(f"Unsupported option type: {spec.option_type}")

    # Net payoff (after premium)
    net_payoffs = payoffs - premium

    # Plot
    ax.plot(spot_values, payoffs, label="Gross Payoff", linewidth=2, linestyle="--", alpha=0.7)
    ax.plot(spot_values, net_payoffs, label="Net Payoff", linewidth=2)
    ax.axhline(y=0, color="k", linestyle=":", alpha=0.5)
    ax.axvline(x=spec.strike if hasattr(spec, "strike") and spec.strike else spot_min, color="r", linestyle=":", alpha=0.5, label="Strike")

    ax.set_xlabel("Spot Price")
    ax.set_ylabel("Payoff")
    ax.set_title(f"{spec.option_type.value.upper()} Option Payoff Diagram")
    ax.legend()
    ax.grid(True, alpha=0.3)

    return fig, ax


def plot_strategy_payoff(
    legs: list[tuple["OptionSpec | PayoffSpec", float]],
    spot_range: tuple[float, float] | None = None,
    num_points: int = 1000,
    figsize: tuple[float, float] = (10, 6),
) -> tuple[Figure, Axes]:
    """Plot multi-leg strategy payoff diagram.

    Parameters
    ----------
    legs : list[tuple[OptionSpec | PayoffSpec, float]]
        List of (option_spec, quantity) tuples. Quantity can be positive (long) or negative (short).
    spot_range : tuple[float, float], optional
        (min_spot, max_spot) range for plotting
    num_points : int, optional
        Number of points to plot (default: 1000)
    figsize : tuple[float, float], optional
        Figure size (default: (10, 6))

    Returns
    -------
    tuple[Figure, Axes]
        Matplotlib figure and axes objects
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Determine spot range
    strikes = []
    for leg_spec, _ in legs:
        if hasattr(leg_spec, "strike") and leg_spec.strike is not None:
            strikes.append(leg_spec.strike)

    if spot_range is None:
        if strikes:
            strike_min = min(strikes)
            strike_max = max(strikes)
            spot_min = strike_min * 0.7
            spot_max = strike_max * 1.3
        else:
            spot_min = 50.0
            spot_max = 150.0
    else:
        spot_min, spot_max = spot_range

    spot_values = np.linspace(spot_min, spot_max, num_points)
    total_payoff = np.zeros_like(spot_values)

    # Calculate individual leg payoffs
    leg_payoffs = []
    for leg_spec, quantity in legs:
        if hasattr(leg_spec, "payoff_fn"):
            leg_payoff = leg_spec.payoff_fn(spot_values) * quantity
        elif leg_spec.option_type.value == "call":
            leg_payoff = np.maximum(spot_values - leg_spec.strike, 0.0) * quantity
        elif leg_spec.option_type.value == "put":
            leg_payoff = np.maximum(leg_spec.strike - spot_values, 0.0) * quantity
        else:
            leg_payoff = np.zeros_like(spot_values)

        leg_payoffs.append(leg_payoff)
        total_payoff += leg_payoff

    # Plot individual legs (optional, can be commented out for cleaner plot)
    for i, (leg_spec, quantity) in enumerate(legs):
        ax.plot(
            spot_values,
            leg_payoffs[i],
            label=f"Leg {i+1} ({leg_spec.option_type.value}, qty={quantity})",
            linewidth=1.5,
            alpha=0.5,
            linestyle="--",
        )

    # Plot total strategy payoff
    ax.plot(spot_values, total_payoff, label="Total Strategy", linewidth=3, color="black")

    # Add strike lines
    for leg_spec, _ in legs:
        if hasattr(leg_spec, "strike") and leg_spec.strike is not None:
            ax.axvline(x=leg_spec.strike, color="r", linestyle=":", alpha=0.3)

    ax.axhline(y=0, color="k", linestyle=":", alpha=0.5)
    ax.set_xlabel("Spot Price")
    ax.set_ylabel("Payoff")
    ax.set_title("Strategy Payoff Diagram")
    ax.legend()
    ax.grid(True, alpha=0.3)

    return fig, ax
