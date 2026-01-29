"""Plot binomial trees and convergence analysis."""

from typing import TYPE_CHECKING
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes

if TYPE_CHECKING:
    from ..valuation.core import OptionValuation


def plot_binomial_tree(
    valuation: "OptionValuation",
    max_steps: int = 5,
    figsize: tuple[float, float] = (12, 8),
) -> tuple[Figure, Axes]:
    """Plot binomial tree structure (for small trees only).

    Parameters
    ----------
    valuation : OptionValuation
        Option valuation object (must use BINOMIAL pricing method)
    max_steps : int, optional
        Maximum number of steps to display (default: 5). Trees with more steps
        will be truncated for visualization.
    figsize : tuple[float, float], optional
        Figure size (default: (12, 8))

    Returns
    -------
    tuple[Figure, Axes]
        Matplotlib figure and axes objects

    Notes
    -----
    This function is intended for educational purposes and works best with
    small trees (<= 5 steps). For larger trees, use plot_tree_convergence instead.
    """
    if valuation.pricing_method.value != "binomial":
        raise ValueError("plot_binomial_tree requires BINOMIAL pricing method")

    fig, ax = plt.subplots(figsize=figsize)

    # Get tree data from solver
    # Note: This requires accessing internal tree structure
    # For now, we'll create a simplified visualization
    # In a full implementation, you'd extract the actual tree from the solver

    # Simplified tree visualization
    num_steps = min(max_steps, getattr(valuation.params, "num_steps", 5) if valuation.params else 5)

    # Calculate tree structure
    spot = valuation.underlying.initial_value
    strike = valuation.strike
    time_to_maturity = (
        valuation.maturity - valuation.pricing_date
    ).total_seconds() / (365 * 24 * 3600)

    # Simplified up/down factors (Cox-Ross-Rubinstein)
    dt = time_to_maturity / num_steps
    vol = valuation.underlying.volatility
    r = valuation.discount_curve.short_rate
    u = np.exp(vol * np.sqrt(dt))
    d = 1 / u
    p = (np.exp(r * dt) - d) / (u - d)

    # Plot nodes
    y_positions = []
    x_positions = []
    values = []

    for step in range(num_steps + 1):
        for node in range(step + 1):
            x = step
            y = step - 2 * node  # Center around 0
            spot_val = spot * (u ** (step - node)) * (d ** node)
            x_positions.append(x)
            y_positions.append(y)
            values.append(spot_val)

    # Plot connections
    for step in range(num_steps):
        for node in range(step + 1):
            idx = sum(range(step + 1)) + node
            # Up branch
            up_idx = sum(range(step + 2)) + node
            ax.plot([x_positions[idx], x_positions[up_idx]], [y_positions[idx], y_positions[up_idx]], "b-", alpha=0.5)
            # Down branch
            down_idx = sum(range(step + 2)) + node + 1
            ax.plot([x_positions[idx], x_positions[down_idx]], [y_positions[idx], y_positions[down_idx]], "b-", alpha=0.5)

    # Plot nodes
    ax.scatter(x_positions, y_positions, s=100, c=values, cmap="viridis", edgecolors="black", linewidths=1)

    # Add value labels
    for i, (x, y, val) in enumerate(zip(x_positions, y_positions, values)):
        ax.text(x, y + 0.3, f"{val:.2f}", ha="center", fontsize=8)

    ax.set_xlabel("Time Step")
    ax.set_ylabel("Node Position")
    ax.set_title(f"Binomial Tree (Simplified, {num_steps} steps)")
    ax.grid(True, alpha=0.3)

    return fig, ax


def plot_tree_convergence(
    valuation: "OptionValuation",
    step_range: tuple[int, int] = (10, 200),
    num_points: int = 20,
    reference_price: float | None = None,
    figsize: tuple[float, float] = (10, 6),
) -> tuple[Figure, Axes]:
    """Plot option price convergence as function of binomial tree steps.

    Parameters
    ----------
    valuation : OptionValuation
        Option valuation object (must use BINOMIAL pricing method)
    step_range : tuple[int, int], optional
        (min_steps, max_steps) range for convergence analysis (default: (10, 200))
    num_points : int, optional
        Number of step values to test (default: 20)
    reference_price : float, optional
        Reference price (e.g., BSM analytical) to plot for comparison
    figsize : tuple[float, float], optional
        Figure size (default: (10, 6))

    Returns
    -------
    tuple[Figure, Axes]
        Matplotlib figure and axes objects
    """
    if valuation.pricing_method.value != "binomial":
        raise ValueError("plot_tree_convergence requires BINOMIAL pricing method")

    fig, ax = plt.subplots(figsize=figsize)

    min_steps, max_steps = step_range
    step_values = np.linspace(min_steps, max_steps, num_points, dtype=int)
    prices = []

    for num_steps in step_values:
        # Create new params with updated num_steps
        from ..valuation.params import BinomialParams

        new_params = BinomialParams(num_steps=num_steps)
        price = valuation.present_value(params=new_params)
        prices.append(price)

    ax.plot(step_values, prices, marker="o", linewidth=2, label="Binomial Tree Price")

    if reference_price is not None:
        ax.axhline(y=reference_price, color="r", linestyle="--", linewidth=2, label=f"Reference: {reference_price:.4f}")

    ax.set_xlabel("Number of Steps")
    ax.set_ylabel("Option Price")
    ax.set_title("Binomial Tree Convergence")
    ax.legend()
    ax.grid(True, alpha=0.3)

    return fig, ax
