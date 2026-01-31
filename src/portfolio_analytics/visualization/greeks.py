"""Plot Greeks surfaces and dashboards."""

from typing import TYPE_CHECKING
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D

if TYPE_CHECKING:
    from ..valuation.core import OptionValuation


def plot_delta_surface(
    valuation: "OptionValuation",
    spot_range: tuple[float, float] | None = None,
    vol_range: tuple[float, float] | None = None,
    num_points: int = 50,
    figsize: tuple[float, float] = (12, 8),
) -> tuple[Figure, Axes3D]:
    """Plot delta surface as function of spot and volatility.

    Parameters
    ----------
    valuation : OptionValuation
        Option valuation object
    spot_range : tuple[float, float], optional
        (min_spot, max_spot) range. If None, uses ±50% around current spot.
    vol_range : tuple[float, float], optional
        (min_vol, max_vol) range. If None, uses ±50% around current vol.
    num_points : int, optional
        Number of points per axis (default: 50)
    figsize : tuple[float, float], optional
        Figure size (default: (12, 8))

    Returns
    -------
    tuple[Figure, Axes3D]
        Matplotlib figure and 3D axes objects
    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    # Determine ranges
    current_spot = valuation.underlying.initial_value
    current_vol = valuation.underlying.volatility

    if spot_range is None:
        spot_min = current_spot * 0.5
        spot_max = current_spot * 1.5
    else:
        spot_min, spot_max = spot_range

    if vol_range is None:
        vol_min = max(current_vol * 0.1, 0.01)
        vol_max = current_vol * 2.0
    else:
        vol_min, vol_max = vol_range

    spot_grid = np.linspace(spot_min, spot_max, num_points)
    vol_grid = np.linspace(vol_min, vol_max, num_points)
    spot_mesh, vol_mesh = np.meshgrid(spot_grid, vol_grid)

    # Calculate delta for each point
    delta_grid = np.zeros_like(spot_mesh)
    for i in range(num_points):
        for j in range(num_points):
            # Create bumped underlying
            bumped_underlying = valuation.underlying.replace(
                initial_value=spot_mesh[i, j], volatility=vol_mesh[i, j]
            )
            bumped_valuation = type(valuation)(
                name=f"{valuation.name}_delta_surface",
                underlying=bumped_underlying,
                spec=valuation.spec,
                pricing_method=valuation.pricing_method,
                params=valuation.params,
            )
            delta_grid[i, j] = bumped_valuation.delta()

    # Plot surface
    surf = ax.plot_surface(spot_mesh, vol_mesh, delta_grid, cmap="viridis", alpha=0.8, edgecolor="none")
    ax.set_xlabel("Spot Price")
    ax.set_ylabel("Volatility")
    ax.set_zlabel("Delta")
    ax.set_title("Delta Surface")
    fig.colorbar(surf, ax=ax, shrink=0.5)

    return fig, ax


def plot_greeks_dashboard(
    valuation: "OptionValuation",
    spot_range: tuple[float, float] | None = None,
    num_points: int = 200,
    figsize: tuple[float, float] = (14, 10),
) -> Figure:
    """Plot dashboard of all Greeks as functions of spot price.

    Parameters
    ----------
    valuation : OptionValuation
        Option valuation object
    spot_range : tuple[float, float], optional
        (min_spot, max_spot) range. If None, uses ±50% around current spot.
    num_points : int, optional
        Number of points to plot (default: 200)
    figsize : tuple[float, float], optional
        Figure size (default: (14, 10))

    Returns
    -------
    Figure
        Matplotlib figure object
    """
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    axes = axes.flatten()

    # Determine spot range
    current_spot = valuation.underlying.initial_value
    if spot_range is None:
        spot_min = current_spot * 0.5
        spot_max = current_spot * 1.5
    else:
        spot_min, spot_max = spot_range

    spot_values = np.linspace(spot_min, spot_max, num_points)

    # Calculate Greeks
    deltas = []
    gammas = []
    vegas = []
    thetas = []
    rhos = []
    prices = []

    for spot in spot_values:
        bumped_underlying = valuation.underlying.replace(initial_value=spot)
        bumped_valuation = type(valuation)(
            name=f"{valuation.name}_greeks",
            underlying=bumped_underlying,
            spec=valuation.spec,
            pricing_method=valuation.pricing_method,
            params=valuation.params,
        )
        deltas.append(bumped_valuation.delta())
        gammas.append(bumped_valuation.gamma())
        vegas.append(bumped_valuation.vega())
        thetas.append(bumped_valuation.theta())
        rhos.append(bumped_valuation.rho())
        prices.append(bumped_valuation.present_value())

    # Plot Delta
    axes[0].plot(spot_values, deltas, linewidth=2)
    axes[0].axvline(x=valuation.strike, color="r", linestyle="--", alpha=0.5)
    axes[0].set_xlabel("Spot Price")
    axes[0].set_ylabel("Delta")
    axes[0].set_title("Delta")
    axes[0].grid(True, alpha=0.3)

    # Plot Gamma
    axes[1].plot(spot_values, gammas, linewidth=2, color="orange")
    axes[1].axvline(x=valuation.strike, color="r", linestyle="--", alpha=0.5)
    axes[1].set_xlabel("Spot Price")
    axes[1].set_ylabel("Gamma")
    axes[1].set_title("Gamma")
    axes[1].grid(True, alpha=0.3)

    # Plot Vega
    axes[2].plot(spot_values, vegas, linewidth=2, color="green")
    axes[2].axvline(x=valuation.strike, color="r", linestyle="--", alpha=0.5)
    axes[2].set_xlabel("Spot Price")
    axes[2].set_ylabel("Vega")
    axes[2].set_title("Vega")
    axes[2].grid(True, alpha=0.3)

    # Plot Theta
    axes[3].plot(spot_values, thetas, linewidth=2, color="purple")
    axes[3].axvline(x=valuation.strike, color="r", linestyle="--", alpha=0.5)
    axes[3].set_xlabel("Spot Price")
    axes[3].set_ylabel("Theta")
    axes[3].set_title("Theta")
    axes[3].grid(True, alpha=0.3)

    # Plot Rho
    axes[4].plot(spot_values, rhos, linewidth=2, color="brown")
    axes[4].axvline(x=valuation.strike, color="r", linestyle="--", alpha=0.5)
    axes[4].set_xlabel("Spot Price")
    axes[4].set_ylabel("Rho")
    axes[4].set_title("Rho")
    axes[4].grid(True, alpha=0.3)

    # Plot Price
    axes[5].plot(spot_values, prices, linewidth=2, color="black")
    axes[5].axvline(x=valuation.strike, color="r", linestyle="--", alpha=0.5)
    axes[5].set_xlabel("Spot Price")
    axes[5].set_ylabel("Option Price")
    axes[5].set_title("Option Price")
    axes[5].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig
