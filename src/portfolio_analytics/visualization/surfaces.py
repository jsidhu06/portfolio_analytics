"""Plot volatility surfaces."""

from typing import TYPE_CHECKING
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D

if TYPE_CHECKING:
    from ..volatility.surface import VolatilitySurface


def plot_volatility_surface(
    surface: "VolatilitySurface",
    strike_range: tuple[float, float] | None = None,
    expiry_range: tuple[float, float] | None = None,
    num_points: int = 50,
    figsize: tuple[float, float] = (12, 8),
) -> tuple[Figure, Axes3D]:
    """Plot 3D volatility surface.

    Parameters
    ----------
    surface : VolatilitySurface
        Volatility surface object
    strike_range : tuple[float, float], optional
        (min_strike, max_strike) range. If None, uses available strikes.
    expiry_range : tuple[float, float], optional
        (min_expiry, max_expiry) range in years. If None, uses available expiries.
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

    # Get available strikes and expiries from surface
    available_strikes = surface.get_available_strikes()
    available_expiries = surface.get_available_expiries()

    if strike_range is None:
        strike_min = min(available_strikes) if available_strikes else 50.0
        strike_max = max(available_strikes) if available_strikes else 150.0
    else:
        strike_min, strike_max = strike_range

    if expiry_range is None:
        expiry_min = min(available_expiries) if available_expiries else 0.1
        expiry_max = max(available_expiries) if available_expiries else 1.0
    else:
        expiry_min, expiry_max = expiry_range

    strike_grid = np.linspace(strike_min, strike_max, num_points)
    expiry_grid = np.linspace(expiry_min, expiry_max, num_points)
    strike_mesh, expiry_mesh = np.meshgrid(strike_grid, expiry_grid)

    # Calculate volatility for each point
    vol_grid = np.zeros_like(strike_mesh)
    for i in range(num_points):
        for j in range(num_points):
            try:
                vol_grid[i, j] = surface.get_vol(strike_mesh[i, j], expiry_mesh[i, j])
            except (ValueError, KeyError):
                vol_grid[i, j] = np.nan

    # Plot surface
    surf = ax.plot_surface(strike_mesh, expiry_mesh, vol_grid, cmap="coolwarm", alpha=0.8, edgecolor="none")
    ax.set_xlabel("Strike")
    ax.set_ylabel("Expiry (years)")
    ax.set_zlabel("Implied Volatility")
    ax.set_title("Volatility Surface")
    fig.colorbar(surf, ax=ax, shrink=0.5)

    return fig, ax
