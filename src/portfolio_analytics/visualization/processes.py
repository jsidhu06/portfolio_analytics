"""Plot stochastic process paths and distributions."""

from typing import TYPE_CHECKING
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes

if TYPE_CHECKING:
    from ..stochastic_processes import PathSimulation


def plot_simulation_paths(
    process: "PathSimulation",
    random_seed: int | None = None,
    num_paths_to_plot: int = 50,
    figsize: tuple[float, float] = (12, 6),
) -> tuple[Figure, Axes]:
    """Plot sample paths from stochastic process simulation.

    Parameters
    ----------
    process : PathSimulation
        Stochastic process simulation object
    random_seed : int, optional
        Random seed for path generation
    num_paths_to_plot : int, optional
        Number of paths to display (default: 50)
    figsize : tuple[float, float], optional
        Figure size (default: (12, 6))

    Returns
    -------
    tuple[Figure, Axes]
        Matplotlib figure and axes objects
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Generate paths
    paths = process.get_instrument_values(random_seed=random_seed)

    # Generate time grid if needed
    if process.time_grid is None:
        process.generate_time_grid()
    time_grid = process.time_grid

    # Plot paths
    num_paths = min(num_paths_to_plot, paths.shape[1])
    for i in range(num_paths):
        ax.plot(time_grid, paths[:, i], alpha=0.3, linewidth=0.5)

    # Plot mean path
    mean_path = np.mean(paths, axis=1)
    ax.plot(time_grid, mean_path, color="red", linewidth=2, label="Mean Path")

    # Plot initial value
    ax.axhline(y=process.initial_value, color="green", linestyle="--", linewidth=1, label="Initial Value")

    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    ax.set_title(f"{process.name} - Simulated Paths")
    ax.legend()
    ax.grid(True, alpha=0.3)

    return fig, ax


def plot_terminal_distribution(
    process: "PathSimulation",
    random_seed: int | None = None,
    num_bins: int = 50,
    figsize: tuple[float, float] = (10, 6),
) -> tuple[Figure, Axes]:
    """Plot terminal distribution of simulated paths.

    Parameters
    ----------
    process : PathSimulation
        Stochastic process simulation object
    random_seed : int, optional
        Random seed for path generation
    num_bins : int, optional
        Number of histogram bins (default: 50)
    figsize : tuple[float, float], optional
        Figure size (default: (10, 6))

    Returns
    -------
    tuple[Figure, Axes]
        Matplotlib figure and axes objects
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Generate paths
    paths = process.get_instrument_values(random_seed=random_seed)

    # Terminal values (last time step)
    terminal_values = paths[-1, :]

    # Plot histogram
    ax.hist(terminal_values, bins=num_bins, density=True, alpha=0.7, edgecolor="black")

    # Add statistics
    mean_val = np.mean(terminal_values)
    std_val = np.std(terminal_values)
    median_val = np.median(terminal_values)

    ax.axvline(x=mean_val, color="red", linestyle="--", linewidth=2, label=f"Mean: {mean_val:.2f}")
    ax.axvline(x=median_val, color="blue", linestyle="--", linewidth=2, label=f"Median: {median_val:.2f}")

    # Add normal distribution overlay (if applicable)
    if hasattr(process, "volatility"):
        from scipy.stats import norm

        x_range = np.linspace(terminal_values.min(), terminal_values.max(), 200)
        # Approximate log-normal distribution
        log_mean = np.log(mean_val) - 0.5 * np.log(1 + (std_val / mean_val) ** 2)
        log_std = np.sqrt(np.log(1 + (std_val / mean_val) ** 2))
        normal_pdf = norm.pdf(np.log(x_range), log_mean, log_std) / x_range
        ax.plot(x_range, normal_pdf, color="green", linewidth=2, label="Log-Normal Fit", alpha=0.7)

    ax.set_xlabel("Terminal Value")
    ax.set_ylabel("Density")
    ax.set_title(f"{process.name} - Terminal Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)

    return fig, ax
