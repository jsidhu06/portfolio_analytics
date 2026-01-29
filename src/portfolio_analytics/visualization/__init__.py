"""Visualization module for portfolio analytics.

This module provides plotting functions for:
- Option payoffs and strategies
- Greeks surfaces and dashboards
- Stochastic process paths
- Binomial trees
- Volatility surfaces
"""

from .payoffs import plot_option_payoff, plot_strategy_payoff
from .greeks import plot_delta_surface, plot_greeks_dashboard
from .processes import plot_simulation_paths, plot_terminal_distribution
from .trees import plot_binomial_tree, plot_tree_convergence
from .surfaces import plot_volatility_surface

__all__ = [
    "plot_option_payoff",
    "plot_strategy_payoff",
    "plot_delta_surface",
    "plot_greeks_dashboard",
    "plot_simulation_paths",
    "plot_terminal_distribution",
    "plot_binomial_tree",
    "plot_tree_convergence",
    "plot_volatility_surface",
]
