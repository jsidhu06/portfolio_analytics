"""PnL attribution using Taylor expansion of option value.

Decomposes option PnL into contributions from different Greeks.
"""

from dataclasses import dataclass
import numpy as np


@dataclass
class PnLAttribution:
    """PnL attribution breakdown.

    Parameters
    ----------
    delta_pnl : float
        PnL from delta (spot movement)
    gamma_pnl : float
        PnL from gamma (convexity)
    theta_pnl : float
        PnL from theta (time decay)
    vega_pnl : float
        PnL from vega (volatility change)
    rho_pnl : float
        PnL from rho (interest rate change)
    unexplained : float
        Unexplained PnL (residual)
    """

    delta_pnl: float
    gamma_pnl: float
    theta_pnl: float
    vega_pnl: float
    rho_pnl: float
    unexplained: float


def attribute_pnl_taylor_expansion(
    initial_price: float,
    final_price: float,
    delta: float,
    gamma: float,
    theta: float,
    vega: float,
    rho: float,
    spot_change: float,
    vol_change: float,
    time_change: float,
    rate_change: float = 0.0,
) -> PnLAttribution:
    """Attribute PnL using Taylor expansion.

    Uses second-order Taylor expansion:
        ΔV ≈ Δ * ΔS + 0.5 * Γ * (ΔS)^2 + Θ * Δt + ν * Δσ + ρ * Δr + ...

    Parameters
    ----------
    initial_price : float
        Initial option price
    final_price : float
        Final option price
    delta : float
        Option delta
    gamma : float
        Option gamma
    theta : float
        Option theta (per day)
    vega : float
        Option vega (per 1% vol change)
    rho : float
        Option rho (per 1% rate change)
    spot_change : float
        Change in spot price
    vol_change : float
        Change in volatility (as decimal, e.g., 0.01 for 1%)
    time_change : float
        Change in time (in days)
    rate_change : float, optional
        Change in interest rate (as decimal, e.g., 0.01 for 1%) (default: 0.0)

    Returns
    -------
    PnLAttribution
        PnL attribution breakdown
    """
    # Calculate PnL components
    delta_pnl = delta * spot_change
    gamma_pnl = 0.5 * gamma * spot_change**2
    theta_pnl = theta * time_change
    vega_pnl = vega * vol_change * 100  # vega is per 1% change
    rho_pnl = rho * rate_change * 100  # rho is per 1% change

    # Total explained PnL
    explained_pnl = delta_pnl + gamma_pnl + theta_pnl + vega_pnl + rho_pnl

    # Actual PnL
    actual_pnl = final_price - initial_price

    # Unexplained (residual)
    unexplained = actual_pnl - explained_pnl

    return PnLAttribution(
        delta_pnl=delta_pnl,
        gamma_pnl=gamma_pnl,
        theta_pnl=theta_pnl,
        vega_pnl=vega_pnl,
        rho_pnl=rho_pnl,
        unexplained=unexplained,
    )
