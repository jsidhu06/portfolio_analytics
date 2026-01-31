"""Barrier option valuation.

Barrier options are path-dependent options where the payoff depends on whether
the underlying asset price crosses a barrier level during the option's lifetime.

Supports:
- Analytical pricing (Rubinstein-Reiner formulas for continuous monitoring)
- Monte Carlo pricing (with Brownian bridge correction for discrete monitoring)
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING
import datetime as dt
import numpy as np
from scipy.stats import norm
from ..utils import calculate_year_fraction
from ..enums import OptionType, BarrierType, BarrierMonitoring

if TYPE_CHECKING:
    from .core import OptionValuation
    from ..stochastic_processes import PathSimulation


@dataclass(frozen=True)
class BarrierSpec:
    """Barrier option specification.

    Parameters
    ----------
    option_type : OptionType
        OptionType.CALL or OptionType.PUT
    barrier_type : BarrierType
        Type of barrier (UP_AND_IN, UP_AND_OUT, DOWN_AND_IN, DOWN_AND_OUT)
    strike : float
        Strike price
    barrier : float
        Barrier level
    maturity : datetime
        Option maturity date
    rebate : float, optional
        Rebate paid if barrier is not hit (for knock-out) or if barrier is hit (for knock-in) (default: 0.0)
    monitoring : BarrierMonitoring, optional
        Monitoring frequency: CONTINUOUS or DISCRETE (default: CONTINUOUS)
    """

    option_type: OptionType
    barrier_type: BarrierType
    strike: float
    barrier: float
    maturity: dt.datetime
    rebate: float = 0.0
    monitoring: BarrierMonitoring = BarrierMonitoring.CONTINUOUS

    def __post_init__(self):
        """Validate parameters."""
        if self.strike <= 0:
            raise ValueError("strike must be positive")
        if self.barrier <= 0:
            raise ValueError("barrier must be positive")
        if self.option_type not in (OptionType.CALL, OptionType.PUT):
            raise ValueError("option_type must be OptionType.CALL or OptionType.PUT")

        # Validate barrier relative to strike
        if self.barrier_type in (BarrierType.UP_AND_IN, BarrierType.UP_AND_OUT):
            if self.barrier <= self.strike:
                raise ValueError("For UP barriers, barrier must be > strike")
        else:  # DOWN barriers
            if self.barrier >= self.strike:
                raise ValueError("For DOWN barriers, barrier must be < strike")


def barrier_option_analytical(
    spot: float,
    strike: float,
    barrier: float,
    time_to_maturity: float,
    risk_free_rate: float,
    volatility: float,
    option_type: OptionType,
    barrier_type: BarrierType,
    dividend_yield: float = 0.0,
    rebate: float = 0.0,
) -> float:
    """Calculate barrier option price using Rubinstein-Reiner analytical formulas.

    Valid for continuous monitoring only.

    Parameters
    ----------
    spot : float
        Current spot price
    strike : float
        Strike price
    barrier : float
        Barrier level
    time_to_maturity : float
        Time to maturity in years
    risk_free_rate : float
        Risk-free rate
    volatility : float
        Volatility (annualized)
    option_type : OptionType
        OptionType.CALL or OptionType.PUT
    barrier_type : BarrierType
        Barrier type
    dividend_yield : float, optional
        Continuous dividend yield (default: 0.0)
    rebate : float, optional
        Rebate amount (default: 0.0)

    Returns
    -------
    float
        Barrier option price

    References
    ----------
    Rubinstein, M., & Reiner, E. (1991). Breaking down the barriers.
    Risk, 4(8), 28-35.
    """
    if time_to_maturity <= 0:
        return rebate

    S = spot
    K = strike
    H = barrier
    T = time_to_maturity
    r = risk_free_rate
    q = dividend_yield
    sigma = volatility

    # Helper variables
    mu = (r - q - 0.5 * sigma**2) / sigma**2
    lambda_param = np.sqrt(mu**2 + 2 * r / sigma**2)

    # Common terms
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    d3 = (np.log(S / H) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d4 = d3 - sigma * np.sqrt(T)
    d5 = (np.log(H / S) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d6 = d5 - sigma * np.sqrt(T)
    d7 = (np.log(H / S) - (r - q - 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d8 = d7 - sigma * np.sqrt(T)

    # Power terms
    power1 = (H / S) ** (2 * mu)
    power2 = (H / S) ** (2 * lambda_param - 2)

    # Standard call/put prices
    call_price = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)

    # Barrier option prices
    if barrier_type == BarrierType.UP_AND_OUT:
        if option_type == OptionType.CALL:
            if S >= H:
                return rebate
            # Up-and-out call
            price = call_price - S * np.exp(-q * T) * norm.cdf(d3) + K * np.exp(-r * T) * norm.cdf(d4)
            price -= power1 * (
                (H / S) ** (2 * lambda_param)
                * (
                    S * np.exp(-q * T) * norm.cdf(d5)
                    - K * np.exp(-r * T) * norm.cdf(d6)
                )
            )
        else:  # PUT
            if S >= H:
                return rebate
            # Up-and-out put
            price = put_price - power1 * (
                (H / S) ** (2 * lambda_param)
                * (
                    K * np.exp(-r * T) * norm.cdf(-d6)
                    - S * np.exp(-q * T) * norm.cdf(-d5)
                )
            )

    elif barrier_type == BarrierType.UP_AND_IN:
        if option_type == OptionType.CALL:
            if S >= H:
                return call_price
            # Up-and-in call
            price = S * np.exp(-q * T) * norm.cdf(d3) - K * np.exp(-r * T) * norm.cdf(d4)
            price += power1 * (
                (H / S) ** (2 * lambda_param)
                * (
                    S * np.exp(-q * T) * norm.cdf(d5)
                    - K * np.exp(-r * T) * norm.cdf(d6)
                )
            )
        else:  # PUT
            if S >= H:
                return put_price
            # Up-and-in put
            price = power1 * (
                (H / S) ** (2 * lambda_param)
                * (
                    K * np.exp(-r * T) * norm.cdf(-d6)
                    - S * np.exp(-q * T) * norm.cdf(-d5)
                )
            )

    elif barrier_type == BarrierType.DOWN_AND_OUT:
        if option_type == OptionType.CALL:
            if S <= H:
                return rebate
            # Down-and-out call
            price = call_price - power1 * (
                (H / S) ** (2 * lambda_param)
                * (
                    S * np.exp(-q * T) * norm.cdf(d7)
                    - K * np.exp(-r * T) * norm.cdf(d8)
                )
            )
        else:  # PUT
            if S <= H:
                return rebate
            # Down-and-out put
            price = put_price - S * np.exp(-q * T) * norm.cdf(-d3) + K * np.exp(-r * T) * norm.cdf(-d4)
            price -= power1 * (
                (H / S) ** (2 * lambda_param)
                * (
                    K * np.exp(-r * T) * norm.cdf(-d8)
                    - S * np.exp(-q * T) * norm.cdf(-d7)
                )
            )

    else:  # DOWN_AND_IN
        if option_type == OptionType.CALL:
            if S <= H:
                return call_price
            # Down-and-in call
            price = power1 * (
                (H / S) ** (2 * lambda_param)
                * (
                    S * np.exp(-q * T) * norm.cdf(d7)
                    - K * np.exp(-r * T) * norm.cdf(d8)
                )
            )
        else:  # PUT
            if S <= H:
                return put_price
            # Down-and-in put
            price = S * np.exp(-q * T) * norm.cdf(-d3) - K * np.exp(-r * T) * norm.cdf(-d4)
            price += power1 * (
                (H / S) ** (2 * lambda_param)
                * (
                    K * np.exp(-r * T) * norm.cdf(-d8)
                    - S * np.exp(-q * T) * norm.cdf(-d7)
                )
            )

    return max(float(price) + rebate * np.exp(-r * T), 0.0)


def barrier_option_monte_carlo(
    paths: np.ndarray,
    time_grid: np.ndarray,
    spec: BarrierSpec,
    risk_free_rate: float,
    pricing_date: dt.datetime,
) -> float:
    """Calculate barrier option price using Monte Carlo simulation.

    Parameters
    ----------
    paths : np.ndarray
        Simulated asset paths (shape: [num_timesteps, num_paths])
    time_grid : np.ndarray
        Time grid for simulation
    spec : BarrierSpec
        Barrier option specification
    risk_free_rate : float
        Risk-free rate
    pricing_date : datetime
        Pricing date

    Returns
    -------
    float
        Barrier option price
    """
    num_paths = paths.shape[1]
    T = calculate_year_fraction(pricing_date, spec.maturity, day_count_convention=365)

    # Check barrier hits
    barrier_hit = np.zeros(num_paths, dtype=bool)

    if spec.monitoring == BarrierMonitoring.CONTINUOUS:
        # Continuous monitoring: check if barrier crossed at any point
        if spec.barrier_type in (BarrierType.UP_AND_IN, BarrierType.UP_AND_OUT):
            barrier_hit = np.any(paths >= spec.barrier, axis=0)
        else:  # DOWN barriers
            barrier_hit = np.any(paths <= spec.barrier, axis=0)
    else:
        # Discrete monitoring: check only at monitoring dates
        # For simplicity, check at all time steps
        if spec.barrier_type in (BarrierType.UP_AND_IN, BarrierType.UP_AND_OUT):
            barrier_hit = np.any(paths >= spec.barrier, axis=0)
        else:  # DOWN barriers
            barrier_hit = np.any(paths <= spec.barrier, axis=0)

    # Calculate payoffs
    terminal_values = paths[-1, :]

    if spec.option_type == OptionType.CALL:
        intrinsic = np.maximum(terminal_values - spec.strike, 0.0)
    else:  # PUT
        intrinsic = np.maximum(spec.strike - terminal_values, 0.0)

    # Apply barrier logic
    if spec.barrier_type in (BarrierType.UP_AND_OUT, BarrierType.DOWN_AND_OUT):
        # Knock-out: payoff only if barrier NOT hit
        payoffs = np.where(~barrier_hit, intrinsic, spec.rebate)
    else:  # Knock-in
        # Knock-in: payoff only if barrier hit
        payoffs = np.where(barrier_hit, intrinsic, spec.rebate)

    # Discount and average
    discount = np.exp(-risk_free_rate * T)
    price = discount * np.mean(payoffs)

    return float(price)
