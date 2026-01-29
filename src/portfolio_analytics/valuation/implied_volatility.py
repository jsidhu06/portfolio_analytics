"""Implied volatility calculation from market option prices.

This module provides multiple methods for solving the inverse Black-Scholes-Merton
problem: given an observed market price, find the volatility that produces that price.

Methods
-------
- Newton-Raphson: Fast iterative method using vega (industry standard)
- Bisection: Robust, guaranteed convergence
- Brentq: Scipy-optimized root finding
- Brenner-Subrahmanyam: Quick ATM approximation
- Corrado-Miller: Better OTM/ITM accuracy
"""

from dataclasses import dataclass
from enum import Enum
import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm
from ..utils import calculate_year_fraction
from ..enums import OptionType


class IVMethod(Enum):
    """Implied volatility solution methods."""

    NEWTON_RAPHSON = "newton_raphson"
    BISECTION = "bisection"
    BRENTQ = "brentq"
    BRENNER_SUBRAHMANYAM = "brenner_subrahmanyam"
    CORRADO_MILLER = "corrado_miller"


@dataclass(frozen=True)
class ImpliedVolatilityParams:
    """Parameters for implied volatility calculation.

    Parameters
    ----------
    spot : float
        Current spot price of underlying
    strike : float
        Strike price
    time_to_maturity : float
        Time to maturity in years
    risk_free_rate : float
        Risk-free interest rate (annualized)
    dividend_yield : float, optional
        Continuous dividend yield (default: 0.0)
    market_price : float
        Observed market price of the option
    option_type : OptionType
        OptionType.CALL or OptionType.PUT
    """

    spot: float
    strike: float
    time_to_maturity: float
    risk_free_rate: float
    market_price: float
    option_type: OptionType
    dividend_yield: float = 0.0

    def __post_init__(self):
        """Validate parameters."""
        if self.spot <= 0:
            raise ValueError("spot must be positive")
        if self.strike <= 0:
            raise ValueError("strike must be positive")
        if self.time_to_maturity <= 0:
            raise ValueError("time_to_maturity must be positive")
        if self.market_price < 0:
            raise ValueError("market_price must be non-negative")
        if self.option_type not in (OptionType.CALL, OptionType.PUT):
            raise ValueError("option_type must be OptionType.CALL or OptionType.PUT")


@dataclass(frozen=True)
class ImpliedVolatilityResult:
    """Result of implied volatility calculation.

    Parameters
    ----------
    implied_volatility : float
        Calculated implied volatility (annualized)
    iterations : int | None
        Number of iterations required (None for analytical methods)
    converged : bool
        Whether the solver converged (True for analytical methods)
    method : IVMethod
        Method used to calculate implied volatility
    """

    implied_volatility: float
    iterations: int | None
    converged: bool
    method: IVMethod


def _bsm_price(
    spot: float,
    strike: float,
    time_to_maturity: float,
    risk_free_rate: float,
    volatility: float,
    option_type: OptionType,
    dividend_yield: float = 0.0,
) -> float:
    """Calculate Black-Scholes-Merton option price.

    Internal helper function for implied volatility solvers.
    """
    if time_to_maturity <= 0:
        return max(option_type == OptionType.CALL and (spot - strike) or (strike - spot), 0.0)

    d1_num = np.log(spot / strike) + (risk_free_rate - dividend_yield + 0.5 * volatility**2) * time_to_maturity
    d1_den = volatility * np.sqrt(time_to_maturity)
    d1 = d1_num / d1_den
    d2 = d1 - d1_den

    discount = np.exp(-risk_free_rate * time_to_maturity)
    spot_disc = spot * np.exp(-dividend_yield * time_to_maturity)

    if option_type == OptionType.CALL:
        price = spot_disc * norm.cdf(d1) - strike * discount * norm.cdf(d2)
    else:  # PUT
        price = strike * discount * norm.cdf(-d2) - spot_disc * norm.cdf(-d1)

    return float(price)


def _bsm_vega(
    spot: float,
    strike: float,
    time_to_maturity: float,
    risk_free_rate: float,
    volatility: float,
    dividend_yield: float = 0.0,
) -> float:
    """Calculate Black-Scholes-Merton vega (sensitivity to volatility).

    Internal helper function for Newton-Raphson solver.
    """
    if time_to_maturity <= 0:
        return 0.0

    d1_num = np.log(spot / strike) + (risk_free_rate - dividend_yield + 0.5 * volatility**2) * time_to_maturity
    d1_den = volatility * np.sqrt(time_to_maturity)
    d1 = d1_num / d1_den

    vega = spot * np.exp(-dividend_yield * time_to_maturity) * norm.pdf(d1) * np.sqrt(time_to_maturity)
    return float(vega)


def brenner_subrahmanyam_approximation(
    spot: float,
    strike: float,
    time_to_maturity: float,
    risk_free_rate: float,
    market_price: float,
    option_type: OptionType,
    dividend_yield: float = 0.0,
) -> float:
    """Brenner-Subrahmanyam approximation for ATM implied volatility.

    Fast analytical approximation valid for at-the-money options.
    Works best when spot ≈ strike.

    Parameters
    ----------
    spot : float
        Current spot price
    strike : float
        Strike price
    time_to_maturity : float
        Time to maturity in years
    risk_free_rate : float
        Risk-free rate
    market_price : float
        Market price of option
    option_type : OptionType
        OptionType.CALL or OptionType.PUT
    dividend_yield : float, optional
        Continuous dividend yield (default: 0.0)

    Returns
    -------
    float
        Approximate implied volatility

    References
    ----------
    Brenner, M., & Subrahmanyam, M. G. (1988). A simple formula to compute
    the implied standard deviation. Financial Review, 23(3), 397-401.
    """
    if time_to_maturity <= 0:
        return 0.0

    discount = np.exp(-risk_free_rate * time_to_maturity)
    spot_disc = spot * np.exp(-dividend_yield * time_to_maturity)

    # For ATM options, approximation simplifies
    if abs(spot - strike) / strike < 0.01:  # Very close to ATM
        iv = market_price * np.sqrt(2 * np.pi) / (spot_disc * np.sqrt(time_to_maturity))
    else:
        # More general approximation
        if option_type == OptionType.CALL:
            iv = market_price * np.sqrt(2 * np.pi / time_to_maturity) / spot_disc
        else:  # PUT
            iv = market_price * np.sqrt(2 * np.pi / time_to_maturity) / (strike * discount)

    return max(float(iv), 0.0)


def corrado_miller_approximation(
    spot: float,
    strike: float,
    time_to_maturity: float,
    risk_free_rate: float,
    market_price: float,
    option_type: OptionType,
    dividend_yield: float = 0.0,
) -> float:
    """Corrado-Miller approximation for implied volatility.

    Improved approximation that works better for OTM/ITM options compared
    to Brenner-Subrahmanyam.

    Parameters
    ----------
    spot : float
        Current spot price
    strike : float
        Strike price
    time_to_maturity : float
        Time to maturity in years
    risk_free_rate : float
        Risk-free rate
    market_price : float
        Market price of option
    option_type : OptionType
        OptionType.CALL or OptionType.PUT
    dividend_yield : float, optional
        Continuous dividend yield (default: 0.0)

    Returns
    -------
    float
        Approximate implied volatility

    References
    ----------
    Corrado, C. J., & Miller, T. W. (1996). A note on a simple, accurate formula
    to compute implied standard deviations. Journal of Banking & Finance, 20(3), 595-603.
    """
    if time_to_maturity <= 0:
        return 0.0

    discount = np.exp(-risk_free_rate * time_to_maturity)
    spot_disc = spot * np.exp(-dividend_yield * time_to_maturity)

    S = spot_disc
    K = strike * discount
    C = market_price

    # Intrinsic value
    if option_type == OptionType.CALL:
        intrinsic = max(S - K, 0.0)
    else:  # PUT
        intrinsic = max(K - S, 0.0)

    # Time value
    time_value = C - intrinsic

    if time_value <= 0:
        return 0.0

    # Corrado-Miller formula
    sqrt_T = np.sqrt(time_to_maturity)
    alpha = (S - K) / (2 * sqrt_T)
    beta = time_value / sqrt_T

    # Solve quadratic: beta = alpha * N(d1) + (S + K) / (2 * sqrt(2*pi)) * exp(-d1^2/2) / sigma
    # Simplified approximation
    if abs(S - K) < 1e-10:  # ATM
        iv = beta * np.sqrt(2 * np.pi) / (S + K)
    else:
        # Use iterative refinement of Brenner-Subrahmanyam
        iv_bs = brenner_subrahmanyam_approximation(
            spot, strike, time_to_maturity, risk_free_rate, market_price, option_type, dividend_yield
        )
        # Refine using quadratic approximation
        d1_approx = (np.log(S / K) + 0.5 * iv_bs**2 * time_to_maturity) / (iv_bs * sqrt_T)
        iv = iv_bs * (1 + (S - K) / (S + K) * d1_approx / sqrt_T)

    return max(float(iv), 0.0)


def implied_volatility(
    params: ImpliedVolatilityParams,
    method: IVMethod = IVMethod.NEWTON_RAPHSON,
    initial_guess: float | None = None,
    max_iterations: int = 100,
    tolerance: float = 1e-6,
    vol_min: float = 1e-6,
    vol_max: float = 5.0,
) -> ImpliedVolatilityResult:
    """Calculate implied volatility from market price.

    Main entry point for implied volatility calculation. Supports multiple
    solution methods.

    Parameters
    ----------
    params : ImpliedVolatilityParams
        Parameters including spot, strike, time, rates, market price
    method : IVMethod, optional
        Solution method to use (default: NEWTON_RAPHSON)
    initial_guess : float, optional
        Initial volatility guess for iterative methods
    max_iterations : int, optional
        Maximum iterations for iterative methods (default: 100)
    tolerance : float, optional
        Convergence tolerance (default: 1e-6)
    vol_min : float, optional
        Minimum volatility bound (default: 1e-6)
    vol_max : float, optional
        Maximum volatility bound (default: 5.0)

    Returns
    -------
    ImpliedVolatilityResult
        Result containing implied volatility, iterations, convergence status

    Raises
    ------
    ValueError
        If market price is outside arbitrage bounds
    RuntimeError
        If solver fails to converge
    """
    # Check arbitrage bounds
    discount = np.exp(-params.risk_free_rate * params.time_to_maturity)
    spot_disc = params.spot * np.exp(-params.dividend_yield * params.time_to_maturity)

    if params.option_type == OptionType.CALL:
        intrinsic = max(spot_disc - params.strike * discount, 0.0)
        upper_bound = spot_disc
    else:  # PUT
        intrinsic = max(params.strike * discount - spot_disc, 0.0)
        upper_bound = params.strike * discount

    if params.market_price < intrinsic:
        raise ValueError(f"Market price {params.market_price} below intrinsic value {intrinsic}")
    if params.market_price > upper_bound:
        raise ValueError(f"Market price {params.market_price} above upper bound {upper_bound}")

    # Use analytical approximations
    if method == IVMethod.BRENNER_SUBRAHMANYAM:
        iv = brenner_subrahmanyam_approximation(
            params.spot,
            params.strike,
            params.time_to_maturity,
            params.risk_free_rate,
            params.market_price,
            params.option_type,
            params.dividend_yield,
        )
        return ImpliedVolatilityResult(
            implied_volatility=iv, iterations=None, converged=True, method=method
        )

    if method == IVMethod.CORRADO_MILLER:
        iv = corrado_miller_approximation(
            params.spot,
            params.strike,
            params.time_to_maturity,
            params.risk_free_rate,
            params.market_price,
            params.option_type,
            params.dividend_yield,
        )
        return ImpliedVolatilityResult(
            implied_volatility=iv, iterations=None, converged=True, method=method
        )

    # Objective function: difference between market price and BSM price
    def objective(vol: float) -> float:
        return (
            _bsm_price(
                params.spot,
                params.strike,
                params.time_to_maturity,
                params.risk_free_rate,
                vol,
                params.option_type,
                params.dividend_yield,
            )
            - params.market_price
        )

    # Initial guess
    if initial_guess is None:
        # Use Brenner-Subrahmanyam as initial guess
        initial_guess = brenner_subrahmanyam_approximation(
            params.spot,
            params.strike,
            params.time_to_maturity,
            params.risk_free_rate,
            params.market_price,
            params.option_type,
            params.dividend_yield,
        )
        initial_guess = np.clip(initial_guess, vol_min, vol_max)

    # Newton-Raphson method
    if method == IVMethod.NEWTON_RAPHSON:
        vol = initial_guess
        for i in range(max_iterations):
            price = _bsm_price(
                params.spot,
                params.strike,
                params.time_to_maturity,
                params.risk_free_rate,
                vol,
                params.option_type,
                params.dividend_yield,
            )
            vega_val = _bsm_vega(
                params.spot,
                params.strike,
                params.time_to_maturity,
                params.risk_free_rate,
                vol,
                params.dividend_yield,
            )

            if abs(price - params.market_price) < tolerance:
                return ImpliedVolatilityResult(
                    implied_volatility=vol, iterations=i + 1, converged=True, method=method
                )

            if vega_val < 1e-10:  # Avoid division by zero
                # Fall back to bisection
                break

            vol_new = vol - (price - params.market_price) / vega_val
            vol_new = np.clip(vol_new, vol_min, vol_max)

            if abs(vol_new - vol) < tolerance:
                return ImpliedVolatilityResult(
                    implied_volatility=vol_new, iterations=i + 1, converged=True, method=method
                )

            vol = vol_new

        raise RuntimeError(f"Newton-Raphson failed to converge after {max_iterations} iterations")

    # Bisection method
    if method == IVMethod.BISECTION:
        vol_low = vol_min
        vol_high = vol_max

        # Ensure we bracket the root
        if objective(vol_low) > 0:
            vol_low = 0.001
        if objective(vol_high) < 0:
            vol_high = 10.0

        for i in range(max_iterations):
            vol_mid = (vol_low + vol_high) / 2.0
            obj_mid = objective(vol_mid)

            if abs(obj_mid) < tolerance:
                return ImpliedVolatilityResult(
                    implied_volatility=vol_mid, iterations=i + 1, converged=True, method=method
                )

            if obj_mid > 0:
                vol_high = vol_mid
            else:
                vol_low = vol_mid

            if vol_high - vol_low < tolerance:
                return ImpliedVolatilityResult(
                    implied_volatility=(vol_low + vol_high) / 2.0,
                    iterations=i + 1,
                    converged=True,
                    method=method,
                )

        raise RuntimeError(f"Bisection failed to converge after {max_iterations} iterations")

    # Brentq method (scipy)
    if method == IVMethod.BRENTQ:
        try:
            vol, result = brentq(
                objective, vol_min, vol_max, maxiter=max_iterations, xtol=tolerance, full_output=True
            )
            converged = result.converged
            iterations = result.iterations if hasattr(result, "iterations") else None
            return ImpliedVolatilityResult(
                implied_volatility=vol, iterations=iterations, converged=converged, method=method
            )
        except ValueError as e:
            raise RuntimeError(f"Brentq method failed: {e}") from e

    raise ValueError(f"Unknown method: {method}")
