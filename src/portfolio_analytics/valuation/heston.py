"""Heston stochastic volatility model option valuation.

This module provides option pricing using the Heston (1993) stochastic volatility model.
Uses characteristic function approach with FFT for efficient pricing.

References
----------
Heston, S. L. (1993). A closed-form solution for options with stochastic volatility
with applications to bond and currency options. The Review of Financial Studies, 6(2), 327-343.
"""

from typing import TYPE_CHECKING
import numpy as np
from scipy.integrate import quad
from scipy.fft import fft
from ..utils import calculate_year_fraction
from ..enums import OptionType
from ..stochastic_processes import HestonParams

if TYPE_CHECKING:
    from .core import OptionValuation


def heston_characteristic_function(
    u: complex,
    spot: float,
    strike: float,
    time_to_maturity: float,
    risk_free_rate: float,
    params: HestonParams,
) -> complex:
    """Calculate Heston characteristic function.

    Parameters
    ----------
    u : complex
        Fourier transform variable
    spot : float
        Current spot price
    strike : float
        Strike price
    time_to_maturity : float
        Time to maturity in years
    risk_free_rate : float
        Risk-free rate
    params : HestonParams
        Heston model parameters

    Returns
    -------
    complex
        Characteristic function value
    """
    v0 = params.v0
    kappa = params.kappa
    theta = params.theta
    xi = params.xi
    rho = params.rho
    q = params.dividend_yield

    # Heston characteristic function
    d = np.sqrt((rho * xi * 1j * u - kappa) ** 2 + xi**2 * (1j * u + u**2))
    g = (kappa - rho * xi * 1j * u - d) / (kappa - rho * xi * 1j * u + d)
    C = (1 - np.exp(-d * time_to_maturity)) / (xi**2 * (1 - g * np.exp(-d * time_to_maturity)))
    D = (kappa - rho * xi * 1j * u - d) / xi**2 * C

    # Characteristic function
    cf = np.exp(
        1j * u * (np.log(spot) + (risk_free_rate - q) * time_to_maturity)
        + kappa * theta * D
        + v0 * C
    )

    return cf


def heston_call_price(
    spot: float,
    strike: float,
    time_to_maturity: float,
    risk_free_rate: float,
    params: HestonParams,
    integration_limit: float = 100.0,
    num_points: int = 1000,
) -> float:
    """Calculate Heston European call option price using characteristic function.

    Uses the Carr-Madan FFT approach for efficient pricing.

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
    params : HestonParams
        Heston model parameters
    integration_limit : float, optional
        Upper limit for integration (default: 100.0)
    num_points : int, optional
        Number of integration points (default: 1000)

    Returns
    -------
    float
        Option price
    """
    if time_to_maturity <= 0:
        return max(spot - strike, 0.0)

    # Carr-Madan FFT approach
    # Price = S0 * P1 - K * exp(-r*T) * P2
    # where P1 and P2 are probabilities calculated via characteristic function

    def integrand1(u: float) -> float:
        """Integrand for P1."""
        cf = heston_characteristic_function(
            u - 1j, spot, strike, time_to_maturity, risk_free_rate, params
        )
        numerator = np.exp(-1j * u * np.log(strike)) * cf
        denominator = 1j * u * spot * np.exp((risk_free_rate - params.dividend_yield) * time_to_maturity)
        return np.real(numerator / denominator)

    def integrand2(u: float) -> float:
        """Integrand for P2."""
        cf = heston_characteristic_function(u, spot, strike, time_to_maturity, risk_free_rate, params)
        numerator = np.exp(-1j * u * np.log(strike)) * cf
        denominator = 1j * u
        return np.real(numerator / denominator)

    # Numerical integration
    u_grid = np.linspace(1e-6, integration_limit, num_points)
    du = u_grid[1] - u_grid[0]

    # P1
    integrand1_vals = [integrand1(u) for u in u_grid]
    P1 = 0.5 + (1 / np.pi) * np.trapz(integrand1_vals, dx=du)

    # P2
    integrand2_vals = [integrand2(u) for u in u_grid]
    P2 = 0.5 + (1 / np.pi) * np.trapz(integrand2_vals, dx=du)

    # Option price
    discount = np.exp(-risk_free_rate * time_to_maturity)
    spot_disc = spot * np.exp(-params.dividend_yield * time_to_maturity)
    price = spot_disc * P1 - strike * discount * P2

    return max(float(price), 0.0)


def heston_implied_vol(
    spot: float,
    strike: float,
    time_to_maturity: float,
    risk_free_rate: float,
    params: HestonParams,
    heston_price: float | None = None,
) -> float:
    """Calculate implied volatility from Heston model price.

    Uses Black-Scholes inversion to find implied volatility.

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
    params : HestonParams
        Heston model parameters
    heston_price : float, optional
        Heston model price. If None, calculates it.

    Returns
    -------
    float
        Implied volatility (annualized)
    """
    from scipy.optimize import brentq
    from .bsm import _BSMEuropeanValuation
    from .implied_volatility import _bsm_price

    if heston_price is None:
        heston_price = heston_call_price(spot, strike, time_to_maturity, risk_free_rate, params)

    # Invert Black-Scholes to find implied volatility
    def objective(vol: float) -> float:
        return _bsm_price(
            spot,
            strike,
            time_to_maturity,
            risk_free_rate,
            vol,
            OptionType.CALL,
            params.dividend_yield,
        ) - heston_price

    try:
        implied_vol = brentq(objective, 1e-6, 5.0, maxiter=100)
        return float(implied_vol)
    except ValueError:
        # Fallback: use ATM approximation
        return np.sqrt(params.v0)


class _HestonEuropeanValuation:
    """Heston European option valuation."""

    def __init__(self, parent: "OptionValuation"):
        self.parent = parent

    def solve(self, params: None = None) -> float:
        """Compute the Heston option value."""
        # Extract parameters
        spot = float(self.parent.underlying.initial_value)
        strike = self.parent.strike
        risk_free_rate = self.parent.discount_curve.short_rate

        # Get Heston parameters from underlying
        # Check if underlying is a HestonProcess (PathSimulation)
        if hasattr(self.parent.underlying, "v0") and hasattr(self.parent.underlying, "kappa"):
            # Extract from HestonProcess
            heston_params = HestonParams(
                initial_value=spot,
                v0=self.parent.underlying.v0,
                kappa=self.parent.underlying.kappa,
                theta=self.parent.underlying.theta,
                xi=self.parent.underlying.xi,
                rho=self.parent.underlying.rho,
                dividend_yield=getattr(self.parent.underlying, "dividend_yield", 0.0),
            )
        elif isinstance(self.parent.underlying, HestonParams):
            heston_params = self.parent.underlying
        else:
            raise ValueError(
                "Heston pricing requires underlying to be a HestonProcess "
                "(PathSimulation with HestonParams) or HestonParams instance"
            )

        time_to_maturity = calculate_year_fraction(
            self.parent.pricing_date, self.parent.maturity, day_count_convention=365
        )

        if self.parent.option_type == OptionType.CALL:
            price = heston_call_price(spot, strike, time_to_maturity, risk_free_rate, heston_params)
        else:  # PUT
            # Use put-call parity
            call_price = heston_call_price(spot, strike, time_to_maturity, risk_free_rate, heston_params)
            discount = np.exp(-risk_free_rate * time_to_maturity)
            spot_disc = spot * np.exp(-heston_params.dividend_yield * time_to_maturity)
            price = call_price - spot_disc + strike * discount

        return price

    def present_value(self, params: None = None) -> float:
        """Calculate present value using Heston formula."""
        return float(self.solve())
