"""SABR (Stochastic Alpha Beta Rho) volatility model.

The SABR model is a stochastic volatility model used to parameterize
the implied volatility smile. It's particularly popular in interest rate
derivatives markets.

References
----------
Hagan, P. S., Kumar, D., Lesniewski, A. S., & Woodward, D. E. (2002).
Managing smile risk. The Best of Wilmott, 1, 249-296.
"""

from dataclasses import dataclass
import numpy as np
from scipy.optimize import minimize
from .surface import VolatilityInterpolator, VolatilityQuote


@dataclass(frozen=True)
class SABRParams:
    """SABR model parameters.

    Parameters
    ----------
    alpha : float
        Initial volatility level (volatility of volatility scale)
    beta : float
        Skew parameter (0 <= beta <= 1). Common values:
        - beta = 0: Normal model
        - beta = 0.5: CIR model
        - beta = 1: Lognormal model (Black-Scholes)
    rho : float
        Correlation between asset and volatility (-1 <= rho <= 1)
    nu : float
        Volatility of volatility (vol of vol)
    forward : float
        Forward price of the underlying
    """

    alpha: float
    beta: float
    rho: float
    nu: float
    forward: float

    def __post_init__(self):
        """Validate parameters."""
        if self.alpha <= 0:
            raise ValueError("alpha must be positive")
        if not (0 <= self.beta <= 1):
            raise ValueError("beta must be in [0, 1]")
        if not (-1 <= self.rho <= 1):
            raise ValueError("rho must be in [-1, 1]")
        if self.nu < 0:
            raise ValueError("nu must be non-negative")
        if self.forward <= 0:
            raise ValueError("forward must be positive")


def sabr_implied_vol(
    strike: float,
    params: SABRParams,
    expiry: float = 1.0,
) -> float:
    """Calculate SABR implied volatility.

    Uses the Hagan et al. (2002) asymptotic expansion formula.

    Parameters
    ----------
    strike : float
        Strike price
    params : SABRParams
        SABR model parameters
    expiry : float, optional
        Time to expiry in years (default: 1.0)

    Returns
    -------
    float
        Implied volatility (annualized)

    Notes
    -----
    The formula is valid for strikes not too far from the forward.
    For very deep OTM options, the expansion may break down.
    """
    if expiry <= 0:
        return 0.0

    F = params.forward
    K = strike
    alpha = params.alpha
    beta = params.beta
    rho = params.rho
    nu = params.nu

    # Handle ATM case
    if abs(K - F) < 1e-10:
        # ATM volatility
        vol_atm = alpha / (F ** (1 - beta))
        return vol_atm

    # Log moneyness
    z = (nu / alpha) * (F * K) ** ((1 - beta) / 2) * np.log(F / K)
    x = np.log((np.sqrt(1 - 2 * rho * z + z**2) + z - rho) / (1 - rho))

    # Avoid division by zero
    if abs(x) < 1e-10:
        x = z

    # SABR formula
    factor1 = alpha / ((F * K) ** ((1 - beta) / 2))
    factor2 = 1 + ((1 - beta) ** 2 / 24) * (np.log(F / K)) ** 2
    factor3 = 1 + ((1 - beta) ** 4 / 1920) * (np.log(F / K)) ** 4

    numerator = z / x
    denominator = factor2 * factor3

    correction = (
        1
        + (
            ((1 - beta) ** 2 / 24) * (alpha**2 / ((F * K) ** (1 - beta)))
            + (rho * beta * nu * alpha) / (4 * ((F * K) ** ((1 - beta) / 2)))
            + ((2 - 3 * rho**2) / 24) * nu**2
        )
        * expiry
    )

    vol = factor1 * (numerator / denominator) * correction

    # Ensure non-negative
    return max(float(vol), 0.0)


class SABRInterpolator(VolatilityInterpolator):
    """SABR-based volatility interpolator.

    Fits SABR parameters to market quotes and provides interpolation.
    """

    def __init__(
        self,
        quotes: list[VolatilityQuote],
        beta: float = 0.5,
        initial_guess: dict[str, float] | None = None,
    ):
        """Initialize SABR interpolator.

        Parameters
        ----------
        quotes : list[VolatilityQuote]
            Market volatility quotes
        beta : float, optional
            Fixed beta parameter (default: 0.5). Common practice is to fix beta
            and calibrate alpha, rho, nu.
        initial_guess : dict[str, float], optional
            Initial guess for parameters: {'alpha', 'rho', 'nu', 'forward'}
        """
        if not quotes:
            raise ValueError("quotes list cannot be empty")

        self.quotes = quotes
        self.beta = beta

        # Extract forward (use ATM strike or median)
        if initial_guess and "forward" in initial_guess:
            self.forward = initial_guess["forward"]
        else:
            # Use median strike as proxy for forward
            strikes = [q.strike for q in quotes]
            self.forward = np.median(strikes)

        # Calibrate SABR parameters
        self.params = self._calibrate(initial_guess)

    def _calibrate(self, initial_guess: dict[str, float] | None = None) -> SABRParams:
        """Calibrate SABR parameters to market quotes."""
        # Initial parameter guess
        if initial_guess:
            alpha_init = initial_guess.get("alpha", 0.2)
            rho_init = initial_guess.get("rho", 0.0)
            nu_init = initial_guess.get("nu", 0.5)
        else:
            # Default initial guess
            vols = [q.implied_volatility for q in self.quotes]
            alpha_init = np.median(vols) * (self.forward ** (1 - self.beta))
            rho_init = 0.0
            nu_init = 0.5

        # Objective function: sum of squared errors
        def objective(params: np.ndarray) -> float:
            alpha, rho, nu = params
            try:
                sabr_params = SABRParams(
                    alpha=max(alpha, 1e-6),
                    beta=self.beta,
                    rho=np.clip(rho, -0.99, 0.99),
                    nu=max(nu, 1e-6),
                    forward=self.forward,
                )
                errors = []
                for quote in self.quotes:
                    model_vol = sabr_implied_vol(quote.strike, sabr_params, quote.expiry)
                    error = (model_vol - quote.implied_volatility) ** 2
                    errors.append(error)
                return sum(errors)
            except (ValueError, ZeroDivisionError):
                return 1e10

        # Optimize
        result = minimize(
            objective,
            x0=[alpha_init, rho_init, nu_init],
            method="L-BFGS-B",
            bounds=[(1e-6, 10.0), (-0.99, 0.99), (1e-6, 5.0)],
        )

        alpha_opt, rho_opt, nu_opt = result.x

        return SABRParams(
            alpha=alpha_opt,
            beta=self.beta,
            rho=rho_opt,
            nu=nu_opt,
            forward=self.forward,
        )

    def get_vol(self, strike: float, expiry: float) -> float:
        """Get SABR interpolated volatility.

        Parameters
        ----------
        strike : float
            Strike price
        expiry : float
            Time to expiry in years

        Returns
        -------
        float
            Implied volatility
        """
        return sabr_implied_vol(strike, self.params, expiry)
