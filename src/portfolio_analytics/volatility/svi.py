"""SVI (Stochastic Volatility Inspired) parameterization.

The SVI parameterization is a flexible model for the total variance smile.
It's widely used in equity derivatives markets.

References
----------
Gatheral, J., & Jacquier, A. (2014). Arbitrage-free SVI volatility surfaces.
Quantitative Finance, 14(1), 59-71.
"""

from dataclasses import dataclass
import numpy as np
from scipy.optimize import minimize
from .surface import VolatilityInterpolator, VolatilityQuote


@dataclass(frozen=True)
class SVIParams:
    """SVI model parameters.

    Parameters
    ----------
    a : float
        Vertical translation (overall level)
    b : float
        Total variance slope (controls smile steepness)
    rho : float
        Skew parameter (-1 <= rho <= 1)
    m : float
        Minimum of smile (location parameter)
    sigma : float
        Smile width parameter (must be positive)
    """

    a: float
    b: float
    rho: float
    m: float
    sigma: float

    def __post_init__(self):
        """Validate parameters."""
        if self.b < 0:
            raise ValueError("b must be non-negative")
        if not (-1 <= self.rho <= 1):
            raise ValueError("rho must be in [-1, 1]")
        if self.sigma <= 0:
            raise ValueError("sigma must be positive")


def svi_total_variance(
    log_moneyness: float,
    params: SVIParams,
) -> float:
    """Calculate SVI total variance.

    The SVI parameterization models total variance w(k) = sigma^2 * T,
    where k = log(K/F) is the log moneyness.

    Parameters
    ----------
    log_moneyness : float
        Log moneyness: log(strike / forward)
    params : SVIParams
        SVI parameters

    Returns
    -------
    float
        Total variance (sigma^2 * T)

    Notes
    -----
    Formula: w(k) = a + b * (rho * (k - m) + sqrt((k - m)^2 + sigma^2))
    """
    k = log_moneyness
    diff = k - params.m
    sqrt_term = np.sqrt(diff**2 + params.sigma**2)
    total_var = params.a + params.b * (params.rho * diff + sqrt_term)

    return max(float(total_var), 0.0)


def svi_implied_vol(
    strike: float,
    forward: float,
    params: SVIParams,
    expiry: float = 1.0,
) -> float:
    """Calculate SVI implied volatility from total variance.

    Parameters
    ----------
    strike : float
        Strike price
    forward : float
        Forward price
    params : SVIParams
        SVI parameters
    expiry : float, optional
        Time to expiry in years (default: 1.0)

    Returns
    -------
    float
        Implied volatility (annualized)
    """
    if expiry <= 0:
        return 0.0

    log_moneyness = np.log(strike / forward)
    total_var = svi_total_variance(log_moneyness, params)

    # Convert total variance to implied volatility
    implied_vol = np.sqrt(total_var / expiry)

    return float(implied_vol)


class SVIInterpolator(VolatilityInterpolator):
    """SVI-based volatility interpolator.

    Fits SVI parameters to market quotes and provides interpolation.
    """

    def __init__(
        self,
        quotes: list[VolatilityQuote],
        forward: float | None = None,
        initial_guess: dict[str, float] | None = None,
    ):
        """Initialize SVI interpolator.

        Parameters
        ----------
        quotes : list[VolatilityQuote]
            Market volatility quotes (should be for same expiry)
        forward : float, optional
            Forward price. If None, uses median strike.
        initial_guess : dict[str, float], optional
            Initial guess for parameters: {'a', 'b', 'rho', 'm', 'sigma'}
        """
        if not quotes:
            raise ValueError("quotes list cannot be empty")

        self.quotes = quotes

        # Extract forward
        if forward is None:
            strikes = [q.strike for q in quotes]
            self.forward = np.median(strikes)
        else:
            self.forward = forward

        # Check if all quotes have same expiry (SVI is typically per-expiry)
        expiries = [q.expiry for q in quotes]
        if len(set(expiries)) > 1:
            # Use median expiry
            self.expiry = np.median(expiries)
        else:
            self.expiry = expiries[0]

        # Calibrate SVI parameters
        self.params = self._calibrate(initial_guess)

    def _calibrate(self, initial_guess: dict[str, float] | None = None) -> SVIParams:
        """Calibrate SVI parameters to market quotes."""
        # Initial parameter guess
        if initial_guess:
            a_init = initial_guess.get("a", 0.04)
            b_init = initial_guess.get("b", 0.1)
            rho_init = initial_guess.get("rho", -0.3)
            m_init = initial_guess.get("m", 0.0)
            sigma_init = initial_guess.get("sigma", 0.1)
        else:
            # Default initial guess
            vols = [q.implied_volatility for q in self.quotes]
            total_vars = [v**2 * self.expiry for v in vols]
            a_init = np.median(total_vars)
            b_init = 0.1
            rho_init = -0.3
            m_init = 0.0
            sigma_init = 0.1

        # Objective function: sum of squared errors
        def objective(params: np.ndarray) -> float:
            a, b, rho, m, sigma = params
            try:
                svi_params = SVIParams(
                    a=a,
                    b=max(b, 1e-6),
                    rho=np.clip(rho, -0.99, 0.99),
                    m=m,
                    sigma=max(sigma, 1e-6),
                )
                errors = []
                for quote in self.quotes:
                    log_moneyness = np.log(quote.strike / self.forward)
                    model_total_var = svi_total_variance(log_moneyness, svi_params)
                    market_total_var = quote.implied_volatility**2 * quote.expiry
                    error = (model_total_var - market_total_var) ** 2
                    errors.append(error)
                return sum(errors)
            except (ValueError, ZeroDivisionError):
                return 1e10

        # Optimize
        result = minimize(
            objective,
            x0=[a_init, b_init, rho_init, m_init, sigma_init],
            method="L-BFGS-B",
            bounds=[
                (-1.0, 1.0),  # a
                (1e-6, 10.0),  # b
                (-0.99, 0.99),  # rho
                (-2.0, 2.0),  # m
                (1e-6, 2.0),  # sigma
            ],
        )

        a_opt, b_opt, rho_opt, m_opt, sigma_opt = result.x

        return SVIParams(
            a=a_opt,
            b=b_opt,
            rho=rho_opt,
            m=m_opt,
            sigma=sigma_opt,
        )

    def get_vol(self, strike: float, expiry: float) -> float:
        """Get SVI interpolated volatility.

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

        Notes
        -----
        SVI is typically calibrated per-expiry. This implementation uses
        the calibrated expiry. For different expiries, you should create
        separate SVIInterpolator instances.
        """
        return svi_implied_vol(strike, self.forward, self.params, expiry)
