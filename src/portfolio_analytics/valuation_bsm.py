"""Black-Scholes-Merton option valuation with continuous dividend yield."""

from typing import TYPE_CHECKING
import numpy as np
from scipy.stats import norm
from .utils import calculate_year_fraction
from .enums import OptionType

if TYPE_CHECKING:
    from .valuation import OptionValuation


class _BSMValuationBase:
    """Base class for Black-Scholes-Merton option valuation."""

    def __init__(self, parent: "OptionValuation"):
        self.parent = parent

    def _calculate_d_values(
        self,
        spot: float,
        strike: float,
        time_to_maturity: float,
        risk_free_rate: float,
        volatility: float,
        dividend_yield: float = 0.0,
    ) -> tuple[float, float]:
        """Calculate d1 and d2 for BSM model.

        Parameters
        ==========
        spot: float
            current spot price
        strike: float
            strike price
        time_to_maturity: float
            time to maturity in years
        risk_free_rate: float
            risk-free rate
        volatility: float
            volatility (annualized)
        dividend_yield: float, optional
            continuous dividend yield (default: 0.0)

        Returns
        =======
        tuple of (d1, d2)
        """
        if time_to_maturity <= 0:
            raise ValueError("time_to_maturity must be positive")

        numerator = (
            np.log(spot / strike)
            + (risk_free_rate - dividend_yield + 0.5 * volatility**2) * time_to_maturity
        )
        denominator = volatility * np.sqrt(time_to_maturity)

        d1 = numerator / denominator
        d2 = d1 - denominator

        return d1, d2

    def _get_discount_factor(self, time_to_maturity: float, risk_free_rate: float) -> float:
        """Calculate discount factor e^(-r*T)."""
        return np.exp(-risk_free_rate * time_to_maturity)


class _BSMEuropeanValuation(_BSMValuationBase):
    """Black-Scholes-Merton European option valuation."""

    def solve(self, **kwargs) -> float:
        """Compute the BSM option value.

        Parameters
        ==========
        **kwargs:
            placeholder for compatibility with other pricing methods
            No BSM specific parameters currently

        Returns
        =======
        float
            option value (PV)
        """

        # Extract parameters from parent OptionValuation object
        spot = self.parent.underlying.initial_value
        strike = self.parent.strike
        volatility = self.parent.underlying.volatility
        risk_free_rate = self.parent.discount_curve.short_rate
        dividend_yield = self.parent.underlying.dividend_yield

        # Calculate time to maturity in years
        time_to_maturity = calculate_year_fraction(
            self.parent.pricing_date, self.parent.maturity, day_count_convention=365
        )

        # Calculate d1 and d2
        d1, d2 = self._calculate_d_values(
            spot, strike, time_to_maturity, risk_free_rate, volatility, dividend_yield
        )

        # Calculate option value based on option type
        discount_factor = self._get_discount_factor(time_to_maturity, risk_free_rate)

        if self.parent.option_type is OptionType.CALL:
            option_value = spot * np.exp(-dividend_yield * time_to_maturity) * norm.cdf(
                d1
            ) - strike * discount_factor * norm.cdf(d2)
        else:  # PUT
            option_value = strike * discount_factor * norm.cdf(-d2) - spot * np.exp(
                -dividend_yield * time_to_maturity
            ) * norm.cdf(-d1)

        return option_value

    def present_value(
        self,
        **kwargs,
    ) -> float:
        """Calculate present value using BSM formula.

        Parameters
        ==========
        **kwargs:
            dividend_yield: float, optional (default: 0.0)
                continuous dividend yield

        Returns
        =======
        float or tuple of (pv, pv)
            present value of the option
        """
        pv = self.solve(**kwargs)

        return float(pv)

    def delta(self, **kwargs) -> float:
        """Calculate analytical delta for European option using closed-form BSM formula.

        delta = N(d1) for calls
        delta = N(d1) - 1 for puts

        Where N() is the cumulative standard normal distribution.

        Returns
        =======
        float
            analytical delta of the option
        """
        # Extract parameters from parent OptionValuation object
        spot = self.parent.underlying.initial_value
        strike = self.parent.strike
        volatility = self.parent.underlying.volatility
        risk_free_rate = self.parent.discount_curve.short_rate
        dividend_yield = self.parent.underlying.dividend_yield

        # Calculate time to maturity in years
        time_to_maturity = calculate_year_fraction(
            self.parent.pricing_date, self.parent.maturity, day_count_convention=365
        )

        # Calculate d1
        d1, _ = self._calculate_d_values(
            spot, strike, time_to_maturity, risk_free_rate, volatility, dividend_yield
        )

        # Adjust for dividend yield
        if self.parent.option_type is OptionType.CALL:
            delta = np.exp(-dividend_yield * time_to_maturity) * norm.cdf(d1)
        else:  # PUT
            delta = np.exp(-dividend_yield * time_to_maturity) * (norm.cdf(d1) - 1)

        return delta

    def gamma(self, **kwargs) -> float:
        """Calculate analytical gamma for European option using closed-form BSM formula.

        gamma = N'(d1) / (S * sigma * sqrt(T-t))

        Where N'() is the standard normal probability density function.

        Returns
        =======
        float
            analytical gamma of the option
        """
        # Extract parameters from parent OptionValuation object
        spot = self.parent.underlying.initial_value
        strike = self.parent.strike
        volatility = self.parent.underlying.volatility
        risk_free_rate = self.parent.discount_curve.short_rate
        dividend_yield = self.parent.underlying.dividend_yield

        # Calculate time to maturity in years
        time_to_maturity = calculate_year_fraction(
            self.parent.pricing_date, self.parent.maturity, day_count_convention=365
        )

        if time_to_maturity <= 0:
            return 0.0

        # Calculate d1
        d1, _ = self._calculate_d_values(
            spot, strike, time_to_maturity, risk_free_rate, volatility, dividend_yield
        )

        # Standard normal PDF evaluated at d1
        n_prime_d1 = norm.pdf(d1)

        # Adjust for dividend yield
        gamma = (
            np.exp(-dividend_yield * time_to_maturity)
            * n_prime_d1
            / (spot * volatility * np.sqrt(time_to_maturity))
        )

        return gamma

    def vega(self, **kwargs) -> float:
        """Calculate analytical vega for European option using closed-form BSM formula.

        vega = S * N'(d1) * sqrt(T-t) / 100

        Where N'() is the standard normal probability density function.
        Vega is expressed as a 1% point change in volatility (hence the division by 100).

        Returns
        =======
        float
            analytical vega of the option (per 1% point change in volatility)
        """
        # Extract parameters from parent OptionValuation object
        spot = self.parent.underlying.initial_value
        strike = self.parent.strike
        volatility = self.parent.underlying.volatility
        risk_free_rate = self.parent.discount_curve.short_rate
        dividend_yield = self.parent.underlying.dividend_yield

        # Calculate time to maturity in years
        time_to_maturity = calculate_year_fraction(
            self.parent.pricing_date, self.parent.maturity, day_count_convention=365
        )

        if time_to_maturity <= 0:
            return 0.0

        # Calculate d1
        d1, _ = self._calculate_d_values(
            spot, strike, time_to_maturity, risk_free_rate, volatility, dividend_yield
        )

        # Standard normal PDF evaluated at d1
        n_prime_d1 = norm.pdf(d1)

        # Vega is per 1% change in volatility
        vega = (
            spot
            * np.exp(-dividend_yield * time_to_maturity)
            * n_prime_d1
            * np.sqrt(time_to_maturity)
            / 100
        )

        return vega
