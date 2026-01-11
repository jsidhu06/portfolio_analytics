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

    def generate_payoff(self, **kwargs) -> float:
        """Generate option payoff at maturity using BSM formula.

        Parameters
        ==========
        **kwargs:
            dividend_yield: float, optional (default: 0.0)
                continuous dividend yield

        Returns
        =======
        float
            option payoff value
        """
        # TO DO: The current implementation is for testing purposes. In future, we will
        # likely make dividend_yield a param and attr of the UnderlyingData class.
        # This will require amending MCS and Binomial calcs to handle a continuous dividend yield.
        dividend_yield = kwargs.get("dividend_yield", 0.0)

        # Extract parameters from parent OptionValuation object
        spot = self.parent.underlying.initial_value
        strike = self.parent.strike
        volatility = self.parent.underlying.volatility
        risk_free_rate = self.parent.discount_curve.short_rate

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
        full: bool = False,
        **kwargs,
    ) -> float | tuple[float, float]:
        """Calculate present value using BSM formula.

        Parameters
        ==========
        full: bool
            if True, return tuple of (pv, pv); otherwise return pv
        **kwargs:
            dividend_yield: float, optional (default: 0.0)
                continuous dividend yield

        Returns
        =======
        float or tuple of (pv, pv)
            present value of the option
        """
        pv = self.generate_payoff(**kwargs)

        if full:
            return pv, pv
        return pv
