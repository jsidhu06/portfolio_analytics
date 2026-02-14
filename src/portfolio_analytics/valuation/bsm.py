"""Black-Scholes-Merton option valuation with continuous dividend yield."""

from typing import TYPE_CHECKING
import numpy as np
from scipy.stats import norm
from ..utils import calculate_year_fraction, pv_discrete_dividends
from ..enums import DayCountConvention, OptionType

if TYPE_CHECKING:
    from .core import OptionValuation


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

    def _adjusted_spot(self) -> float:
        """Adjust spot for discrete dividends using PV of future dividends."""
        spot = float(self.parent.underlying.initial_value)
        discrete_dividends = self.parent.underlying.discrete_dividends
        if not discrete_dividends:
            return spot
        if self.parent.discount_curve.flat_rate is None:
            raise NotImplementedError("Discrete dividend adjustments require a flat curve.")
        pv_divs = pv_discrete_dividends(
            discrete_dividends,
            self.parent.pricing_date,
            self.parent.maturity,
            float(self.parent.discount_curve.flat_rate),
        )
        return max(spot - pv_divs, 0.0)


class _BSMEuropeanValuation(_BSMValuationBase):
    """Black-Scholes-Merton European option valuation."""

    def solve(self) -> float:
        """Compute the BSM option value."""

        # Extract parameters from parent OptionValuation object
        spot = self._adjusted_spot()
        strike = self.parent.strike
        volatility = self.parent.underlying.volatility
        risk_free_rate = float(self.parent.discount_curve.flat_rate)
        dividend_yield = self.parent.underlying.dividend_yield

        # Calculate time to maturity in years
        time_to_maturity = calculate_year_fraction(
            self.parent.pricing_date,
            self.parent.maturity,
            day_count_convention=DayCountConvention.ACT_365F,
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

    def present_value(self) -> float:
        """Calculate present value using BSM formula."""
        pv = self.solve()

        return float(pv)

    def delta(self) -> float:
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
        spot = self._adjusted_spot()
        strike = self.parent.strike
        volatility = self.parent.underlying.volatility
        risk_free_rate = float(self.parent.discount_curve.flat_rate)
        dividend_yield = self.parent.underlying.dividend_yield

        # Calculate time to maturity in years
        time_to_maturity = calculate_year_fraction(
            self.parent.pricing_date,
            self.parent.maturity,
            day_count_convention=DayCountConvention.ACT_365F,
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

    def gamma(self) -> float:
        """Calculate analytical gamma for European option using closed-form BSM formula.

        gamma = N'(d1) / (S * sigma * sqrt(T-t))

        Where N'() is the standard normal probability density function.

        Returns
        =======
        float
            analytical gamma of the option
        """
        # Extract parameters from parent OptionValuation object
        spot = self._adjusted_spot()
        strike = self.parent.strike
        volatility = self.parent.underlying.volatility
        risk_free_rate = float(self.parent.discount_curve.flat_rate)
        dividend_yield = self.parent.underlying.dividend_yield

        # Calculate time to maturity in years
        time_to_maturity = calculate_year_fraction(
            self.parent.pricing_date,
            self.parent.maturity,
            day_count_convention=DayCountConvention.ACT_365F,
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

    def vega(self) -> float:
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
        spot = self._adjusted_spot()
        strike = self.parent.strike
        volatility = self.parent.underlying.volatility
        risk_free_rate = float(self.parent.discount_curve.flat_rate)
        dividend_yield = self.parent.underlying.dividend_yield

        # Calculate time to maturity in years
        time_to_maturity = calculate_year_fraction(
            self.parent.pricing_date,
            self.parent.maturity,
            day_count_convention=DayCountConvention.ACT_365F,
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

    def theta(self) -> float:
        """Calculate analytical theta for European option using closed-form BSM formula.

        Theta measures the time decay of the option value (rate of change with respect to time).
        Returns the change in option value for a 1-day decrease in time to maturity.

        For call:
            theta = -(S * N'(d1) * sigma * e^(-qT)) / (2 * sqrt(T))
                    - r * K * e^(-rT) * N(d2)
                    + q * S * e^(-qT) * N(d1)

        For put:
            theta = -(S * N'(d1) * sigma * e^(-qT)) / (2 * sqrt(T))
                    + r * K * e^(-rT) * N(-d2)
                    - q * S * e^(-qT) * N(-d1)

        Returns
        =======
        float
            analytical theta of the option (per day)
        """
        # Extract parameters from parent OptionValuation object
        spot = self._adjusted_spot()
        strike = self.parent.strike
        volatility = self.parent.underlying.volatility
        risk_free_rate = float(self.parent.discount_curve.flat_rate)
        dividend_yield = self.parent.underlying.dividend_yield

        # Calculate time to maturity in years
        time_to_maturity = calculate_year_fraction(
            self.parent.pricing_date,
            self.parent.maturity,
            day_count_convention=DayCountConvention.ACT_365F,
        )

        if time_to_maturity <= 0:
            return 0.0

        # Calculate d1 and d2
        d1, d2 = self._calculate_d_values(
            spot, strike, time_to_maturity, risk_free_rate, volatility, dividend_yield
        )

        # Standard normal PDF evaluated at d1
        n_prime_d1 = norm.pdf(d1)

        # Common term for both call and put
        term1 = -(
            spot
            * np.exp(-dividend_yield * time_to_maturity)
            * n_prime_d1
            * volatility
            / (2 * np.sqrt(time_to_maturity))
        )

        if self.parent.option_type is OptionType.CALL:
            term2 = (
                -risk_free_rate * strike * np.exp(-risk_free_rate * time_to_maturity) * norm.cdf(d2)
            )
            term3 = (
                dividend_yield * spot * np.exp(-dividend_yield * time_to_maturity) * norm.cdf(d1)
            )
            theta_annual = term1 + term2 + term3
        else:  # PUT
            term2 = (
                risk_free_rate * strike * np.exp(-risk_free_rate * time_to_maturity) * norm.cdf(-d2)
            )
            term3 = (
                -dividend_yield * spot * np.exp(-dividend_yield * time_to_maturity) * norm.cdf(-d1)
            )
            theta_annual = term1 + term2 + term3

        # Convert from annual to per-day (divide by 365)
        theta_daily = theta_annual / 365

        return theta_daily

    def rho(self) -> float:
        """Calculate analytical rho for European option using closed-form BSM formula.

        Rho measures sensitivity to interest rate changes.
        Returns the change in option value for a 1% (0.01) change in the risk-free rate.

        For call: rho = K * T * e^(-rT) * N(d2) / 100
        For put:  rho = -K * T * e^(-rT) * N(-d2) / 100

        Returns
        =======
        float
            analytical rho of the option (per 1% change in interest rate)
        """
        # Extract parameters from parent OptionValuation object
        spot = self._adjusted_spot()
        strike = self.parent.strike
        volatility = self.parent.underlying.volatility
        risk_free_rate = float(self.parent.discount_curve.flat_rate)
        dividend_yield = self.parent.underlying.dividend_yield

        # Calculate time to maturity in years
        time_to_maturity = calculate_year_fraction(
            self.parent.pricing_date,
            self.parent.maturity,
            day_count_convention=DayCountConvention.ACT_365F,
        )

        if time_to_maturity <= 0:
            return 0.0

        # Calculate d2
        _, d2 = self._calculate_d_values(
            spot, strike, time_to_maturity, risk_free_rate, volatility, dividend_yield
        )

        # Rho is per 1% change in interest rate
        if self.parent.option_type is OptionType.CALL:
            rho = (
                strike
                * time_to_maturity
                * np.exp(-risk_free_rate * time_to_maturity)
                * norm.cdf(d2)
                / 100
            )
        else:  # PUT
            rho = (
                -strike
                * time_to_maturity
                * np.exp(-risk_free_rate * time_to_maturity)
                * norm.cdf(-d2)
                / 100
            )

        return rho
