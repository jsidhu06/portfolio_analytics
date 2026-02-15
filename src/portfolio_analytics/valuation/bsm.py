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
        volatility: float,
        df_r: float,
        df_q: float,
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
        volatility: float
            volatility (annualized)
        df_r: float
            discount factor P(0,T)
        df_q: float
            dividend discount factor Dq(0,T)

        Returns
        =======
        tuple of (d1, d2)
        """
        if time_to_maturity <= 0:
            raise ValueError("time_to_maturity must be positive")

        forward = spot * df_q / df_r
        numerator = np.log(forward / strike) + 0.5 * volatility**2 * time_to_maturity
        denominator = volatility * np.sqrt(time_to_maturity)

        d1 = numerator / denominator
        d2 = d1 - denominator

        return d1, d2

    def _effective_discount_factors(self, time_to_maturity: float) -> tuple[float, float]:
        df_r = float(self.parent.discount_curve.df(time_to_maturity))
        dividend_curve = self.parent.underlying.dividend_curve
        if dividend_curve is not None:
            df_q = float(dividend_curve.df(time_to_maturity))
        else:
            df_q = 1.0
        return df_r, df_q

    @staticmethod
    def _implied_rate_from_df(df: float, time_to_maturity: float) -> float:
        return -np.log(df) / time_to_maturity

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

        # Calculate time to maturity in years
        time_to_maturity = calculate_year_fraction(
            self.parent.pricing_date,
            self.parent.maturity,
            day_count_convention=DayCountConvention.ACT_365F,
        )

        df_r, df_q = self._effective_discount_factors(time_to_maturity)

        # Calculate d1 and d2
        d1, d2 = self._calculate_d_values(spot, strike, time_to_maturity, volatility, df_r, df_q)

        # Calculate option value based on option type
        discount_factor = df_r

        if self.parent.option_type is OptionType.CALL:
            option_value = spot * df_q * norm.cdf(d1) - strike * discount_factor * norm.cdf(d2)
        else:  # PUT
            option_value = strike * discount_factor * norm.cdf(-d2) - spot * df_q * norm.cdf(-d1)

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

        # Calculate time to maturity in years
        time_to_maturity = calculate_year_fraction(
            self.parent.pricing_date,
            self.parent.maturity,
            day_count_convention=DayCountConvention.ACT_365F,
        )
        df_r, df_q = self._effective_discount_factors(time_to_maturity)

        # Calculate d1
        d1, _ = self._calculate_d_values(spot, strike, time_to_maturity, volatility, df_r, df_q)

        # Adjust for dividend yield
        if self.parent.option_type is OptionType.CALL:
            delta = df_q * norm.cdf(d1)
        else:  # PUT
            delta = df_q * (norm.cdf(d1) - 1)

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

        # Calculate time to maturity in years
        time_to_maturity = calculate_year_fraction(
            self.parent.pricing_date,
            self.parent.maturity,
            day_count_convention=DayCountConvention.ACT_365F,
        )
        df_r, df_q = self._effective_discount_factors(time_to_maturity)

        if time_to_maturity <= 0:
            return 0.0

        # Calculate d1
        d1, _ = self._calculate_d_values(spot, strike, time_to_maturity, volatility, df_r, df_q)

        # Standard normal PDF evaluated at d1
        n_prime_d1 = norm.pdf(d1)

        # Adjust for dividend yield
        gamma = df_q * n_prime_d1 / (spot * volatility * np.sqrt(time_to_maturity))

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

        # Calculate time to maturity in years
        time_to_maturity = calculate_year_fraction(
            self.parent.pricing_date,
            self.parent.maturity,
            day_count_convention=DayCountConvention.ACT_365F,
        )
        df_r, df_q = self._effective_discount_factors(time_to_maturity)

        if time_to_maturity <= 0:
            return 0.0

        # Calculate d1
        d1, _ = self._calculate_d_values(spot, strike, time_to_maturity, volatility, df_r, df_q)

        # Standard normal PDF evaluated at d1
        n_prime_d1 = norm.pdf(d1)

        # Vega is per 1% change in volatility
        vega = spot * df_q * n_prime_d1 * np.sqrt(time_to_maturity) / 100

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

        # Calculate time to maturity in years
        time_to_maturity = calculate_year_fraction(
            self.parent.pricing_date,
            self.parent.maturity,
            day_count_convention=DayCountConvention.ACT_365F,
        )
        df_r, df_q = self._effective_discount_factors(time_to_maturity)
        risk_free_rate = self._implied_rate_from_df(df_r, time_to_maturity)
        dividend_yield = self._implied_rate_from_df(df_q, time_to_maturity)

        if time_to_maturity <= 0:
            return 0.0

        # Calculate d1 and d2
        d1, d2 = self._calculate_d_values(spot, strike, time_to_maturity, volatility, df_r, df_q)

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
            term2 = -risk_free_rate * strike * df_r * norm.cdf(d2)
            term3 = dividend_yield * spot * df_q * norm.cdf(d1)
            theta_annual = term1 + term2 + term3
        else:  # PUT
            term2 = risk_free_rate * strike * df_r * norm.cdf(-d2)
            term3 = -dividend_yield * spot * df_q * norm.cdf(-d1)
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

        # Calculate time to maturity in years
        time_to_maturity = calculate_year_fraction(
            self.parent.pricing_date,
            self.parent.maturity,
            day_count_convention=DayCountConvention.ACT_365F,
        )
        df_r, df_q = self._effective_discount_factors(time_to_maturity)

        if time_to_maturity <= 0:
            return 0.0

        # Calculate d2
        _, d2 = self._calculate_d_values(spot, strike, time_to_maturity, volatility, df_r, df_q)

        # Rho is per 1% change in interest rate
        if self.parent.option_type is OptionType.CALL:
            rho = strike * time_to_maturity * df_r * norm.cdf(d2) / 100
        else:  # PUT
            rho = -strike * time_to_maturity * df_r * norm.cdf(-d2) / 100

        return rho
