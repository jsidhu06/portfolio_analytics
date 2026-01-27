"""Valuation of European and American options using the binomial option pricing model of
Cox-Ross-Rubinstein
"""

from typing import TYPE_CHECKING
import numpy as np
import pandas as pd
from ..enums import OptionType
from ..utils import calculate_year_fraction
from .params import BinomialParams

if TYPE_CHECKING:
    from .core import OptionValuation


class _BinomialValuationBase:
    """Base class for binomial tree option valuation."""

    def __init__(self, parent: "OptionValuation"):
        self.parent = parent

    def _setup_binomial_parameters(self, num_steps: int) -> tuple:
        """Setup binomial tree parameters.

        Parameters
        ==========
        num_steps: int
            number of steps in the binomial tree

        Returns
        =======
        tuple of (discount_factor, p, binomial_matrix)
        """
        start = self.parent.pricing_date
        end = self.parent.maturity
        time_intervals = pd.date_range(start, end, periods=num_steps + 1)

        # Numerical Parameters
        delta_t = calculate_year_fraction(time_intervals[0], time_intervals[1])
        r = self.parent.discount_curve.short_rate
        discount_factor = np.exp(-r * delta_t)
        sigma = self.parent.underlying.volatility
        dividend_yield = self.parent.underlying.dividend_yield
        u = np.exp(sigma * np.sqrt(delta_t))
        d = 1 / u
        assert (
            d < np.exp((r - dividend_yield) * delta_t) < u
        ), "Arbitrage condition violated: d < e^( (r - q) * dt ) < u"
        p = (np.exp((r - dividend_yield) * delta_t) - d) / (u - d)

        # Build binomial tree of stock prices
        up = np.arange(num_steps + 1)
        up = np.resize(up, (num_steps + 1, num_steps + 1))
        down = up.T * 2

        S_0 = self.parent.underlying.initial_value
        binomial_matrix = S_0 * np.exp(sigma * np.sqrt(delta_t) * (up - down))

        return discount_factor, p, binomial_matrix

    def _get_intrinsic_values(self, instrument_values: np.ndarray) -> np.ndarray:
        """Calculate intrinsic values at each node.

        Parameters
        ==========
        instrument_values: np.ndarray
            price of underlying at each node

        Returns
        =======
        np.ndarray
            intrinsic values
        """
        K = self.parent.strike
        if self.parent.option_type in (OptionType.CALL, OptionType.PUT):
            if K is None:
                raise ValueError("strike is required for vanilla American call/put payoff.")
            if self.parent.option_type is OptionType.CALL:
                return np.maximum(instrument_values - K, 0)
            return np.maximum(K - instrument_values, 0)

        payoff_fn = self.parent.spec.payoff
        return payoff_fn(instrument_values)


class _BinomialEuropeanValuation(_BinomialValuationBase):
    """Implementation of European option valuation using binomial tree."""

    def solve(self, params: BinomialParams) -> np.ndarray:
        """Compute the option value lattice using a binomial tree."""
        num_steps = int(params.num_steps)
        discount_factor, p, binomial_matrix = self._setup_binomial_parameters(num_steps)

        # Initialize with intrinsic values at maturity
        V = self._get_intrinsic_values(binomial_matrix)

        # Backward induction through the tree
        z = 0
        for i in range(num_steps - 1, -1, -1):
            V[0 : num_steps - z, i] = (
                p * V[0 : num_steps - z, i + 1] + (1 - p) * V[1 : num_steps - z + 1, i + 1]
            ) * discount_factor
            z += 1

        return V

    def present_value(self, params: BinomialParams) -> float:
        """Return PV using binomial tree method."""
        option_value_matrix = self.solve(params)
        pv = option_value_matrix[0, 0]

        return float(pv)


class _BinomialAmericanValuation(_BinomialValuationBase):
    """Implementation of American option valuation using binomial tree."""

    def solve(self, params: BinomialParams) -> np.ndarray:
        """Compute the option value lattice using a binomial tree with early exercise."""
        num_steps = int(params.num_steps)
        discount_factor, p, binomial_matrix = self._setup_binomial_parameters(num_steps)

        # Initialize with intrinsic values at maturity
        V = self._get_intrinsic_values(binomial_matrix)
        intrinsic_values = V.copy()

        # Backward induction with early exercise decision
        z = 0
        for i in range(num_steps - 1, -1, -1):
            # Calculate discounted present continuation values
            continuation = (
                p * V[0 : num_steps - z, i + 1] + (1 - p) * V[1 : num_steps - z + 1, i + 1]
            ) * discount_factor

            # American option: take max of intrinsic vs discounted present continuation value
            V[0 : num_steps - z, i] = np.maximum(
                intrinsic_values[0 : num_steps - z, i], continuation
            )
            z += 1

        return V

    def present_value(self, params: BinomialParams) -> float:
        """Return PV using binomial tree method with American early exercise."""
        option_value_matrix = self.solve(params)
        pv = option_value_matrix[0, 0]

        return float(pv)
