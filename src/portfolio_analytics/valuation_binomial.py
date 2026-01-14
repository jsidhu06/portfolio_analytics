"""Valuation of European and American options using the binomial option pricing model of
Cox-Ross-Rubinstein
"""

from typing import TYPE_CHECKING
import numpy as np
import pandas as pd
from .enums import OptionType
from .utils import calculate_year_fraction

if TYPE_CHECKING:
    from .valuation import OptionValuation


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
        if self.parent.option_type in (OptionType.CALL, OptionType.PUT) and K is None:
            raise ValueError("strike is required for vanilla American call/put payoff.")

        if self.parent.option_type is OptionType.CALL:
            return np.maximum(instrument_values - K, 0)

        elif self.parent.option_type is OptionType.PUT:
            return np.maximum(K - instrument_values, 0)

        elif self.parent.option_type is OptionType.CONDOR:
            payoff_fn = getattr(self.parent.spec, "payoff", None)
            if payoff_fn is None:
                raise TypeError(
                    "Condor payoff requires parent.spec.payoff(spot) to be defined (e.g., CondorSpec)"
                )
            return payoff_fn(instrument_values)
        else:
            raise ValueError("Unsupported option type for binomial valuation.")


class _BinomialEuropeanValuation(_BinomialValuationBase):
    """Implementation of European option valuation using binomial tree."""

    def generate_payoff(self, **kwargs) -> np.ndarray:
        """Generate option value matrix using binomial tree.

        Parameters
        ==========
        **kwargs:
            num_steps: int, optional (default: 500)
                number of steps in the binomial tree

        Returns
        =======
        np.ndarray
            option values at each node in the tree
        """
        num_steps = kwargs.get("num_steps", 500)
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

    def present_value(
        self,
        full: bool = False,
        **kwargs,
    ) -> float | tuple[float, np.ndarray]:
        """Return PV using binomial tree method.

        Parameters
        ==========
        full: bool
            return also full option value matrix
        **kwargs:
            num_steps: int, optional (default: 500)
                number of steps in the binomial tree

        Returns
        =======
        float or tuple of (pv, option_value_matrix)
        """
        num_steps = kwargs.get("num_steps", 500)
        option_value_matrix = self.generate_payoff(num_steps=num_steps)
        pv = option_value_matrix[0, 0]

        if full:
            return pv, option_value_matrix
        return pv


class _BinomialAmericanValuation(_BinomialValuationBase):
    """Implementation of American option valuation using binomial tree."""

    def generate_payoff(self, **kwargs) -> np.ndarray:
        """Generate option value matrix using binomial tree with early exercise.

        Parameters
        ==========
        **kwargs:
            num_steps: int, optional (default: 500)
                number of steps in the binomial tree

        Returns
        =======
        np.ndarray
            option values at each node in the tree
        """
        num_steps = kwargs.get("num_steps", 500)
        discount_factor, p, binomial_matrix = self._setup_binomial_parameters(num_steps)

        # Initialize with intrinsic values at maturity
        V = self._get_intrinsic_values(binomial_matrix)
        intrinsic_values = V.copy()

        # Backward induction with early exercise decision
        z = 0
        for i in range(num_steps - 1, -1, -1):
            # Calculate continuation values
            continuation = (
                p * V[0 : num_steps - z, i + 1] + (1 - p) * V[1 : num_steps - z + 1, i + 1]
            ) * discount_factor

            # American option: take max of intrinsic vs continuation value
            V[0 : num_steps - z, i] = np.maximum(
                intrinsic_values[0 : num_steps - z, i], continuation
            )
            z += 1

        return V

    def present_value(
        self,
        full: bool = False,
        **kwargs,
    ) -> float | tuple[float, np.ndarray]:
        """Return PV using binomial tree method with American early exercise.

        Parameters
        ==========
        full: bool
            return also full option value matrix
        **kwargs:
            num_steps: int, optional (default: 500)
                number of steps in the binomial tree

        Returns
        =======
        float or tuple of (pv, option_value_matrix)
        """
        num_steps = kwargs.get("num_steps", 500)
        option_value_matrix = self.generate_payoff(num_steps=num_steps)
        pv = option_value_matrix[0, 0]

        if full:
            return pv, option_value_matrix
        return pv
