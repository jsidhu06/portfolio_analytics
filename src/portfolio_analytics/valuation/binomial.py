"""Valuation of European and American options using the binomial option pricing model of
Cox-Ross-Rubinstein
"""

from typing import TYPE_CHECKING
import numpy as np
import pandas as pd
from ..enums import OptionType
from ..utils import calculate_year_fraction, pv_discrete_dividends
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

        S_0 = float(self.parent.underlying.initial_value)
        discrete_dividends = getattr(self.parent.underlying, "discrete_dividends", [])

        # If no discrete dividends, use the closed-form CRR lattice.
        if not discrete_dividends:
            binomial_matrix = S_0 * np.exp(sigma * np.sqrt(delta_t) * (up - down))
            return discount_factor, p, binomial_matrix

        # Hull-style approach (prepaid forward): build lattice for S* = S - PV(divs)
        pv_all = pv_discrete_dividends(discrete_dividends, start, end, float(r))
        S_star0 = max(S_0 - pv_all, 0.0)
        binomial_matrix = S_star0 * np.exp(sigma * np.sqrt(delta_t) * (up - down))

        # Add PV of remaining dividends to each time step (same adjustment per column)
        for i in range(num_steps + 1):
            pv_remaining = pv_discrete_dividends(
                discrete_dividends, time_intervals[i], end, float(r)
            )
            if pv_remaining != 0.0:
                binomial_matrix[: i + 1, i] += pv_remaining

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


class _BinomialMCAsianValuation(_BinomialValuationBase):
    """Asian option valuation by Monte Carlo sampling on a binomial tree."""

    def solve(self, params: BinomialParams) -> np.ndarray:
        num_steps = int(params.num_steps)
        discount_factor, p, binomial_matrix = self._setup_binomial_parameters(num_steps)

        if params.mc_paths is None:
            raise ValueError("BinomialParams.mc_paths must be set for Asian binomial MC")

        mc_paths = int(params.mc_paths)
        rng = np.random.default_rng(params.random_seed)

        K = self.parent.strike
        if K is None:
            raise ValueError("strike is required for Asian option payoff.")

        # Vectorized simulation of binomial paths:
        # draw Bernoulli down-steps for each path and step
        downs = rng.random((mc_paths, num_steps)) > p  # (I,M)
        # cumulative count of downs gives row index at each step
        down_counts = np.cumsum(downs, axis=1)  # (I,M)
        # prepend time 0 (row=0)
        row_idx = np.concatenate(
            [np.zeros((mc_paths, 1), dtype=int), down_counts], axis=1
        )  # (I,M+1)
        col_idx = np.arange(num_steps + 1, dtype=int)  # (M+1,)

        # gather prices along each simulated path
        # binomial_matrix is (M+1, M+1)
        # Advanced indexing: col_idx broadcasts to (I, M+1), so each
        # (row_idx[i, t], col_idx[t]) selects the node for path i at time t.
        prices = binomial_matrix[row_idx, col_idx]  # (I, M+1)
        avg_s = prices.mean(axis=1)  # (I,)

        if self.parent.spec.call_put is OptionType.CALL:
            payoffs = np.maximum(avg_s - K, 0.0)
        else:
            payoffs = np.maximum(K - avg_s, 0.0)

        return payoffs * (discount_factor**num_steps)

    def present_value(self, params: BinomialParams) -> float:
        pv_pathwise = self.solve(params)
        return float(np.mean(pv_pathwise))
