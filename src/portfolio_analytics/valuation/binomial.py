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
        """Setup binomial tree parameters and lattice.

        Parameters
        ==========
        num_steps: int
            number of steps in the binomial tree

        Returns
        =======
        tuple of (discount_factor, p, spot_lattice)
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
        if not (d < np.exp((r - dividend_yield) * delta_t) < u):
            raise ValueError("Arbitrage condition violated: d < exp((r-q)*dt) < u")
        p = (np.exp((r - dividend_yield) * delta_t) - d) / (u - d)

        spot_lattice = self._build_spot_lattice(
            num_steps=num_steps,
            time_intervals=time_intervals,
            up=u,
            down=d,
            short_rate=float(r),
        )

        return discount_factor, p, spot_lattice

    def _build_spot_lattice(
        self,
        *,
        num_steps: int,
        time_intervals: pd.DatetimeIndex,
        up: float,
        down: float,
        short_rate: float,
    ) -> np.ndarray:
        """Build a CRR spot lattice with time on columns (row=down moves, col=time)."""
        spot = float(self.parent.underlying.initial_value)
        discrete_dividends = self.parent.underlying.discrete_dividends

        i_idx = np.arange(num_steps + 1)[:, None]
        t_idx = np.arange(num_steps + 1)[None, :]
        up_pow = t_idx - i_idx
        down_pow = i_idx

        if discrete_dividends:
            pv_all = pv_discrete_dividends(
                discrete_dividends, time_intervals[0], time_intervals[-1], short_rate
            )
            spot = max(spot - pv_all, 0.0)

        lattice = spot * (up**up_pow) * (down**down_pow)

        if not discrete_dividends:
            return lattice

        pv_remaining = np.array(
            [
                pv_discrete_dividends(discrete_dividends, t, time_intervals[-1], short_rate)
                for t in time_intervals
            ],
            dtype=float,
        )
        lattice += pv_remaining[None, :]
        return lattice

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
        discount_factor, p, spot_lattice = self._setup_binomial_parameters(num_steps)

        option_lattice = np.zeros_like(spot_lattice)
        option_lattice[:, num_steps] = self._get_intrinsic_values(spot_lattice[:, num_steps])

        # Backward induction through the tree (time in columns)
        for t in range(num_steps - 1, -1, -1):
            continuation = (
                p * option_lattice[: t + 1, t + 1] + (1 - p) * option_lattice[1 : t + 2, t + 1]
            ) * discount_factor
            option_lattice[: t + 1, t] = continuation

        return option_lattice

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
        discount_factor, p, spot_lattice = self._setup_binomial_parameters(num_steps)

        option_lattice = np.zeros_like(spot_lattice)
        intrinsic = self._get_intrinsic_values(spot_lattice)
        option_lattice[:, num_steps] = intrinsic[:, num_steps]

        # Backward induction with early exercise decision
        for t in range(num_steps - 1, -1, -1):
            continuation = (
                p * option_lattice[: t + 1, t + 1] + (1 - p) * option_lattice[1 : t + 2, t + 1]
            ) * discount_factor
            option_lattice[: t + 1, t] = np.maximum(intrinsic[: t + 1, t], continuation)

        return option_lattice

    def present_value(self, params: BinomialParams) -> float:
        """Return PV using binomial tree method with American early exercise."""
        option_value_matrix = self.solve(params)
        pv = option_value_matrix[0, 0]

        return float(pv)


class _BinomialMCAsianValuation(_BinomialValuationBase):
    """Asian option valuation by Monte Carlo sampling on a binomial tree."""

    def solve(self, params: BinomialParams) -> np.ndarray:
        num_steps = int(params.num_steps)
        discount_factor, p, spot_lattice = self._setup_binomial_parameters(num_steps)

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
        prices = spot_lattice[row_idx, col_idx]  # (I, M+1)
        avg_s = prices.mean(axis=1)  # (I,)

        if self.parent.spec.call_put is OptionType.CALL:
            payoffs = np.maximum(avg_s - K, 0.0)
        else:
            payoffs = np.maximum(K - avg_s, 0.0)

        return payoffs * (discount_factor**num_steps)

    def present_value(self, params: BinomialParams) -> float:
        pv_pathwise = self.solve(params)
        return float(np.mean(pv_pathwise))
