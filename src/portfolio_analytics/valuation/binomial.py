"""Valuation of European and American options using the binomial option pricing model of
Cox-Ross-Rubinstein
"""

from __future__ import annotations
from typing import TYPE_CHECKING
import logging
import numpy as np
import pandas as pd
from ..enums import OptionType, AsianAveraging, ExerciseType
from ..utils import calculate_year_fraction, pv_discrete_dividends, log_timing
from ..exceptions import (
    ArbitrageViolationError,
    ConfigurationError,
    NumericalError,
    UnsupportedFeatureError,
    ValidationError,
)
from .params import BinomialParams

if TYPE_CHECKING:
    from .core import OptionValuation, AsianOptionSpec


logger = logging.getLogger(__name__)


class _BinomialValuationBase:
    """Base class for binomial tree option valuation."""

    def __init__(self, parent: OptionValuation) -> None:
        self.parent = parent
        if not isinstance(parent.params, BinomialParams):
            raise ConfigurationError(
                "Binomial valuation requires BinomialParams on OptionValuation"
            )
        self.binom_params: BinomialParams = parent.params

    def _setup_binomial_parameters(
        self, num_steps: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Setup binomial tree parameters and lattice.

        Parameters
        ==========
        num_steps: int
            number of steps in the binomial tree

        Returns
        =======
        tuple of (discount_factors, p, spot_lattice)
            discount_factors : shape (num_steps,) — per-step discount factors
            p : shape (num_steps,) — risk-neutral up-move probabilities
            spot_lattice : shape (num_steps+1, num_steps+1) — CRR spot prices
        """
        start = self.parent.pricing_date
        end = self.parent.maturity
        time_intervals = pd.date_range(start, end, periods=num_steps + 1)

        # Numerical Parameters
        time_grid = np.array(
            [calculate_year_fraction(start, t) for t in time_intervals], dtype=float
        )
        dt_steps = np.diff(time_grid)
        if not np.allclose(dt_steps, dt_steps[0]):
            raise ValidationError("Binomial tree requires equal time steps.")
        delta_t = float(dt_steps[0])

        sigma = float(self.parent.underlying.volatility)
        u = np.exp(sigma * np.sqrt(delta_t))
        d = 1.0 / u

        discount_curve = self.parent.discount_curve
        forward_rates = discount_curve.step_forward_rates(time_grid)

        dividend_curve = self.parent.underlying.dividend_curve
        if dividend_curve is not None:
            dividend_forwards = dividend_curve.step_forward_rates(time_grid)
        else:
            dividend_forwards = np.zeros(num_steps, dtype=float)

        growth = np.exp((forward_rates - dividend_forwards) * delta_t)
        if np.any(growth <= d) or np.any(growth >= u):
            raise ArbitrageViolationError(
                "Arbitrage condition violated: d < exp((f-g)*dt) < u for at least one step"
            )

        p = (growth - d) / (u - d)

        spot_lattice = self._build_spot_lattice(
            num_steps=num_steps,
            time_intervals=time_intervals,
            up=u,
            down=d,
            discount_curve=discount_curve,
        )

        discount_factors = np.exp(-forward_rates * delta_t)
        return discount_factors, p, spot_lattice

    def _build_spot_lattice(
        self,
        *,
        num_steps: int,
        time_intervals: pd.DatetimeIndex,
        up: float,
        down: float,
        discount_curve,
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
                discrete_dividends,
                curve_date=time_intervals[0],
                end_date=time_intervals[-1],
                discount_curve=discount_curve,
            )
            spot = max(spot - pv_all, 0.0)

        lattice = spot * (up**up_pow) * (down**down_pow)

        if not discrete_dividends:
            return lattice

        pv_remaining = np.array(
            [
                pv_discrete_dividends(
                    discrete_dividends,
                    curve_date=time_intervals[0],
                    end_date=time_intervals[-1],
                    discount_curve=discount_curve,
                    start_date=t,
                )
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
            if self.parent.option_type is OptionType.CALL:
                return np.maximum(instrument_values - K, 0)
            return np.maximum(K - instrument_values, 0)

        payoff_fn = self.parent.spec.payoff  # type: ignore[union-attr]
        return payoff_fn(instrument_values)

    # ------------------------------------------------------------------
    # Tree Greeks (Hull Ch. 13)
    # ------------------------------------------------------------------

    def _tree_greeks_data(self) -> tuple[np.ndarray, np.ndarray, float]:
        """Return (option_lattice, spot_lattice, delta_t) for tree Greek extraction.

        Raises UnsupportedFeatureError for Asian or other non-vanilla specs
        whose ``solve()`` does not return a 2-D option lattice.
        """
        num_steps = int(self.binom_params.num_steps)
        _, _, spot_lattice = self._setup_binomial_parameters(num_steps)
        option_lattice = self.solve()  # type: ignore[attr-defined]
        if not isinstance(option_lattice, np.ndarray) or option_lattice.ndim != 2:
            raise UnsupportedFeatureError(
                "Tree Greeks are only available for vanilla binomial options, "
                "not Asian or other path-dependent specs."
            )
        T = calculate_year_fraction(self.parent.pricing_date, self.parent.maturity)
        dt = T / num_steps
        return option_lattice, spot_lattice, dt

    def delta(self) -> float:
        """Extract delta from the binomial tree (Hull Ch. 13).

        .. math::

            \\Delta = \\frac{f_u - f_d}{S_u - S_d}

        where :math:`f_u, f_d` are the option values and :math:`S_u, S_d`
        the spot prices at step 1.
        """
        f, S, _ = self._tree_greeks_data()
        return float((f[0, 1] - f[1, 1]) / (S[0, 1] - S[1, 1]))

    def gamma(self) -> float:
        """Extract gamma from the binomial tree (Hull Ch. 13).

        Uses the three nodes at step 2:

        .. math::

            \\Gamma = \\frac{\\Delta_+ - \\Delta_-}{h}

        where :math:`\\Delta_+ = (f_{uu}-f_{ud})/(S_{uu}-S_{ud})`,
        :math:`\\Delta_- = (f_{ud}-f_{dd})/(S_{ud}-S_{dd})`,
        and :math:`h = (S_{uu}-S_{dd})/2`.
        """
        if self.binom_params.num_steps < 2:
            raise ValidationError("Tree gamma requires num_steps >= 2.")
        f, S, _ = self._tree_greeks_data()
        delta_up = (f[0, 2] - f[1, 2]) / (S[0, 2] - S[1, 2])
        delta_down = (f[1, 2] - f[2, 2]) / (S[1, 2] - S[2, 2])
        h = (S[0, 2] - S[2, 2]) / 2.0
        return float((delta_up - delta_down) / h)

    def theta(self) -> float:
        """Extract theta from the binomial tree (Hull Ch. 13).

        .. math::

            \\Theta = \\frac{f_{ud} - f_0}{2\\Delta t}

        where :math:`f_{ud}` is the central node at step 2 and :math:`f_0`
        the root value.  Returned per **calendar day** (divided by 365).
        """
        if self.binom_params.num_steps < 2:
            raise ValidationError("Tree theta requires num_steps >= 2.")
        f, _, dt = self._tree_greeks_data()
        theta_per_year = (f[1, 2] - f[0, 0]) / (2.0 * dt)
        return float(theta_per_year / 365.0)


class _BinomialEuropeanValuation(_BinomialValuationBase):
    """Implementation of European option valuation using binomial tree."""

    def solve(self) -> np.ndarray:
        """Compute the option value lattice using a binomial tree."""
        num_steps = int(self.binom_params.num_steps)
        logger.debug("Binomial European num_steps=%d", num_steps)
        discount_factors, p, spot_lattice = self._setup_binomial_parameters(num_steps)

        option_lattice = np.zeros_like(spot_lattice)
        option_lattice[:, num_steps] = self._get_intrinsic_values(spot_lattice[:, num_steps])

        # Backward induction through the tree (time in columns)
        for t in range(num_steps - 1, -1, -1):
            continuation = (
                p[t] * option_lattice[: t + 1, t + 1]
                + (1 - p[t]) * option_lattice[1 : t + 2, t + 1]
            ) * discount_factors[t]
            option_lattice[: t + 1, t] = continuation

        return option_lattice

    def present_value(self) -> float:
        """Return PV using binomial tree method."""
        with log_timing(logger, "Binomial European present_value", self.binom_params.log_timings):
            option_value_matrix = self.solve()
        pv = option_value_matrix[0, 0]

        return float(pv)


class _BinomialAmericanValuation(_BinomialValuationBase):
    """Implementation of American option valuation using binomial tree."""

    def solve(self) -> np.ndarray:
        """Compute the option value lattice using a binomial tree with early exercise."""
        num_steps = int(self.binom_params.num_steps)
        logger.debug("Binomial American num_steps=%d", num_steps)
        discount_factors, p, spot_lattice = self._setup_binomial_parameters(num_steps)

        option_lattice = np.zeros_like(spot_lattice)
        intrinsic = self._get_intrinsic_values(spot_lattice)
        option_lattice[:, num_steps] = intrinsic[:, num_steps]

        # Backward induction with early exercise decision
        for t in range(num_steps - 1, -1, -1):
            continuation = (
                p[t] * option_lattice[: t + 1, t + 1]
                + (1 - p[t]) * option_lattice[1 : t + 2, t + 1]
            ) * discount_factors[t]
            option_lattice[: t + 1, t] = np.maximum(intrinsic[: t + 1, t], continuation)

        return option_lattice

    def present_value(self) -> float:
        """Return PV using binomial tree method with American early exercise."""
        with log_timing(logger, "Binomial American present_value", self.binom_params.log_timings):
            option_value_matrix = self.solve()
        pv = option_value_matrix[0, 0]

        return float(pv)


class _BinomialAsianValuation(_BinomialValuationBase):
    """Asian option valuation using binomial MC sampling or Hull's representative averages."""

    def __init__(self, parent: OptionValuation) -> None:
        super().__init__(parent)
        self.spec: AsianOptionSpec = parent.spec  # type: ignore[assignment]

    def _solve_mc(self) -> np.ndarray:
        num_steps = int(self.binom_params.num_steps)
        logger.debug(
            "Binomial Asian MC num_steps=%d paths=%s", num_steps, self.binom_params.mc_paths
        )
        discount_factors, p, spot_lattice = self._setup_binomial_parameters(num_steps)

        if self.binom_params.mc_paths is None:
            raise ValidationError("BinomialParams.mc_paths must be set for Asian binomial MC")

        mc_paths = int(self.binom_params.mc_paths)
        rng = np.random.default_rng(self.binom_params.random_seed)

        K = self.parent.strike

        # Vectorized simulation of binomial paths:
        # draw Bernoulli down-steps for each path and step
        downs = rng.random((mc_paths, num_steps)) > p[None, :]  # (I,M)
        # cumulative count of downs gives row index at each step
        down_counts = np.cumsum(downs, axis=1)  # (I,M)
        # prepend time 0 (row=0)
        row_idx = np.concatenate(
            [np.zeros((mc_paths, 1), dtype=int), down_counts], axis=1
        )  # (I,M+1)
        col_idx: np.ndarray = np.arange(num_steps + 1, dtype=int)  # (M+1,)

        # gather prices along each simulated path
        # binomial_matrix is (M+1, M+1)
        # Advanced indexing: col_idx broadcasts to (I, M+1), so each
        # (row_idx[i, t], col_idx[t]) selects the node for path i at time t.
        prices = spot_lattice[row_idx, col_idx]  # (I, M+1)
        if self.spec.averaging is AsianAveraging.GEOMETRIC:
            if np.any(prices <= 0.0):
                raise NumericalError("Geometric averaging requires strictly positive path prices.")
            avg_s = np.exp(np.mean(np.log(prices), axis=1))  # (I,)
        else:
            avg_s = prices.mean(axis=1)  # (I,)

        if self.spec.call_put is OptionType.CALL:
            payoffs = np.maximum(avg_s - K, 0.0)
        else:
            payoffs = np.maximum(K - avg_s, 0.0)

        return payoffs * float(np.prod(discount_factors))

    def _average_payoff(self, avg_price: np.ndarray | float) -> np.ndarray:
        """Compute Asian option intrinsic payoff given average price(s)."""
        K = self.parent.strike
        if self.spec.call_put is OptionType.CALL:
            return np.maximum(avg_price - K, 0.0)
        return np.maximum(K - avg_price, 0.0)

    @staticmethod
    def _interp_value(x: float, grid: np.ndarray, values: np.ndarray) -> float:
        if grid[0] == grid[-1]:
            return float(values[0])
        return float(np.interp(x, grid, values))

    @staticmethod
    def _compute_ordering_bounds(
        lattice: np.ndarray, num_steps: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute min/max running-average bounds achievable at each node.

        Works for any additive state variable: pass spot values for arithmetic
        averaging, or log-spot values for geometric averaging.

        For each node (row, t), avg_min is the mean obtained when all down moves
        precede all up moves, and avg_max is the mean when all up moves come first.
        """
        avg_min = np.zeros_like(lattice)
        avg_max = np.zeros_like(lattice)

        for t in range(num_steps + 1):
            time_idx = np.arange(t + 1)
            row = np.arange(t + 1)[:, None]
            time = time_idx[None, :]

            downs_first_rows = np.minimum(time, row)
            prices_min = lattice[downs_first_rows, time]
            avg_min[: t + 1, t] = prices_min.mean(axis=1)

            ups_first: np.ndarray = t - row
            ups_first_rows = np.maximum(0, time - ups_first)
            prices_max = lattice[ups_first_rows, time]
            avg_max[: t + 1, t] = prices_max.mean(axis=1)

        return avg_min, avg_max

    def _solve_hull(self) -> tuple[np.ndarray, np.ndarray]:
        """Price an Asian option using Hull's representative-averages binomial tree.

        At each node, k linearly spaced average values between the attainable
        minimum and maximum are maintained. Backward induction interpolates
        continuation values from the child nodes' grids. For geometric
        averaging, all state variables are stored in log space.

        Returns
        =======
        tuple of (values, avg_grid)
            values : shape (k, num_steps+1, num_steps+1) — option values per average bucket
            avg_grid : shape (k, num_steps+1, num_steps+1) — representative averages
        """
        num_steps = int(self.binom_params.num_steps)
        logger.debug(
            "Binomial Asian Hull num_steps=%d averages=%s",
            num_steps,
            self.binom_params.asian_tree_averages,
        )
        discount_factors, p, spot_lattice = self._setup_binomial_parameters(num_steps)

        averaging = self.spec.averaging
        if averaging not in (AsianAveraging.ARITHMETIC, AsianAveraging.GEOMETRIC):
            raise ValidationError("Unsupported Asian averaging type for Hull binomial valuation.")

        if self.binom_params.asian_tree_averages is None:
            raise ValidationError(
                "BinomialParams.asian_tree_averages must be set for Hull binomial Asian valuation"
            )

        averaging_start = self.spec.averaging_start
        if averaging_start is not None and averaging_start != self.parent.pricing_date:
            raise ValidationError(
                "Hull binomial Asian valuation requires averaging_start=pricing_date."
            )

        k = int(self.binom_params.asian_tree_averages)
        if averaging is AsianAveraging.GEOMETRIC:
            if np.any(spot_lattice <= 0.0):
                raise NumericalError(
                    "Hull binomial geometric averaging requires strictly positive spot lattice."
                )
            log_spot_lattice = np.log(spot_lattice)
            avg_min, avg_max = self._compute_ordering_bounds(log_spot_lattice, num_steps)
        else:
            avg_min, avg_max = self._compute_ordering_bounds(spot_lattice, num_steps)

        avg_grid: np.ndarray = np.zeros((k, num_steps + 1, num_steps + 1), dtype=float)
        for t in range(num_steps + 1):
            rows = np.arange(t + 1)
            alpha = np.linspace(0.0, 1.0, k)[:, None]
            avg_grid[:, rows, t] = (
                avg_min[rows, t][None, :] + (avg_max[rows, t] - avg_min[rows, t])[None, :] * alpha
            )

        values = np.zeros_like(avg_grid)

        for row in range(num_steps + 1):
            maturity_avg = avg_grid[:, row, num_steps]
            if averaging is AsianAveraging.GEOMETRIC:
                maturity_avg = np.exp(maturity_avg)
            values[:, row, num_steps] = self._average_payoff(maturity_avg)

        # values[0,:,:] is option values corresponding to S_avg,min.
        # values[-1,:,:] is option values corresponding to S_avg,max.
        is_american = self.parent.spec.exercise_type is ExerciseType.AMERICAN

        for t in range(num_steps - 1, -1, -1):
            rows = np.arange(t + 1)
            if averaging is AsianAveraging.GEOMETRIC:
                s_up = log_spot_lattice[rows, t + 1]
                s_down = log_spot_lattice[rows + 1, t + 1]
            else:
                s_up = spot_lattice[rows, t + 1]
                s_down = spot_lattice[rows + 1, t + 1]
            grid_here = avg_grid[:, rows, t]  # shape (k,t+1)

            avg_up = ((t + 1) * grid_here + s_up) / (t + 2)
            avg_down = ((t + 1) * grid_here + s_down) / (t + 2)

            v_up = np.empty_like(avg_up)
            v_down = np.empty_like(avg_down)
            for j, row in enumerate(rows):  # type: ignore[assignment]
                v_up[:, j] = np.interp(avg_up[:, j], avg_grid[:, row, t + 1], values[:, row, t + 1])
                v_down[:, j] = np.interp(
                    avg_down[:, j], avg_grid[:, row + 1, t + 1], values[:, row + 1, t + 1]
                )

            continuation = discount_factors[t] * (p[t] * v_up + (1.0 - p[t]) * v_down)

            if is_american:
                exercise_avg = grid_here
                if averaging is AsianAveraging.GEOMETRIC:
                    exercise_avg = np.exp(exercise_avg)
                exercise = self._average_payoff(exercise_avg)
                values[:, rows, t] = np.maximum(continuation, exercise)
            else:
                values[:, rows, t] = continuation

        return values, avg_grid

    def present_value(self) -> float:
        with log_timing(logger, "Binomial Asian present_value", self.binom_params.log_timings):
            if self.binom_params.mc_paths is not None:
                pv_pathwise = self._solve_mc()
                return float(np.mean(pv_pathwise))

            values, _ = self._solve_hull()
            return float(values[0, 0, 0])

    def solve(self) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        if self.binom_params.mc_paths is not None:
            return self._solve_mc()
        return self._solve_hull()
