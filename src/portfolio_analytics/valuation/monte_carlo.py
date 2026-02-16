"""Monte Carlo Simulation option valuation implementations."""

from typing import TYPE_CHECKING
import logging
import numpy as np

from ..utils import calculate_year_fraction, log_timing

from ..enums import OptionType, AsianAveraging
from .params import MonteCarloParams

if TYPE_CHECKING:
    from .core import OptionValuation


logger = logging.getLogger(__name__)


def _warn_if_high_std_error(
    *,
    pv_pathwise: np.ndarray,
    pv_mean: float,
    params: MonteCarloParams,
    label: str,
) -> None:
    """Emit a warning log if MC standard error is high relative to the PV estimate."""
    if params.std_error_warn_ratio is None:
        return
    n_paths = pv_pathwise.size
    if n_paths < 2:
        return
    std_error = float(np.std(pv_pathwise, ddof=1) / np.sqrt(n_paths))
    scale = max(abs(pv_mean), 1.0e-12)
    ratio = std_error / scale
    logger.debug(
        "MC %s std_error=%.6g ratio=%.6g paths=%d",
        label,
        std_error,
        ratio,
        n_paths,
    )
    if ratio > params.std_error_warn_ratio:
        logger.warning(
            "MC %s standard error high: std_error=%.6g ratio=%.6g (>%.3g) paths=%d",
            label,
            std_error,
            ratio,
            params.std_error_warn_ratio,
            n_paths,
        )


def _find_time_index(time_grid: np.ndarray, target, label: str) -> int:
    """Return the index of *target* in *time_grid*, raising ValueError if absent."""
    idx = np.where(time_grid == target)[0]
    if idx.size == 0:
        raise ValueError(f"{label} not in underlying time_grid.")
    return int(idx[0])


def _vanilla_payoff(option_type: OptionType, strike: float, spot: np.ndarray) -> np.ndarray:
    """Vectorized vanilla payoff: max(S-K,0) for calls, max(K-S,0) for puts."""
    if option_type is OptionType.CALL:
        return np.maximum(spot - strike, 0.0)
    return np.maximum(strike - spot, 0.0)


def _year_fractions(pricing_date, dates: np.ndarray) -> np.ndarray:
    return np.array(
        [calculate_year_fraction(pricing_date, d) for d in dates],
        dtype=float,
    )


class _MCEuropeanValuation:
    """Implementation of European option valuation using Monte Carlo."""

    def __init__(self, parent: "OptionValuation") -> None:
        self.parent = parent

    def solve(self) -> np.ndarray:
        """Generate undiscounted payoff vector at maturity (one value per path)."""
        params = self.parent.params
        if not isinstance(params, MonteCarloParams):
            raise TypeError("Monte Carlo valuation requires MonteCarloParams on OptionValuation")
        paths = self.parent.underlying.get_instrument_values(random_seed=params.random_seed)
        time_grid = self.parent.underlying.time_grid

        time_index_end = _find_time_index(time_grid, self.parent.maturity, "maturity")
        maturity_value = paths[time_index_end]

        K = self.parent.strike
        if self.parent.option_type in (OptionType.CALL, OptionType.PUT):
            if K is None:
                raise ValueError("strike is required for vanilla European call/put payoff.")
            return _vanilla_payoff(self.parent.option_type, K, maturity_value)

        payoff_fn = getattr(self.parent.spec, "payoff", None)
        if payoff_fn is None:
            raise ValueError("Unsupported option type for Monte Carlo valuation.")
        return payoff_fn(maturity_value)

    def present_value(self) -> float:
        """Return the scalar present value."""
        params = self.parent.params
        if not isinstance(params, MonteCarloParams):
            raise TypeError("Monte Carlo valuation requires MonteCarloParams on OptionValuation")
        with log_timing(logger, "MC European present_value", params.log_timings):
            pv_pathwise = self.present_value_pathwise()
            pv = float(np.mean(pv_pathwise))
        logger.debug(
            "MC European paths=%d time_steps=%d",
            pv_pathwise.size,
            len(self.parent.underlying.time_grid) - 1,
        )
        _warn_if_high_std_error(
            pv_pathwise=pv_pathwise,
            pv_mean=pv,
            params=params,
            label="European",
        )
        return pv

    def present_value_pathwise(self) -> np.ndarray:
        """Return discounted present values for each path."""
        payoff_vector = self.solve()
        ttm = float(calculate_year_fraction(self.parent.pricing_date, self.parent.maturity))
        discount_factor = float(self.parent.discount_curve.df(ttm))
        return discount_factor * payoff_vector


class _MCAmericanValuation:
    """Implementation of American option valuation using Longstaff-Schwartz Monte Carlo."""

    def __init__(self, parent: "OptionValuation") -> None:
        self.parent = parent

    def solve(self) -> tuple[np.ndarray, np.ndarray, int, int]:
        """Generate underlying paths and intrinsic payoff matrix over time.

        Parameters
        ==========
        random_seed: int, optional
            random seed for path generation

        Returns
        =======
        tuple of (instrument_values, payoff, time_index_start, time_index_end)
        """
        params = self.parent.params
        if not isinstance(params, MonteCarloParams):
            raise TypeError("Monte Carlo valuation requires MonteCarloParams on OptionValuation")
        paths = self.parent.underlying.get_instrument_values(random_seed=params.random_seed)
        time_grid = self.parent.underlying.time_grid
        time_index_start = _find_time_index(time_grid, self.parent.pricing_date, "Pricing date")
        time_index_end = _find_time_index(time_grid, self.parent.maturity, "maturity")

        instrument_values = paths[time_index_start : time_index_end + 1]

        K = self.parent.strike
        if self.parent.option_type in (OptionType.CALL, OptionType.PUT):
            if K is None:
                raise ValueError("strike is required for vanilla American call/put payoff.")
            payoff = _vanilla_payoff(self.parent.option_type, K, instrument_values)
        else:
            payoff_fn = self.parent.spec.payoff
            payoff = payoff_fn(instrument_values)

        return instrument_values, payoff, time_index_start, time_index_end

    def present_value(self) -> float:
        """Calculate PV using Longstaff-Schwartz regression method."""
        params = self.parent.params
        if not isinstance(params, MonteCarloParams):
            raise TypeError("Monte Carlo valuation requires MonteCarloParams on OptionValuation")
        with log_timing(logger, "MC American present_value", params.log_timings):
            pv_pathwise = self.present_value_pathwise()
            pv = float(np.mean(pv_pathwise))
        logger.debug(
            "MC American paths=%d time_steps=%d deg=%d",
            pv_pathwise.size,
            len(self.parent.underlying.time_grid) - 1,
            params.deg,
        )
        _warn_if_high_std_error(
            pv_pathwise=pv_pathwise,
            pv_mean=pv,
            params=params,
            label="American",
        )
        return pv

    def present_value_pathwise(self) -> np.ndarray:
        """Return discounted present values for each path (LSM output at pricing date)."""
        instrument_values, intrinsic_values, time_index_start, time_index_end = self.solve()
        time_list = self.parent.underlying.time_grid[time_index_start : time_index_end + 1]
        t_grid = _year_fractions(self.parent.pricing_date, time_list)
        discount_factors = self.parent.discount_curve.df(t_grid)
        values = np.zeros_like(intrinsic_values)
        values[-1] = intrinsic_values[-1]

        params = self.parent.params
        if not isinstance(params, MonteCarloParams):
            raise TypeError("Monte Carlo valuation requires MonteCarloParams on OptionValuation")

        for t in range(len(time_list) - 2, 0, -1):
            df_step = discount_factors[t + 1] / discount_factors[t]
            itm = intrinsic_values[t] > 0

            continuation = np.zeros_like(instrument_values[t])
            if np.any(itm):
                S_itm = instrument_values[t][itm]
                V_itm = df_step * values[t + 1][itm]
                poly = np.polynomial.Polynomial.fit(S_itm, V_itm, deg=params.deg)
                continuation[itm] = poly(instrument_values[t][itm])

            values[t] = np.where(
                intrinsic_values[t] > continuation,
                intrinsic_values[t],
                df_step * values[t + 1],
            )

        df0 = discount_factors[1] / discount_factors[0]
        return df0 * values[1]


class _MCAsianValuation:
    """Implementation of Asian option valuation using Monte Carlo.

    Asian options are path-dependent options where the payoff depends on the average
    price of the underlying over the averaging period.
    """

    def __init__(self, parent: "OptionValuation") -> None:
        self.parent = parent

    def solve(self) -> np.ndarray:
        """Generate undiscounted payoff vector based on path averages.

        Returns
        -------
        np.ndarray
            Payoff for each path based on the average spot price
        """
        params = self.parent.params
        if not isinstance(params, MonteCarloParams):
            raise TypeError("Monte Carlo valuation requires MonteCarloParams on OptionValuation")
        paths = self.parent.underlying.get_instrument_values(random_seed=params.random_seed)
        time_grid = self.parent.underlying.time_grid

        # Determine averaging period
        spec = self.parent.spec
        averaging_start = spec.averaging_start if spec.averaging_start else self.parent.pricing_date

        # Find indices for averaging period
        idx_start = np.where(time_grid == averaging_start)[0]
        idx_end = np.where(time_grid == self.parent.maturity)[0]

        if idx_start.size == 0:
            raise ValueError(f"averaging_start {averaging_start} not in underlying time_grid.")
        if idx_end.size == 0:
            raise ValueError(f"maturity {self.parent.maturity} not in underlying time_grid.")

        time_index_start = int(idx_start[0])
        time_index_end = int(idx_end[0])

        # Extract paths over averaging period (inclusive)
        averaging_paths = paths[time_index_start : time_index_end + 1, :]

        # Calculate average for each path
        if spec.averaging is AsianAveraging.ARITHMETIC:
            # Arithmetic average: (1/N) * Σ S_i
            avg_prices = np.mean(averaging_paths, axis=0)
        elif spec.averaging is AsianAveraging.GEOMETRIC:
            # Geometric average: (Π S_i)^(1/N)
            # Use log space for numerical stability: exp(mean(log(S_i)))
            if np.any(averaging_paths <= 0.0):
                raise ValueError("Geometric averaging requires strictly positive path prices.")
            log_prices = np.log(averaging_paths)
            avg_prices = np.exp(np.mean(log_prices, axis=0))
        else:
            raise ValueError(f"Unsupported averaging method for Asian valuation: {spec.averaging}")

        # Calculate payoff based on average
        K = self.parent.strike
        if K is None:
            raise ValueError("strike is required for Asian option payoff.")

        # Asian call: max(S_avg - K, 0)
        # Asian put: max(K - S_avg, 0)
        if self.parent.spec.call_put is OptionType.CALL:
            return np.maximum(avg_prices - K, 0.0)
        return np.maximum(K - avg_prices, 0.0)

    def present_value(self) -> float:
        """Return the scalar present value."""
        params = self.parent.params
        if not isinstance(params, MonteCarloParams):
            raise TypeError("Monte Carlo valuation requires MonteCarloParams on OptionValuation")
        with log_timing(logger, "MC Asian present_value", params.log_timings):
            pv_pathwise = self.present_value_pathwise()
            pv = float(np.mean(pv_pathwise))
        logger.debug(
            "MC Asian paths=%d time_steps=%d",
            pv_pathwise.size,
            len(self.parent.underlying.time_grid) - 1,
        )
        _warn_if_high_std_error(
            pv_pathwise=pv_pathwise,
            pv_mean=pv,
            params=params,
            label="Asian",
        )
        return pv

    def present_value_pathwise(self) -> np.ndarray:
        """Return discounted present values for each path."""
        payoff_vector = self.solve()
        ttm = float(calculate_year_fraction(self.parent.pricing_date, self.parent.maturity))
        discount_factor = float(self.parent.discount_curve.df(ttm))
        return discount_factor * payoff_vector
