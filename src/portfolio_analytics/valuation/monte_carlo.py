"""Monte Carlo Simulation option valuation implementations."""

from typing import TYPE_CHECKING
import numpy as np

from ..enums import OptionType
from .params import MonteCarloParams

if TYPE_CHECKING:
    from .core import OptionValuation


class _MCEuropeanValuation:
    """Implementation of European option valuation using Monte Carlo."""

    def __init__(self, parent: "OptionValuation"):
        self.parent = parent

    def solve(self, params: MonteCarloParams) -> np.ndarray:
        """Generate undiscounted payoff vector at maturity (one value per path)."""
        paths = self.parent.underlying.get_instrument_values(random_seed=params.random_seed)
        time_grid = self.parent.underlying.time_grid

        # locate indices
        idx_end = np.where(time_grid == self.parent.maturity)[0]
        if idx_end.size == 0:
            raise ValueError("maturity not in underlying time_grid.")
        time_index_end = int(idx_end[0])

        maturity_value = paths[time_index_end]

        K = self.parent.strike
        if self.parent.option_type in (OptionType.CALL, OptionType.PUT):
            if K is None:
                raise ValueError("strike is required for vanilla European call/put payoff.")
            if self.parent.option_type is OptionType.CALL:
                return np.maximum(maturity_value - K, 0.0)
            return np.maximum(K - maturity_value, 0.0)

        payoff_fn = getattr(self.parent.spec, "payoff", None)
        if payoff_fn is None:
            raise ValueError("Unsupported option type for Monte Carlo valuation.")
        return payoff_fn(maturity_value)

    def present_value(self, params: MonteCarloParams) -> float:
        """Return the scalar present value."""
        pv_pathwise = self.present_value_pathwise(params)
        pv = np.mean(pv_pathwise)

        return float(pv)

    def present_value_pathwise(self, params: MonteCarloParams) -> np.ndarray:
        """Return discounted present values for each path."""
        cash_flow = self.solve(params)
        discount_factor = float(
            self.parent.discount_curve.get_discount_factors(
                (self.parent.pricing_date, self.parent.maturity)
            )[-1, 1]
        )
        return discount_factor * cash_flow


class _MCAmerianValuation:
    """Implementation of American option valuation using Longstaff-Schwartz Monte Carlo."""

    def __init__(self, parent: "OptionValuation"):
        self.parent = parent

    def solve(self, params: MonteCarloParams) -> tuple[np.ndarray, np.ndarray, int, int]:
        """Generate underlying paths and intrinsic payoff matrix over time.

        Parameters
        ==========
        random_seed: int, optional
            random seed for path generation

        Returns
        =======
        tuple of (instrument_values, payoff, time_index_start, time_index_end)
        """
        paths = self.parent.underlying.get_instrument_values(random_seed=params.random_seed)
        time_grid = self.parent.underlying.time_grid
        # locate indices
        idx_start = np.where(time_grid == self.parent.pricing_date)[0]
        idx_end = np.where(time_grid == self.parent.maturity)[0]
        if idx_start.size == 0:
            raise ValueError("Pricing date not in underlying time_grid.")
        if idx_end.size == 0:
            raise ValueError("maturity not in underlying time_grid.")

        time_index_start = int(idx_start[0])
        time_index_end = int(idx_end[0])

        instrument_values = paths[time_index_start : time_index_end + 1]

        K = self.parent.strike
        if self.parent.option_type in (OptionType.CALL, OptionType.PUT):
            if K is None:
                raise ValueError("strike is required for vanilla American call/put payoff.")
            if self.parent.option_type is OptionType.CALL:
                payoff = np.maximum(instrument_values - K, 0)
            else:
                payoff = np.maximum(K - instrument_values, 0)
        else:
            payoff_fn = self.parent.spec.payoff
            payoff = payoff_fn(instrument_values)

        return instrument_values, payoff, time_index_start, time_index_end

    def present_value(self, params: MonteCarloParams) -> float:
        """Calculate PV using Longstaff-Schwartz regression method."""
        pv_pathwise = self.present_value_pathwise(params)
        pv = np.mean(pv_pathwise)
        return float(pv)

    def present_value_pathwise(self, params: MonteCarloParams) -> np.ndarray:
        """Return discounted present values for each path (LSM output at pricing date)."""
        instrument_values, intrinsic_values, time_index_start, time_index_end = self.solve(params)
        time_list = self.parent.underlying.time_grid[time_index_start : time_index_end + 1]
        discount_factors = self.parent.discount_curve.get_discount_factors(
            time_list, dtobjects=True
        )
        V = np.zeros_like(intrinsic_values)
        V[-1] = intrinsic_values[-1]
        for t in range(len(time_list) - 2, 0, -1):
            discount_factor = discount_factors[t + 1, 1] / discount_factors[t, 1]
            itm = intrinsic_values[t] > 0
            S_itm = instrument_values[t][itm]
            V_itm = discount_factor * V[t + 1][itm]
            if len(S_itm) > 0:
                coefficients = np.polyfit(S_itm, V_itm, deg=params.deg)
            else:
                coefficients = np.zeros(params.deg + 1)
            predicted_cv = np.zeros_like(instrument_values[t])
            predicted_cv[itm] = np.polyval(coefficients, instrument_values[t][itm])
            V[t] = np.where(
                intrinsic_values[t] > predicted_cv,
                intrinsic_values[t],
                discount_factor * V[t + 1],
            )

        discount_factor_0 = discount_factors[1, 1] / discount_factors[0, 1]
        return discount_factor_0 * V[1]
