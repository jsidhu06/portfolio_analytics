"""Monte Carlo Simulation option valuation implementations."""

import numpy as np
from .enums import OptionType


class _MCEuropeanValuation:
    """Implementation of European option valuation using Monte Carlo."""

    def __init__(self, parent):
        self.parent = parent

    def generate_payoff(self, **kwargs) -> np.ndarray:
        """Generate payoff vector at maturity (one value per path)."""
        random_seed = kwargs.get("random_seed")
        paths = self.parent.underlying.get_instrument_values(random_seed=random_seed)
        time_grid = self.parent.underlying.time_grid

        # locate indices
        idx_end = np.where(time_grid == self.parent.maturity)[0]
        if idx_end.size == 0:
            raise ValueError("maturity not in underlying time_grid.")
        time_index_end = int(idx_end[0])

        maturity_value = paths[time_index_end]

        K = self.parent.strike
        if K is None:
            raise ValueError("strike is required for vanilla European call/put payoff.")

        if self.parent.option_type is OptionType.CALL:
            payoff = np.maximum(maturity_value - K, 0.0)
        else:
            payoff = np.maximum(K - maturity_value, 0.0)

        return payoff

    def present_value(
        self,
        full: bool = False,
        **kwargs,
    ) -> float | tuple[float, np.ndarray]:
        """Return PV (and optionally pathwise discounted PVs)."""
        random_seed = kwargs.get("random_seed")
        cash_flow = self.generate_payoff(random_seed=random_seed)

        # discount factor from pricing_date to maturity
        discount_factor = float(
            self.parent.discount_curve.get_discount_factors(
                (self.parent.pricing_date, self.parent.maturity)
            )[-1, 1]
        )

        pv_pathwise = discount_factor * cash_flow
        pv = np.mean(pv_pathwise)

        if full:
            return pv, pv_pathwise
        return pv


class _MCAmerianValuation:
    """Implementation of American option valuation using Longstaff-Schwartz Monte Carlo."""

    def __init__(self, parent):
        self.parent = parent

    def generate_payoff(self, **kwargs) -> tuple[np.ndarray, np.ndarray, int, int]:
        """Generate payoff paths and indices.

        Parameters
        ==========
        random_seed: int, optional
            random seed for path generation

        Returns
        =======
        tuple of (instrument_values, payoff, time_index_start, time_index_end)
        """
        random_seed = kwargs.get("random_seed")
        paths = self.parent.underlying.get_instrument_values(random_seed=random_seed)
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
        if K is None:
            raise ValueError("strike is required for vanilla American call/put payoff.")

        if self.parent.option_type is OptionType.CALL:
            payoff = np.maximum(instrument_values - K, 0)
        else:
            payoff = np.maximum(K - instrument_values, 0)

        return instrument_values, payoff, time_index_start, time_index_end

    def present_value(
        self,
        full: bool = False,
        deg: int = 2,
        **kwargs,
    ) -> tuple[float, np.ndarray] | float:
        """Calculate PV using Longstaff-Schwartz regression method.

        Parameters
        ==========
        full: bool
            return also full 1d array of present values
        deg: int
            degree of polynomial for regression
        **kwargs:
            random_seed: int, optional
                random seed for path generation

        Returns
        =======
        float or tuple of (pv, pathwise_discounted_values)
        """
        random_seed = kwargs.get("random_seed")
        instrument_values, intrinsic_values, time_index_start, time_index_end = (
            self.generate_payoff(random_seed=random_seed)
        )
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
                coefficients = np.polyfit(S_itm, V_itm, deg=deg)
            else:
                coefficients = np.zeros(deg + 1)
            predicted_cv = np.zeros_like(instrument_values[t])
            predicted_cv[itm] = np.polyval(coefficients, instrument_values[t][itm])
            V[t] = np.where(
                intrinsic_values[t] > predicted_cv,
                intrinsic_values[t],
                discount_factor * V[t + 1],
            )

        discount_factor = discount_factors[1, 1] / discount_factors[0, 1]
        result = discount_factor * np.mean(V[1])
        if full:
            return result, discount_factor * np.mean(V[1])
        return result
