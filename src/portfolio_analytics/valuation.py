from dataclasses import dataclass
from enum import Enum
import datetime as dt
import numpy as np
from .stochastic_processes import PathSimulation


class OptionType(Enum):
    CALL = "call"
    PUT = "put"


class ExerciseType(Enum):
    EUROPEAN = "european"
    AMERICAN = "american"


@dataclass(frozen=True, slots=True)
class OptionSpec:
    """Contract specification for a vanilla option."""

    option_type: OptionType  # CALL / PUT
    exercise_type: ExerciseType  # EUROPEAN / AMERICAN
    strike: float | None  # allow None for strike-less products
    maturity: dt.datetime
    currency: str
    contract_size: int | float = 100

    def __post_init__(self):
        """Validate option_type and exercise_type are valid enums."""
        if not isinstance(self.option_type, OptionType):
            raise TypeError(
                f"option_type must be OptionType enum, got {type(self.option_type).__name__}"
            )
        if not isinstance(self.exercise_type, ExerciseType):
            raise TypeError(
                f"exercise_type must be ExerciseType enum, got {type(self.exercise_type).__name__}"
            )


class OptionValuation:
    """Single-factor option valuation dispatcher.

    Attributes
    ==========
    name: str
        Name of the valuation object/trade.
    underlying: PathSimulation
        Stochastic process simulator for the underlying risk factor.
    spec: OptionSpec
        Contract terms (type, strike, maturity, currency, contract_size).
    pricing_date: datetime
        Pricing date (taken from MarketData via underlying).
    discount_curve:
        Discount curve used for valuation (taken from MarketData).

    Methods
    =======
    present_value:
        Returns the present value of the derivative.
    delta:
        Numerical delta.
    vega:
        Numerical vega.
    """

    def __init__(self, name: str, underlying: PathSimulation, spec: OptionSpec):
        self.name = name
        self.underlying = underlying
        self.spec = spec

        # Pricing date + discount curve come from the underlying's MarketData
        self.pricing_date = underlying.pricing_date
        self.discount_curve = underlying.discount_curve

        # Convenience aliases
        self.maturity = spec.maturity
        self.strike = spec.strike
        self.currency = spec.currency
        self.option_type = spec.option_type
        self.exercise_type = spec.exercise_type
        self.contract_size = spec.contract_size

        # Optional sanity check: maturity must be after pricing date
        if self.maturity <= self.pricing_date:
            raise ValueError("Option maturity must be after pricing_date.")
        # provide pricing_date and maturity to underlying
        self.underlying.special_dates.extend([self.pricing_date, self.maturity])

        # Dispatch to appropriate implementation
        if spec.exercise_type == ExerciseType.EUROPEAN:
            self._impl = _EuropeanValuation(self)
        elif spec.exercise_type == ExerciseType.AMERICAN:
            self._impl = _AmericanValuation(self)
        else:
            raise ValueError(f"Unknown exercise type: {spec.exercise_type}")

    def generate_payoff(self, random_seed: int | None = None):
        """Generate payoff at maturity for the derivative."""
        return self._impl.generate_payoff(random_seed)

    def present_value(
        self, random_seed: int | None = None, full: bool = False, **kwargs
    ) -> float | tuple[float, np.ndarray]:
        """Calculate present value of the derivative."""
        return self._impl.present_value(random_seed, full, **kwargs)

    def delta(self, epsilon: float | None = None, random_seed: int | None = None):
        """Calculate option delta using central difference approximation."""
        if epsilon is None:
            epsilon = self.underlying.initial_value / 100
        # central difference approximation
        initial_spot = self.underlying.initial_value
        try:
            # calculate left value for numerical Delta
            self.underlying.initial_value -= epsilon
            value_left = self.present_value(random_seed=random_seed)
            # numerical underlying value for right value
            self.underlying.initial_value += 2 * epsilon
            # calculate right value for numerical delta
            value_right = self.present_value(random_seed=random_seed)
        finally:
            # reset the initial_value of the simulation object
            self.underlying.initial_value = initial_spot

        delta = (value_right - value_left) / (2 * epsilon)
        # correct for potential numerical errors
        if delta < -1.0:
            return -1.0
        if delta > 1.0:
            return 1.0
        return delta

    def vega(self, epsilon: float = 0.01, random_seed: int | None = None):
        """Calculate option vega using central difference approximation."""
        epsilon = max(epsilon, self.underlying.volatility / 50.0)
        # central-difference approximation
        initial_vol = self.underlying.volatility
        try:
            # calculate the left value for numerical Vega
            self.underlying.volatility -= epsilon
            value_left = self.present_value(random_seed=random_seed)
            # numerical volatility value for right value
            # update the simulation object
            self.underlying.volatility += 2 * epsilon
            # calculate the right value for numerical Vega
            value_right = self.present_value(random_seed=random_seed)
        finally:
            # reset volatility value of simulation object
            self.underlying.volatility = initial_vol

        vega = (value_right - value_left) / (2 * epsilon)
        return vega


class _EuropeanValuation:
    """Implementation of European option valuation using Monte Carlo."""

    def __init__(self, parent: OptionValuation):
        self.parent = parent

    def generate_payoff(self, random_seed: int | None = None) -> np.ndarray:
        """Generate payoff vector at maturity (one value per path)."""
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
        random_seed: int | None = None,
        full: bool = False,
        **kwargs,
    ) -> float | tuple[float, np.ndarray]:
        """Return PV (and optionally pathwise discounted PVs)."""
        cash_flow = self.generate_payoff(random_seed=random_seed)

        # discount factor from pricing_date to maturity
        discount_factor = self.parent.discount_curve.get_discount_factors(
            (self.parent.pricing_date, self.parent.maturity)
        )[-1, 1]

        pv_pathwise = discount_factor * cash_flow
        pv = np.mean(pv_pathwise)

        if full:
            return pv, pv_pathwise
        return pv


class _AmericanValuation:
    """Implementation of American option valuation using Longstaff-Schwartz Monte Carlo."""

    def __init__(self, parent: OptionValuation):
        self.parent = parent

    def generate_payoff(
        self, random_seed: int | None = None
    ) -> tuple[np.ndarray, np.ndarray, int, int]:
        """Generate payoff paths and indices.

        Parameters
        ==========
        random_seed: int, optional
            random seed for path generation

        Returns
        =======
        tuple of (instrument_values, payoff, time_index_start, time_index_end)
        """
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
        random_seed: int | None = None,
        full: bool = False,
        deg: int = 2,
        **kwargs,
    ) -> tuple[float, np.ndarray] | float:
        """Calculate PV using Longstaff-Schwartz regression method.

        Parameters
        ==========
        random_seed: int, optional
            random seed for path generation
        full: bool
            return also full 1d array of present values
        deg: int
            degree of polynomial for regression

        Returns
        =======
        float or tuple of (pv, pathwise_discounted_values)
        """
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
