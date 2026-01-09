from dataclasses import dataclass
import datetime as dt
import numpy as np
from .stochastic_processes import PathSimulation
from .enums import OptionType, ExerciseType, PricingMethod
from .valuation_mcs import _MCEuropeanValuation, _MCAmerianValuation
from .valuation_binomial import _BinomialEuropeanValuation, _BinomialAmericanValuation


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

    Routes to appropriate pricing method based on PricingMethod parameter.

    Attributes
    ==========
    name: str
        Name of the valuation object/trade.
    underlying: PathSimulation
        Stochastic process simulator for the underlying risk factor.
    spec: OptionSpec
        Contract terms (type, exercise type, strike, maturity, currency, contract_size).
    pricing_method: PricingMethod
        Valuation methodology to use (Monte Carlo, Binomial, BSM, etc).
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

    def __init__(
        self,
        name: str,
        underlying: PathSimulation,
        spec: OptionSpec,
        pricing_method: PricingMethod,
    ):
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
        self.pricing_method = pricing_method
        self.contract_size = spec.contract_size

        # Optional sanity check: maturity must be after pricing date
        if self.maturity <= self.pricing_date:
            raise ValueError("Option maturity must be after pricing_date.")
        # provide pricing_date and maturity to underlying
        self.underlying.special_dates.extend([self.pricing_date, self.maturity])

        # Validate pricing_method
        if not isinstance(pricing_method, PricingMethod):
            raise TypeError(
                f"pricing_method must be PricingMethod enum, got {type(pricing_method).__name__}"
            )

        # Dispatch to appropriate pricing method implementation
        if pricing_method == PricingMethod.MONTE_CARLO:
            if spec.exercise_type == ExerciseType.EUROPEAN:
                self._impl = _MCEuropeanValuation(self)
            elif spec.exercise_type == ExerciseType.AMERICAN:
                self._impl = _MCAmerianValuation(self)
            else:
                raise ValueError(f"Unknown exercise type: {spec.exercise_type}")
        elif pricing_method == PricingMethod.BINOMIAL:
            if spec.exercise_type == ExerciseType.EUROPEAN:
                self._impl = _BinomialEuropeanValuation(self)
            elif spec.exercise_type == ExerciseType.AMERICAN:
                self._impl = _BinomialAmericanValuation(self)
            else:
                raise ValueError(f"Unknown exercise type: {spec.exercise_type}")
        else:
            raise ValueError(f"Pricing method {pricing_method} not yet implemented")

    def generate_payoff(self, **kwargs):
        """Generate payoff at maturity for the derivative.

        Parameters
        ==========
        **kwargs:
            Method-specific parameters:
            - MCS: random_seed (int, optional)
            - Binomial: num_steps (int, optional)
        """
        return self._impl.generate_payoff(**kwargs)

    def present_value(self, *, full: bool = False, **kwargs) -> float | tuple[float, np.ndarray]:
        """Calculate present value of the derivative.

        Parameters
        ==========
        full: bool
            Return full result with values at all nodes/paths
        **kwargs:
            Method-specific parameters:
            - MCS: random_seed (int, optional), deg (int, optional, American only)
            - Binomial: num_steps (int, optional)
        """
        return self._impl.present_value(full=full, **kwargs)

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
