from dataclasses import dataclass
import datetime as dt
import numpy as np
from .stochastic_processes import PathSimulation
from .enums import OptionType, ExerciseType, PricingMethod
from .valuation_mcs import _MCEuropeanValuation, _MCAmerianValuation
from .valuation_binomial import _BinomialEuropeanValuation, _BinomialAmericanValuation
from .rates import ConstantShortRate


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


@dataclass
class UnderlyingConfig:
    """Configuration for an underlying asset in portfolio simulations.

    Specifies the stochastic process model and its parameters for a particular
    underlying asset. Used to instantiate PathSimulation objects in portfolio context.

    Attributes
    ==========
    name: str
        Name/identifier of the underlying asset (e.g., 'STOCK', 'INDEX')
    model: str
        Stochastic process model type: 'gbm', 'jd', or 'srd'
    initial_value: float
        Initial spot price or rate
    volatility: float
        Volatility of the process
    jump_intensity: float
        Jump intensity lambda (for 'jd' model only)
    jump_mean: float
        Mean of jump size log returns (for 'jd' model only)
    jump_std: float
        Standard deviation of jump size log returns (for 'jd' model only)
    kappa: float
        Mean reversion speed (for 'srd' model only)
    theta: float
        Long-run mean level (for 'srd' model only)
    """

    name: str
    model: str  # 'gbm', 'jd', 'srd'
    initial_value: float
    volatility: float
    # Optional JD (Jump Diffusion) parameters
    jump_intensity: float = 0.0
    jump_mean: float = 0.0
    jump_std: float = 0.0
    # Optional SRD (Square Root Diffusion / CIR) parameters
    kappa: float = 0.1
    theta: float = 0.05


class UnderlyingData:
    """Minimal data container for option valuation underlying asset.

    Used when pricing with methods that don't require full stochastic process simulation
    (e.g., binomial trees). Contains only essential parameters: spot price, volatility,
    pricing date, and discount curve.
    """

    def __init__(
        self,
        initial_value: float,
        volatility: float,
        pricing_date: dt.datetime,
        discount_curve: ConstantShortRate,
    ):
        self.initial_value = initial_value
        self.volatility = volatility
        self.pricing_date = pricing_date
        self.discount_curve = discount_curve
        self.special_dates = []  # For compatibility with valuation methods


class OptionValuation:
    """Single-factor option valuation dispatcher.

    Routes to appropriate pricing method based on PricingMethod parameter.

    Attributes
    ==========
    name: str
        Name of the valuation object/trade.
    underlying: PathSimulation | UnderlyingData
        Stochastic process simulator (Monte Carlo) or minimal data container (Binomial).
        For Monte Carlo: must be PathSimulation instance.
        For Binomial: can be PathSimulation or UnderlyingData.
    spec: OptionSpec
        Contract terms (type, exercise type, strike, maturity, currency, contract_size).
    pricing_method: PricingMethod
        Valuation methodology to use (Monte Carlo, Binomial, BSM, etc).
    pricing_date: datetime
        Pricing date (taken from underlying).
    discount_curve:
        Discount curve used for valuation (taken from underlying).

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
        underlying: PathSimulation | UnderlyingData,
        spec: OptionSpec,
        pricing_method: PricingMethod,
    ):
        self.name = name
        self.underlying = underlying
        self.spec = spec

        # Pricing date + discount curve come from the underlying
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

        # Only add special dates for PathSimulation (Monte Carlo)
        if isinstance(underlying, PathSimulation):
            underlying.special_dates.extend([self.pricing_date, self.maturity])

        # Validate pricing_method
        if not isinstance(pricing_method, PricingMethod):
            raise TypeError(
                f"pricing_method must be PricingMethod enum, got {type(pricing_method).__name__}"
            )

        # Validate that MC requires PathSimulation
        if pricing_method == PricingMethod.MONTE_CARLO and not isinstance(
            underlying, PathSimulation
        ):
            raise TypeError(
                "Monte Carlo pricing requires underlying to be a PathSimulation instance"
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

    def delta(self, epsilon: float | None = None, **kwargs) -> float:
        """Calculate option delta using central difference approximation."""
        if epsilon is None:
            epsilon = self.underlying.initial_value / 100
        # central difference approximation
        initial_spot = self.underlying.initial_value
        try:
            # calculate left value for numerical Delta
            self.underlying.initial_value -= epsilon
            value_left = self.present_value(**kwargs)
            # numerical underlying value for right value
            self.underlying.initial_value += 2 * epsilon
            # calculate right value for numerical delta
            value_right = self.present_value(**kwargs)
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

    def vega(self, epsilon: float = 0.01, **kwargs) -> float:
        """Calculate option vega using central difference approximation."""
        # central-difference approximation
        initial_vol = self.underlying.volatility
        try:
            # calculate the left value for numerical Vega
            self.underlying.volatility -= epsilon
            value_left = self.present_value(**kwargs)
            # numerical volatility value for right value
            # update the simulation object
            self.underlying.volatility += 2 * epsilon
            # calculate the right value for numerical Vega
            value_right = self.present_value(**kwargs)
        finally:
            # reset volatility value of simulation object
            self.underlying.volatility = initial_vol

        vega = (value_right - value_left) / (2 * epsilon)
        return vega
