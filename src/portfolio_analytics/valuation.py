from dataclasses import dataclass
from collections.abc import Callable
import datetime as dt
import numpy as np
from .stochastic_processes import PathSimulation
from .enums import (
    OptionType,
    ExerciseType,
    PricingMethod,
    GreekCalculationMethod,
)
from .valuation_mcs import _MCEuropeanValuation, _MCAmerianValuation
from .valuation_binomial import _BinomialEuropeanValuation, _BinomialAmericanValuation
from .valuation_bsm import _BSMEuropeanValuation
from .valuation_pde_fd import _FDAmericanValuation
from .rates import ConstantShortRate
from .market_environment import MarketData


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


@dataclass(frozen=True, slots=True)
class PayoffSpec:
    """Contract specification for a single-contract custom payoff.

    This is useful for pricing payoffs that are not representable as a single vanilla
    call/put (e.g., capped combinations), while still treating the product as ONE
    contract for exercise decisions (American pricing compares intrinsic vs continuation
    on the full payoff).

    Notes
    -----
    - payoff_fn must be vectorized over spot (accept float or np.ndarray and return np.ndarray)
    - strike is kept as None for compatibility with the OptionValuation interface
    """

    exercise_type: ExerciseType
    maturity: dt.datetime
    currency: str
    payoff_fn: Callable[[np.ndarray | float], np.ndarray]
    contract_size: int | float = 100

    # Kept for compatibility with vanilla valuation interfaces
    option_type: OptionType = OptionType.CUSTOM
    strike: None = None

    def __post_init__(self):
        if not isinstance(self.exercise_type, ExerciseType):
            raise TypeError(
                f"exercise_type must be ExerciseType enum, got {type(self.exercise_type).__name__}"
            )
        if not isinstance(self.option_type, OptionType) or self.option_type != OptionType.CUSTOM:
            raise TypeError("PayoffSpec.option_type must be OptionType.CUSTOM")
        if not callable(self.payoff_fn):
            raise TypeError("payoff_fn must be callable")

    def payoff(self, spot: np.ndarray | float) -> np.ndarray:
        """Vectorized payoff as a function of spot."""
        # Ensure a float ndarray output (for downstream math and boolean comparisons).
        return np.asarray(self.payoff_fn(spot), dtype=float)


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
    jump_intensity: float | None = None
    jump_mean: float | None = None
    jump_std: float | None = None
    # Optional SRD (Square Root Diffusion / CIR) parameters
    kappa: float | None = None
    theta: float | None = None

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name.strip():
            raise TypeError("UnderlyingConfig.name must be a non-empty string")
        if not isinstance(self.model, str) or not self.model.strip():
            raise TypeError("UnderlyingConfig.model must be a non-empty string")

        allowed_models = {"gbm", "jd", "srd"}
        if self.model not in allowed_models:
            raise ValueError(
                f"UnderlyingConfig.model must be one of {sorted(allowed_models)}, got '{self.model}'"
            )

        if self.initial_value is None:
            raise ValueError("UnderlyingConfig.initial_value must be not None")
        if self.volatility is None:
            raise ValueError("UnderlyingConfig.volatility must be not None")

        try:
            self.initial_value = float(self.initial_value)
            self.volatility = float(self.volatility)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                "UnderlyingConfig.initial_value and volatility must be numeric"
            ) from exc

        if not np.isfinite(self.initial_value):
            raise ValueError("UnderlyingConfig.initial_value must be finite")
        if not np.isfinite(self.volatility):
            raise ValueError("UnderlyingConfig.volatility must be finite")
        if self.volatility < 0.0:
            raise ValueError("UnderlyingConfig.volatility must be >= 0")

        # Model-specific validation
        if self.model == "jd":
            missing = [
                name
                for name, value in (
                    ("jump_intensity", self.jump_intensity),
                    ("jump_mean", self.jump_mean),
                    ("jump_std", self.jump_std),
                )
                if value is None
            ]
            if missing:
                raise ValueError(
                    f"UnderlyingConfig for model='jd' requires {', '.join(missing)} to be set"
                )
            self.jump_intensity = float(self.jump_intensity)
            self.jump_mean = float(self.jump_mean)
            self.jump_std = float(self.jump_std)
            if not np.isfinite(self.jump_intensity):
                raise ValueError("UnderlyingConfig.jump_intensity must be finite")
            if not np.isfinite(self.jump_mean):
                raise ValueError("UnderlyingConfig.jump_mean must be finite")
            if not np.isfinite(self.jump_std):
                raise ValueError("UnderlyingConfig.jump_std must be finite")
            if self.jump_intensity < 0.0:
                raise ValueError("UnderlyingConfig.jump_intensity must be >= 0")
            if self.jump_std < 0.0:
                raise ValueError("UnderlyingConfig.jump_std must be >= 0")

        if self.model == "srd":
            missing = [
                name
                for name, value in (("kappa", self.kappa), ("theta", self.theta))
                if value is None
            ]
            if missing:
                raise ValueError(
                    f"UnderlyingConfig for model='srd' requires {', '.join(missing)} to be set"
                )
            self.kappa = float(self.kappa)
            self.theta = float(self.theta)
            if not np.isfinite(self.kappa):
                raise ValueError("UnderlyingConfig.kappa must be finite")
            if not np.isfinite(self.theta):
                raise ValueError("UnderlyingConfig.theta must be finite")
            if self.kappa < 0.0:
                raise ValueError("UnderlyingConfig.kappa must be >= 0")
            if self.theta < 0.0:
                raise ValueError("UnderlyingConfig.theta must be >= 0")


class UnderlyingPricingData:
    """Minimal data container for option valuation underlying asset.

    Used when pricing with methods that don't require full stochastic process simulation
    (e.g., binomial trees). Contains only essential parameters: spot price, volatility,
    pricing date, discount curve, and continuous dividend yield (optional, default 0.0).
    """

    def __init__(
        self,
        initial_value: float,
        volatility: float,
        market_data: MarketData,
        dividend_yield: float = 0.0,
    ):
        self.initial_value = initial_value
        self.volatility = volatility
        self.market_data = market_data
        self.dividend_yield = dividend_yield

    @property
    def pricing_date(self) -> dt.datetime:
        return self.market_data.pricing_date

    @property
    def discount_curve(self) -> ConstantShortRate:
        return self.market_data.discount_curve

    @property
    def currency(self) -> str:
        return self.market_data.currency


class OptionValuation:
    """Single-factor option valuation dispatcher.

    Routes to appropriate pricing method based on PricingMethod parameter.

    Attributes
    ==========
    name: str
        Name of the valuation object/trade.
    underlying: PathSimulation | UnderlyingPricingData
        Stochastic process simulator (Monte Carlo) or minimal data container (BSM, Binomial).
        For Monte Carlo: must be PathSimulation instance.
        For BSM and Binomial: UnderlyingPricingData.
    spec: OptionSpec | PayoffSpec
        Contract terms (type, exercise type, strike, maturity, currency, contract_size)
        for either a vanilla option or a custom single-contract payoff.
    pricing_method: PricingMethod
        Valuation methodology to use (Monte Carlo, BSM, Binomial, etc).
    pricing_date: datetime
        Pricing date (taken from underlying).
    discount_curve:
        Discount curve used for valuation (taken from underlying).

    Methods
    =======
    present_value:
        Returns the present value of the derivative.
    delta:
        Calculate option delta. For BSM: defaults to analytical closed-form formula,
        can be overridden with greek_calc_method kwarg.
    gamma:
        Calculate option gamma. For BSM: defaults to analytical closed-form formula,
        can be overridden with greek_calc_method kwarg.
    vega:
        Calculate option vega. For BSM: defaults to analytical closed-form formula,
        can be overridden with greek_calc_method kwarg.
    """

    def __init__(
        self,
        name: str,
        underlying: PathSimulation | UnderlyingPricingData,
        spec: OptionSpec | PayoffSpec,
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

        # Strategy guardrails
        if pricing_method == PricingMethod.BSM_CONTINUOUS and self.option_type not in (
            OptionType.CALL,
            OptionType.PUT,
        ):
            raise NotImplementedError(
                "BSM pricing is only available for vanilla CALL/PUT option types."
            )

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
        elif pricing_method == PricingMethod.BSM_CONTINUOUS:
            if spec.exercise_type == ExerciseType.EUROPEAN:
                self._impl = _BSMEuropeanValuation(self)
            elif spec.exercise_type == ExerciseType.AMERICAN:
                raise NotImplementedError(
                    "BSM is only applicable to European option valuation. "
                    "Select a different pricing method for American options such as Binomial or Monte Carlo."
                )
            else:
                raise ValueError(f"Unknown exercise type: {spec.exercise_type}")
        elif pricing_method == PricingMethod.PDE_FD:
            if spec.exercise_type != ExerciseType.AMERICAN:
                raise NotImplementedError(
                    "PDE_FD pricing is currently implemented only for AMERICAN exercise."
                )
            if self.option_type not in (OptionType.CALL, OptionType.PUT):
                raise NotImplementedError(
                    "PDE_FD pricing is only available for vanilla CALL/PUT option types."
                )
            if self.strike is None:
                raise ValueError("strike is required for PDE_FD valuation")
            self._impl = _FDAmericanValuation(self)
        else:
            raise ValueError(f"Pricing method {pricing_method.name} not yet implemented")

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

    def delta(
        self,
        epsilon: float | None = None,
        greek_calc_method: GreekCalculationMethod | None = None,
        **kwargs,
    ) -> float:
        """Calculate option delta.

        For BSM pricing method:
        - Uses analytical closed-form formula by default
        - Uses numerical central difference approximation if greek_calc_method=GreekCalculationMethod.NUMERICAL

        For other pricing methods:
        - Only support numerical central difference approximation (greek_calc_method analytical
            raises ValueError)

        Parameters
        ==========
        epsilon: float, optional
            Step size for numerical approximation (only used when greek_calc_method=NUMERICAL)
        greek_calc_method: GreekCalculationMethod, optional
            Method for calculating delta. Defaults to ANALYTICAL for BSM, NUMERICAL for others.
            - GreekCalculationMethod.ANALYTICAL: Use closed-form formula (BSM only)
            - GreekCalculationMethod.NUMERICAL: Use central difference approximation
        **kwargs:
            Method-specific parameters for present_value calculation

        Returns
        =======
        float
            option delta
        """
        # Determine which method to use
        if greek_calc_method is None:
            # Default to analytical for BSM, numerical for others
            use_analytical = self.pricing_method == PricingMethod.BSM_CONTINUOUS
        else:
            if not isinstance(greek_calc_method, GreekCalculationMethod):
                raise TypeError(
                    f"greek_calc_method must be GreekCalculationMethod enum, got {type(greek_calc_method).__name__}"
                )
            use_analytical = (
                greek_calc_method == GreekCalculationMethod.ANALYTICAL
                and self.pricing_method == PricingMethod.BSM_CONTINUOUS
            )
            # Validate that analytical is only used with BSM
            if (
                greek_calc_method == GreekCalculationMethod.ANALYTICAL
                and self.pricing_method != PricingMethod.BSM_CONTINUOUS
            ):
                raise ValueError(
                    "Analytical greeks are only available for BSM_CONTINUOUS pricing method. "
                    "Use greek_calc_method=GreekCalculationMethod.NUMERICAL for other pricing methods."
                )

        # Use analytical formula for BSM if specified
        if use_analytical:
            return self._impl.delta(**kwargs)

        # Otherwise use numerical approximation
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

    def gamma(
        self,
        epsilon: float | None = None,
        greek_calc_method: GreekCalculationMethod | None = None,
        **kwargs,
    ) -> float:
        """Calculate option gamma.

        For BSM pricing method:
        - Uses analytical closed-form formula by default
        - Uses numerical central difference approximation if greek_calc_method=GreekCalculationMethod.NUMERICAL

        For other pricing methods:
        - Always uses numerical central difference approximation (greek_calc_method ignored)

        Parameters
        ==========
        epsilon: float, optional
            Step size for numerical approximation (only used when greek_calc_method=NUMERICAL)
        greek_calc_method: GreekCalculationMethod, optional
            Method for calculating gamma. Defaults to ANALYTICAL for BSM, NUMERICAL for others.
            - GreekCalculationMethod.ANALYTICAL: Use closed-form formula (BSM only)
            - GreekCalculationMethod.NUMERICAL: Use central difference approximation
        **kwargs:
            Method-specific parameters for present_value calculation

        Returns
        =======
        float
            option gamma
        """
        # Determine which method to use
        if greek_calc_method is None:
            # Default to analytical for BSM, numerical for others
            use_analytical = self.pricing_method == PricingMethod.BSM_CONTINUOUS
        else:
            if not isinstance(greek_calc_method, GreekCalculationMethod):
                raise TypeError(
                    f"greek_calc_method must be GreekCalculationMethod enum, got {type(greek_calc_method).__name__}"
                )
            use_analytical = (
                greek_calc_method == GreekCalculationMethod.ANALYTICAL
                and self.pricing_method == PricingMethod.BSM_CONTINUOUS
            )
            # Validate that analytical is only used with BSM
            if (
                greek_calc_method == GreekCalculationMethod.ANALYTICAL
                and self.pricing_method != PricingMethod.BSM_CONTINUOUS
            ):
                raise ValueError(
                    "Analytical greeks are only available for BSM_CONTINUOUS pricing method. "
                    "Use greek_calc_method=GreekCalculationMethod.NUMERICAL for other pricing methods."
                )

        # Use analytical formula for BSM if specified
        if use_analytical:
            return self._impl.gamma(**kwargs)

        # Otherwise use numerical approximation
        if epsilon is None:
            epsilon = self.underlying.initial_value / 100
        # central-difference approximation
        initial_spot = self.underlying.initial_value
        try:
            # calculate left value for numerical Gamma
            self.underlying.initial_value -= epsilon
            value_left = self.present_value(**kwargs)
            # numerical underlying value for right value
            self.underlying.initial_value += 2 * epsilon
            # calculate right value for numerical gamma
            value_right = self.present_value(**kwargs)
            # reset to initial spot for center value
            self.underlying.initial_value = initial_spot
            value_center = self.present_value(**kwargs)
        finally:
            # reset the initial_value of the simulation object
            self.underlying.initial_value = initial_spot

        gamma = (value_right - 2 * value_center + value_left) / (epsilon**2)
        return gamma

    def vega(
        self,
        epsilon: float = 0.01,
        greek_calc_method: GreekCalculationMethod | None = None,
        **kwargs,
    ) -> float:
        """Calculate option vega.

        For BSM pricing method:
        - Uses analytical closed-form formula by default
        - Uses numerical central difference approximation if greek_calc_method=GreekCalculationMethod.NUMERICAL

        For other pricing methods:
        - Always uses numerical central difference approximation (greek_calc_method ignored)

        Parameters
        ==========
        epsilon: float, default 0.01
            Step size for numerical approximation (only used when greek_calc_method=NUMERICAL)
        greek_calc_method: GreekCalculationMethod, optional
            Method for calculating vega. Defaults to ANALYTICAL for BSM, NUMERICAL for others.
            - GreekCalculationMethod.ANALYTICAL: Use closed-form formula (BSM only)
            - GreekCalculationMethod.NUMERICAL: Use central difference approximation
        **kwargs:
            Method-specific parameters for present_value calculation

        Returns
        =======
        float
            option vega
        """
        # Determine which method to use
        if greek_calc_method is None:
            # Default to analytical for BSM, numerical for others
            use_analytical = self.pricing_method == PricingMethod.BSM_CONTINUOUS
        else:
            if not isinstance(greek_calc_method, GreekCalculationMethod):
                raise TypeError(
                    f"greek_calc_method must be GreekCalculationMethod enum, got {type(greek_calc_method).__name__}"
                )
            use_analytical = (
                greek_calc_method == GreekCalculationMethod.ANALYTICAL
                and self.pricing_method == PricingMethod.BSM_CONTINUOUS
            )
            # Validate that analytical is only used with BSM
            if (
                greek_calc_method == GreekCalculationMethod.ANALYTICAL
                and self.pricing_method != PricingMethod.BSM_CONTINUOUS
            ):
                raise ValueError(
                    "Analytical greeks are only available for BSM_CONTINUOUS pricing method. "
                    "Use greek_calc_method=GreekCalculationMethod.NUMERICAL for other pricing methods."
                )

        # Use analytical formula for BSM if specified
        if use_analytical:
            return self._impl.vega(**kwargs)

        # Otherwise use numerical approximation
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

        vega = (value_right - value_left) / (2 * epsilon) / 100  # per 1% point change in vol
        return vega
