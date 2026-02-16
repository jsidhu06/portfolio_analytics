from dataclasses import dataclass, replace as dc_replace
from collections.abc import Callable
import datetime as dt
import numpy as np
from ..stochastic_processes import PathSimulation
from ..enums import (
    OptionType,
    AsianAveraging,
    ExerciseType,
    PricingMethod,
    GreekCalculationMethod,
)
from .monte_carlo import _MCEuropeanValuation, _MCAmericanValuation, _MCAsianValuation
from .binomial import (
    _BinomialEuropeanValuation,
    _BinomialAmericanValuation,
    _BinomialAsianValuation,
)
from .bsm import _BSMEuropeanValuation
from .pde import _FDEuropeanValuation, _FDAmericanValuation
from ..rates import DiscountCurve
from ..utils import calculate_year_fraction
from ..market_environment import MarketData
from .params import BinomialParams, MonteCarloParams, PDEParams, ValuationParams

# ── Implementation registries ───────────────────────────────────────
# Maps (PricingMethod, ExerciseType) → implementation class for vanilla specs.
_VANILLA_REGISTRY: dict[tuple[PricingMethod, ExerciseType], type] = {
    (PricingMethod.MONTE_CARLO, ExerciseType.EUROPEAN): _MCEuropeanValuation,
    (PricingMethod.MONTE_CARLO, ExerciseType.AMERICAN): _MCAmericanValuation,
    (PricingMethod.BINOMIAL, ExerciseType.EUROPEAN): _BinomialEuropeanValuation,
    (PricingMethod.BINOMIAL, ExerciseType.AMERICAN): _BinomialAmericanValuation,
    (PricingMethod.BSM, ExerciseType.EUROPEAN): _BSMEuropeanValuation,
    (PricingMethod.PDE_FD, ExerciseType.EUROPEAN): _FDEuropeanValuation,
    (PricingMethod.PDE_FD, ExerciseType.AMERICAN): _FDAmericanValuation,
}

# Maps PricingMethod → implementation class for Asian option specs.
_ASIAN_REGISTRY: dict[PricingMethod, type] = {
    PricingMethod.MONTE_CARLO: _MCAsianValuation,
    PricingMethod.BINOMIAL: _BinomialAsianValuation,
}


@dataclass(frozen=True, slots=True)
class OptionSpec:
    """Contract specification for a vanilla option."""

    option_type: OptionType  # CALL / PUT
    exercise_type: ExerciseType  # EUROPEAN / AMERICAN
    strike: float
    maturity: dt.datetime
    currency: str
    contract_size: int | float = 100

    def __post_init__(self) -> None:
        """Validate option_type/exercise_type and coerce strike."""
        if not isinstance(self.option_type, OptionType):
            raise TypeError(
                f"option_type must be OptionType enum, got {type(self.option_type).__name__}"
            )
        if self.option_type not in (OptionType.CALL, OptionType.PUT):
            raise ValueError("OptionSpec.option_type must be OptionType.CALL or OptionType.PUT")
        if not isinstance(self.exercise_type, ExerciseType):
            raise TypeError(
                f"exercise_type must be ExerciseType enum, got {type(self.exercise_type).__name__}"
            )

        if self.strike is None:
            raise ValueError("OptionSpec.strike must be provided")
        try:
            strike = float(self.strike)
        except (TypeError, ValueError) as exc:
            raise TypeError("OptionSpec.strike must be numeric") from exc
        if not np.isfinite(strike):
            raise ValueError("OptionSpec.strike must be finite")
        if strike < 0.0:
            raise ValueError("OptionSpec.strike must be >= 0")
        object.__setattr__(self, "strike", strike)


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
    strike: None = None

    def __post_init__(self) -> None:
        if not isinstance(self.exercise_type, ExerciseType):
            raise TypeError(
                f"exercise_type must be ExerciseType enum, got {type(self.exercise_type).__name__}"
            )
        if not callable(self.payoff_fn):
            raise TypeError("payoff_fn must be callable")

    def payoff(self, spot: np.ndarray | float) -> np.ndarray:
        """Vectorized payoff as a function of spot."""
        # Ensure a float ndarray output (for downstream math and boolean comparisons).
        return np.asarray(self.payoff_fn(spot), dtype=float)


@dataclass(frozen=True, slots=True)
class AsianOptionSpec:
    """Contract specification for an Asian option.

    Asian options are path-dependent options where the payoff depends on the average
    price of the underlying over a specified averaging period.

    Parameters
    ----------
    averaging : AsianAveraging
        AsianAveraging.ARITHMETIC or AsianAveraging.GEOMETRIC
    call_put : OptionType
        OptionType.CALL or OptionType.PUT to specify payoff direction
    strike : float
        Strike price
    maturity : dt.datetime
        Option maturity date
    currency : str
        Currency denomination
    averaging_start : dt.datetime, optional
        Start of averaging period. If None, uses pricing date.
    exercise_type : ExerciseType
        Exercise style (EUROPEAN or AMERICAN). Default: EUROPEAN.
    contract_size : int | float
        Contract multiplier (default 100)

    Notes
    -----
    - Arithmetic average: S_avg = (1/N) * Σ S_i
    - Geometric average: S_avg = (Π S_i)^(1/N)
    - Payoff for call: max(S_avg - K, 0)
    - Payoff for put: max(K - S_avg, 0)
    - European and American exercise are supported depending on pricing method
    """

    averaging: AsianAveraging
    call_put: OptionType  # CALL or PUT
    strike: float
    maturity: dt.datetime
    currency: str
    averaging_start: dt.datetime | None = None
    contract_size: int | float = 100
    exercise_type: ExerciseType = ExerciseType.EUROPEAN

    def __post_init__(self) -> None:
        """Validate Asian option specification."""
        if not isinstance(self.averaging, AsianAveraging):
            raise TypeError(
                f"averaging must be AsianAveraging enum, got {type(self.averaging).__name__}"
            )
        if not isinstance(self.exercise_type, ExerciseType):
            raise TypeError(
                f"exercise_type must be ExerciseType enum, got {type(self.exercise_type).__name__}"
            )

        if not isinstance(self.call_put, OptionType):
            raise TypeError(f"call_put must be OptionType enum, got {type(self.call_put).__name__}")
        if self.call_put not in (OptionType.CALL, OptionType.PUT):
            raise ValueError("AsianOptionSpec.call_put must be OptionType.CALL or OptionType.PUT")

        if self.strike is None:
            raise ValueError("AsianOptionSpec.strike must be provided")
        try:
            strike = float(self.strike)
        except (TypeError, ValueError) as exc:
            raise TypeError("AsianOptionSpec.strike must be numeric") from exc
        if not np.isfinite(strike):
            raise ValueError("AsianOptionSpec.strike must be finite")
        if strike < 0.0:
            raise ValueError("AsianOptionSpec.strike must be >= 0")
        object.__setattr__(self, "strike", strike)


class UnderlyingPricingData:
    """Minimal data container for option valuation underlying asset.

    Used when pricing with methods that don't require full stochastic process simulation
    (e.g., BSM, binomial trees, FD approximation to PDE).
    Contains only essential parameters: spot price, volatility, pricing date, discount curve,
    continuous dividend yield via dividend_curve,
    and optional discrete dividends as (ex_date, amount) pairs.
    """

    def __init__(
        self,
        initial_value: float,
        volatility: float,
        market_data: MarketData,
        discrete_dividends: list[tuple[dt.datetime, float]] | None = None,
        dividend_curve: DiscountCurve | None = None,
    ) -> None:
        self.initial_value = initial_value
        self.volatility = volatility
        self.market_data = market_data
        self.dividend_curve = dividend_curve
        if discrete_dividends is None:
            self.discrete_dividends = []
        else:
            cleaned: list[tuple[dt.datetime, float]] = []
            for ex_date, amount in discrete_dividends:
                if not isinstance(ex_date, dt.datetime):
                    raise TypeError("discrete_dividends entries must be (datetime, amount) tuples")
                try:
                    amt = float(amount)
                except (TypeError, ValueError) as exc:
                    raise TypeError("dividend amount must be numeric") from exc
                cleaned.append((ex_date, amt))
            self.discrete_dividends = sorted(cleaned, key=lambda x: x[0])

        if self.dividend_curve is not None and self.discrete_dividends:
            raise ValueError("Provide either dividend_curve or discrete_dividends, not both.")

    @property
    def pricing_date(self) -> dt.datetime:
        return self.market_data.pricing_date

    @property
    def discount_curve(self) -> DiscountCurve:
        return self.market_data.discount_curve

    @property
    def currency(self) -> str:
        return self.market_data.currency

    def replace(self, **kwargs) -> "UnderlyingPricingData":
        """Create a new UnderlyingPricingData instance with modified fields.

        This is used for bump-and-revalue calculations (e.g., Greeks) without
        mutating the original object, making it thread-safe and explicit.

        Parameters
        ----------
        **kwargs
            Fields to override (initial_value, volatility, dividend_curve,
            discrete_dividends, market_data)

        Returns
        -------
        UnderlyingPricingData
            New instance with specified fields replaced
        """
        allowed_fields = {
            "initial_value",
            "volatility",
            "market_data",
            "discrete_dividends",
            "dividend_curve",
        }
        unknown = set(kwargs).difference(allowed_fields)
        if unknown:
            raise ValueError(
                f"replace() only supports overriding {sorted(allowed_fields)}; "
                f"unknown: {sorted(unknown)}"
            )

        return UnderlyingPricingData(
            initial_value=kwargs.get("initial_value", self.initial_value),
            volatility=kwargs.get("volatility", self.volatility),
            market_data=kwargs.get("market_data", self.market_data),
            discrete_dividends=kwargs.get("discrete_dividends", self.discrete_dividends),
            dividend_curve=kwargs.get("dividend_curve", self.dividend_curve),
        )


class OptionValuation:
    """Single-factor option valuation dispatcher.

    Routes to appropriate pricing method based on PricingMethod parameter.

    Attributes
    ==========
    name: str
        Name of the valuation object/trade.
    underlying: PathSimulation | UnderlyingPricingData
        Stochastic process simulator (Monte Carlo) or minimal data container (BSM, Binomial, PDE).
        For Monte Carlo: must be PathSimulation instance.
        For BSM, Binomial, PDE: UnderlyingPricingData.
    spec: OptionSpec | PayoffSpec | AsianOptionSpec
        Contract terms (type, exercise type, strike, maturity, currency, contract_size)
        for a vanilla option, a custom single-contract payoff, or an Asian option.
    pricing_method: PricingMethod
        Valuation methodology to use (Monte Carlo, BSM, Binomial, FD approximation to PDE, etc).
    pricing_date: datetime
        Pricing date (taken from underlying).
    discount_curve:
        Discount curve used for valuation (taken from underlying).

    Methods
    =======
    present_value:
        Returns the present value of the derivative.
    present_value_pathwise:
        Returns the present value of the derivative for each MC simulated path (only applicable
        when PricingMethod is MONTE_CARLO).
    delta:
        Calculate option delta. For BSM: defaults to analytical closed-form formula,
        can be overridden with greek_calc_method kwarg.
    gamma:
        Calculate option gamma. For BSM: defaults to analytical closed-form formula,
        can be overridden with greek_calc_method kwarg.
    vega:
        Calculate option vega. For BSM: defaults to analytical closed-form formula,
        can be overridden with greek_calc_method kwarg.
    theta:
        Calculate option theta. For BSM, defaults to analytical closed-form formula,
         can be overridden with greek_calc_method kwarg.
    rho:
        Calculate option rho. For BSM, defaults to analytical closed-form formula,
            can be overridden with greek_calc_method kwarg.
    """

    def __init__(
        self,
        name: str,
        underlying: PathSimulation | UnderlyingPricingData,
        spec: OptionSpec | PayoffSpec | AsianOptionSpec,
        pricing_method: PricingMethod,
        params: ValuationParams | None = None,
    ) -> None:
        self.name = name
        self.underlying = underlying
        self.spec = spec

        # Pricing date + discount curve come from the underlying's market data
        self.pricing_date = underlying.pricing_date
        self.discount_curve = underlying.discount_curve

        # Convenience aliases
        self.maturity = spec.maturity
        self.strike = spec.strike
        self.currency = spec.currency
        if hasattr(spec, "option_type") and isinstance(spec.option_type, OptionType):
            self.option_type = spec.option_type
        elif hasattr(spec, "call_put") and isinstance(spec.call_put, OptionType):
            self.option_type = spec.call_put
        else:
            self.option_type = None

        if self.currency != underlying.currency:
            raise NotImplementedError(
                "Cross-currency valuation is not supported. "
                "Option currency must match the underlying market currency."
            )
        self.exercise_type = spec.exercise_type
        self.pricing_method = pricing_method
        self.contract_size = spec.contract_size

        self.params: ValuationParams | None = self._validate_and_default_params(
            pricing_method=pricing_method, params=params
        )

        # Strategy guardrails
        if pricing_method == PricingMethod.BSM and self.option_type not in (
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
            for d in (self.pricing_date, self.maturity):
                underlying.special_dates.add(d)

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

        # Validate that deterministic methods don't receive PathSimulation
        # (they only need UnderlyingPricingData; paths would be ignored)
        if pricing_method in (
            PricingMethod.BINOMIAL,
            PricingMethod.BSM,
            PricingMethod.PDE_FD,
        ) and isinstance(underlying, PathSimulation):
            raise TypeError(
                f"{pricing_method.name} pricing does not use stochastic path simulation. "
                "Pass an UnderlyingPricingData instance instead of PathSimulation."
            )
        if pricing_method not in (
            PricingMethod.BSM,
            PricingMethod.BINOMIAL,
            PricingMethod.PDE_FD,
            PricingMethod.MONTE_CARLO,
        ):
            if self.discount_curve.flat_rate is None:
                raise NotImplementedError(
                    "Time-varying discount curves are only supported for BSM, BINOMIAL, PDE_FD and MONTE_CARLO."
                )
            if getattr(self.underlying, "dividend_curve", None) is not None:
                raise NotImplementedError(
                    "Time-varying dividend curves are only supported for BSM, BINOMIAL, PDE_FD, and MONTE_CARLO."
                )
        # Dispatch to appropriate pricing method implementation
        if isinstance(spec, AsianOptionSpec):
            if (
                spec.exercise_type is ExerciseType.AMERICAN
                and pricing_method != PricingMethod.BINOMIAL
            ):
                raise ValueError(
                    "American Asian options are only supported with BINOMIAL (Hull tree)."
                )
            impl_cls = _ASIAN_REGISTRY.get(pricing_method)
            if impl_cls is None:
                raise ValueError(
                    "Asian options are path-dependent and require MONTE_CARLO or BINOMIAL pricing."
                )
        else:
            impl_cls = _VANILLA_REGISTRY.get((pricing_method, spec.exercise_type))
            if impl_cls is None:
                if pricing_method is PricingMethod.BSM:
                    raise NotImplementedError(
                        "BSM is only applicable to European option valuation. "
                        "Select a different pricing method for American options such as Binomial, "
                        "PDE_FD or MONTE_CARLO."
                    )
                raise ValueError(
                    f"{pricing_method.name} does not support {spec.exercise_type.name} exercise."
                )
        self._impl = impl_cls(self)

    @staticmethod
    def _validate_and_default_params(
        *,
        pricing_method: PricingMethod,
        params: ValuationParams | None,
    ) -> ValuationParams | None:
        if params is None:
            if pricing_method == PricingMethod.MONTE_CARLO:
                return MonteCarloParams()
            if pricing_method == PricingMethod.BINOMIAL:
                return BinomialParams()
            if pricing_method == PricingMethod.PDE_FD:
                return PDEParams()
            return None

        if pricing_method == PricingMethod.MONTE_CARLO:
            if not isinstance(params, MonteCarloParams):
                raise TypeError("pricing_method=MONTE_CARLO requires params=MonteCarloParams")
            return params

        if pricing_method == PricingMethod.BINOMIAL:
            if not isinstance(params, BinomialParams):
                raise TypeError("pricing_method=BINOMIAL requires params=BinomialParams")
            return params

        if pricing_method == PricingMethod.PDE_FD:
            if not isinstance(params, PDEParams):
                raise TypeError("pricing_method=PDE_FD requires params=PDEParams")
            return params

        if params is not None:
            raise TypeError(
                f"pricing_method={pricing_method.name} does not accept valuation params"
            )
        return None

    def _effective_params(self) -> ValuationParams | None:
        return self._validate_and_default_params(
            pricing_method=self.pricing_method, params=self.params
        )

    def solve(
        self,
    ) -> float | np.ndarray | tuple[np.ndarray, np.ndarray] | tuple[float, np.ndarray, np.ndarray]:
        """Run the pricing method's core solver and return its raw output.

        This is intentionally method-specific:
        - Monte Carlo: pathwise payoff(s) / payoff matrix (undiscounted)
        - Binomial tree: option value lattice (node values)
        - PDE/FD: (pv, spot_grid, value_grid) at pricing time
        - BSM: scalar option value

        Use present_value_pathwise() for discounted pathwise outputs where supported.
        """
        self._effective_params()
        return self._impl.solve()

    def present_value(self) -> float:
        """Calculate present value of the derivative."""
        effective_params = self._effective_params()
        base_pv = float(self._impl.present_value())
        if effective_params is None or not getattr(
            effective_params, "control_variate_european", False
        ):
            return base_pv

        return float(self._apply_control_variate(base_pv))

    def _apply_control_variate(self, base_pv: float) -> float:
        if self.exercise_type is not ExerciseType.AMERICAN:
            raise ValueError("control_variate_european is only valid for American options.")
        if self.pricing_method not in (PricingMethod.BINOMIAL, PricingMethod.PDE_FD):
            raise NotImplementedError(
                "control_variate_european is only supported for BINOMIAL and PDE_FD pricing."
            )
        if not isinstance(self.spec, OptionSpec):
            raise NotImplementedError(
                "control_variate_european is only supported for vanilla option specs."
            )
        if self.option_type not in (OptionType.CALL, OptionType.PUT):
            raise NotImplementedError(
                "control_variate_european is only supported for vanilla CALL/PUT."
            )

        euro_spec = OptionSpec(
            option_type=self.option_type,
            exercise_type=ExerciseType.EUROPEAN,
            strike=self.strike,
            maturity=self.maturity,
            currency=self.currency,
            contract_size=self.contract_size,
        )

        params = self._effective_params()
        cv_params = (
            dc_replace(params, control_variate_european=False)
            if hasattr(params, "control_variate_european")
            else params
        )
        euro_num = OptionValuation(
            name=f"{self.name}_cv_euro_num",
            underlying=self.underlying,
            spec=euro_spec,
            pricing_method=self.pricing_method,
            params=cv_params,
        ).present_value()
        euro_bsm = OptionValuation(
            name=f"{self.name}_cv_euro_bsm",
            underlying=self.underlying,
            spec=euro_spec,
            pricing_method=PricingMethod.BSM,
        ).present_value()

        return base_pv + (euro_bsm - euro_num)

    def present_value_pathwise(self) -> np.ndarray:
        """Return discounted pathwise present values.

        Implemented for Monte Carlo pricing methods. For other pricing methods, this
        is not meaningful and will raise NotImplementedError.
        """
        pv_pathwise = getattr(self._impl, "present_value_pathwise", None)
        if pv_pathwise is None:
            raise NotImplementedError(
                "present_value_pathwise is only implemented for Monte Carlo valuation."
            )
        self._effective_params()
        return pv_pathwise()

    def _resolve_greek_method(
        self,
        greek_calc_method: GreekCalculationMethod | None,
        *,
        error_message: str,
    ) -> bool:
        """Return True for analytical greeks, validating method compatibility."""
        if greek_calc_method is None:
            return self.pricing_method == PricingMethod.BSM

        if not isinstance(greek_calc_method, GreekCalculationMethod):
            raise TypeError(
                f"greek_calc_method must be GreekCalculationMethod enum, got {type(greek_calc_method).__name__}"
            )

        if (
            greek_calc_method == GreekCalculationMethod.ANALYTICAL
            and self.pricing_method != PricingMethod.BSM
        ):
            raise ValueError(error_message)

        return greek_calc_method == GreekCalculationMethod.ANALYTICAL

    def _build_valuation(self, *, name_suffix: str, underlying) -> "OptionValuation":
        """Create a sibling valuation with a modified underlying for bump-and-revalue."""
        return OptionValuation(
            name=f"{self.name}_{name_suffix}",
            underlying=underlying,
            spec=self.spec,
            pricing_method=self.pricing_method,
            params=self.params,
        )

    def delta(
        self,
        epsilon: float | None = None,
        greek_calc_method: GreekCalculationMethod | None = None,
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
        Returns
        =======
        float
            option delta
        """
        use_analytical = self._resolve_greek_method(
            greek_calc_method,
            error_message=(
                "Analytical greeks are only available for BSM pricing method. "
                "Use greek_calc_method=GreekCalculationMethod.NUMERICAL for other pricing methods."
            ),
        )

        # Use analytical formula for BSM if specified
        if use_analytical:
            return self._impl.delta()

        # Otherwise use numerical approximation via bump-and-revalue
        if epsilon is None:
            epsilon = self.underlying.initial_value / 100

        # Create bumped underlyings
        underlying_down = self.underlying.replace(
            initial_value=self.underlying.initial_value - epsilon
        )
        underlying_up = self.underlying.replace(
            initial_value=self.underlying.initial_value + epsilon
        )
        val_down = self._build_valuation(name_suffix="delta_down", underlying=underlying_down)
        val_up = self._build_valuation(name_suffix="delta_up", underlying=underlying_up)

        # Calculate central difference
        value_left = val_down.present_value()
        value_right = val_up.present_value()

        delta = (value_right - value_left) / (2 * epsilon)

        return delta

    def gamma(
        self,
        epsilon: float | None = None,
        greek_calc_method: GreekCalculationMethod | None = None,
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
        Returns
        =======
        float
            option gamma
        """
        use_analytical = self._resolve_greek_method(
            greek_calc_method,
            error_message=(
                "Analytical greeks are only available for BSM pricing method. "
                "Use greek_calc_method=GreekCalculationMethod.NUMERICAL for other pricing methods."
            ),
        )

        # Use analytical formula for BSM if specified
        if use_analytical:
            return self._impl.gamma()

        # Otherwise use numerical approximation via bump-and-revalue
        if epsilon is None:
            epsilon = self.underlying.initial_value / 100

        # Create bumped underlyings
        underlying_down = self.underlying.replace(
            initial_value=self.underlying.initial_value - epsilon
        )
        underlying_up = self.underlying.replace(
            initial_value=self.underlying.initial_value + epsilon
        )

        val_down = self._build_valuation(name_suffix="gamma_down", underlying=underlying_down)
        val_up = self._build_valuation(name_suffix="gamma_up", underlying=underlying_up)

        # Calculate central difference (center uses self.present_value)
        value_left = val_down.present_value()
        value_right = val_up.present_value()
        value_center = self.present_value()

        gamma = (value_right - 2 * value_center + value_left) / (epsilon**2)
        return gamma

    def vega(
        self,
        epsilon: float = 0.01,
        greek_calc_method: GreekCalculationMethod | None = None,
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
        Returns
        =======
        float
            option vega

        Notes
        =====
        Vega is returned per 1% point change in volatility. The implied-vol solver
        scales this back to per-1.0 volatility when computing Newton steps.
        """
        use_analytical = self._resolve_greek_method(
            greek_calc_method,
            error_message=(
                "Analytical greeks are only available for BSM pricing method. "
                "Use greek_calc_method=GreekCalculationMethod.NUMERICAL for other pricing methods."
            ),
        )

        # Use analytical formula for BSM if specified
        if use_analytical:
            return self._impl.vega()

        # Otherwise use numerical approximation via bump-and-revalue
        # Create bumped underlyings
        underlying_down = self.underlying.replace(volatility=self.underlying.volatility - epsilon)
        underlying_up = self.underlying.replace(volatility=self.underlying.volatility + epsilon)

        val_down = self._build_valuation(name_suffix="vega_down", underlying=underlying_down)
        val_up = self._build_valuation(name_suffix="vega_up", underlying=underlying_up)

        # Calculate central difference
        value_left = val_down.present_value()
        value_right = val_up.present_value()

        vega = (value_right - value_left) / (2 * epsilon) / 100  # per 1% point change in vol
        return vega

    def theta(
        self,
        greek_calc_method: GreekCalculationMethod | None = None,
        time_bump_days: float = 1.0,
    ) -> float:
        """Calculate theta (time decay) of the option.

        Theta measures the rate of change of option value with respect to time.
        For the BSM method, closed-form analytical formulas are available (default).
        For other methods, a numerical finite-difference approach is used.

        Parameters
        ==========
        greek_calc_method : GreekCalculationMethod | None, optional
            Method to use for greek calculation (analytical or numerical)
            - None (default): Uses analytical for BSM, numerical for others
            - GreekCalculationMethod.ANALYTICAL: Uses closed-form formulas (BSM only)
            - GreekCalculationMethod.NUMERICAL: Uses bump-and-revalue finite-difference
        time_bump_days : float, optional
            Time bump size in days for numerical calculation (default: 1.0)
            This is the number of days to advance the pricing date

        Returns
        =======
        float
            option theta (change in option value per day)

        Notes
        =====
        Numerical theta bumps the pricing date forward by calendar days. This is
        consistent with the BSM analytical theta using a 365-day scaling, but it
        can differ from trading-day conventions.

        Raises
        ======
        ValueError
            if analytical method requested for non-BSM pricing method
        """
        use_analytical = self._resolve_greek_method(
            greek_calc_method,
            error_message="Analytical theta calculation only available for BSM method",
        )

        if use_analytical:
            return self._impl.theta()

        # Numerical theta: bump pricing date forward by time_bump_days and reprice
        bumped_date = self.pricing_date + dt.timedelta(days=time_bump_days)
        if bumped_date >= self.maturity:
            return 0.0

        value_now = self.present_value()

        # Build bumped underlying
        bumped_market = MarketData(
            pricing_date=bumped_date,
            discount_curve=self.discount_curve,
            currency=self.underlying.currency,
        )
        underlying_bumped = self.underlying.replace(market_data=bumped_market)

        valuation_bumped = OptionValuation(
            name=f"{self.name}_theta_bumped",
            underlying=underlying_bumped,
            spec=self.spec,
            pricing_method=self.pricing_method,
            params=self.params,
        )

        value_bumped = valuation_bumped.present_value()
        return (value_bumped - value_now) / time_bump_days

    def rho(
        self,
        greek_calc_method: GreekCalculationMethod | None = None,
        rate_bump: float = 0.01,
    ) -> float:
        """Calculate rho (interest rate sensitivity) of the option.

        Rho measures the rate of change of option value with respect to the risk-free rate.
        For the BSM method, closed-form analytical formulas are available (default).
        For other methods, a numerical finite-difference approach is used.

        Parameters
        ==========
        greek_calc_method : GreekCalculationMethod | None, optional
            Method to use for greek calculation (analytical or numerical)
            - None (default): Uses analytical for BSM, numerical for others
            - GreekCalculationMethod.ANALYTICAL: Uses closed-form formulas (BSM only)
            - GreekCalculationMethod.NUMERICAL: Uses bump-and-revalue finite-difference
        rate_bump : float, optional
            Rate bump size for numerical calculation (default: 0.01 = 1%)
            The result is normalized to represent change per 1% rate move

        Returns
        =======
        float
            option rho (change in option value per 1% change in risk-free rate)

        Raises
        ======
        ValueError
            if analytical method requested for non-BSM pricing method
        """
        use_analytical = self._resolve_greek_method(
            greek_calc_method,
            error_message="Analytical rho calculation only available for BSM method",
        )

        if use_analytical:
            return self._impl.rho()

        # Numerical rho: bump risk-free rate up and down and reprice
        if self.discount_curve.flat_rate is None:
            raise NotImplementedError("Numerical rho requires a flat discount curve.")

        rate_up = self.discount_curve.flat_rate + rate_bump / 2
        rate_down = self.discount_curve.flat_rate - rate_bump / 2

        ttm = calculate_year_fraction(self.pricing_date, self.maturity)
        curve_up = DiscountCurve.flat(
            f"{self.discount_curve.name}_up",
            rate_up,
            end_time=ttm,
        )
        curve_down = DiscountCurve.flat(
            f"{self.discount_curve.name}_down",
            rate_down,
            end_time=ttm,
        )

        md_up = MarketData(self.pricing_date, curve_up, currency=self.currency)
        md_down = MarketData(self.pricing_date, curve_down, currency=self.currency)

        underlying_up = self.underlying.replace(market_data=md_up)
        underlying_down = self.underlying.replace(market_data=md_down)

        val_up = self._build_valuation(name_suffix="rho_up", underlying=underlying_up)
        val_down = self._build_valuation(name_suffix="rho_down", underlying=underlying_down)

        value_up = val_up.present_value()
        value_down = val_down.present_value()

        # Central difference, normalized to per 1% rate change
        # rate_up/down are bumped by ± rate_bump/2, so denominator is rate_bump.
        return (value_up - value_down) / rate_bump * 0.01
