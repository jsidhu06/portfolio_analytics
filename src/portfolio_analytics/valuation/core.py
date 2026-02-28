from __future__ import annotations
from dataclasses import dataclass, replace as dc_replace
from collections.abc import Callable, Sequence
import datetime as dt
import logging
import numpy as np
from ..stochastic_processes import PathSimulation
from ..exceptions import ConfigurationError, UnsupportedFeatureError, ValidationError
from ..enums import (
    OptionType,
    AsianAveraging,
    ExerciseType,
    PricingMethod,
    GreekCalculationMethod,
)
from .monte_carlo import (
    _MCEuropeanValuation,
    _MCAmericanValuation,
    _MCAsianValuation,
    _MCAsianAmericanValuation,
)
from .binomial import (
    _BinomialEuropeanValuation,
    _BinomialAmericanValuation,
    _BinomialAsianValuation,
)
from .bsm import _BSMEuropeanValuation
from .asian_analytical import _AnalyticalAsianValuation
from .pde import _FDEuropeanValuation, _FDAmericanValuation
from ..rates import DiscountCurve
from ..utils import calculate_year_fraction
from ..market_environment import MarketData
from .params import BinomialParams, MonteCarloParams, PDEParams, ValuationParams

logger = logging.getLogger(__name__)

# ── PV interceptors ─────────────────────────────────────────────────
# Maps a tag → OptionValuation method name that completely replaces the
# normal present_value() flow.  Resolved once during __init__ via
# _resolve_interceptor(); at most one interceptor per instance.
_PV_INTERCEPTORS: dict[str, str] = {
    "SEASONED_ASIAN": "_seasoned_asian_pv",
}


def _resolve_interceptor(
    spec: OptionSpec | PayoffSpec | AsianOptionSpec,
) -> str | None:
    """Return a _PV_INTERCEPTORS key if *spec* requires pre-PV transformation."""
    if isinstance(spec, AsianOptionSpec) and spec.observed_average is not None:
        return "SEASONED_ASIAN"
    return None


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

# Maps (PricingMethod, ExerciseType) → implementation class for Asian option specs.
_ASIAN_REGISTRY: dict[tuple[PricingMethod, ExerciseType], type] = {
    (PricingMethod.MONTE_CARLO, ExerciseType.EUROPEAN): _MCAsianValuation,
    (PricingMethod.MONTE_CARLO, ExerciseType.AMERICAN): _MCAsianAmericanValuation,
    (PricingMethod.BINOMIAL, ExerciseType.EUROPEAN): _BinomialAsianValuation,
    (PricingMethod.BINOMIAL, ExerciseType.AMERICAN): _BinomialAsianValuation,
    (PricingMethod.BSM, ExerciseType.EUROPEAN): _AnalyticalAsianValuation,
}


@dataclass(frozen=True, slots=True)
class OptionSpec:
    """Contract specification for a vanilla option."""

    option_type: OptionType  # CALL / PUT
    exercise_type: ExerciseType  # EUROPEAN / AMERICAN
    strike: float
    maturity: dt.datetime
    currency: str | None = None
    contract_size: int | float = 100

    def __post_init__(self) -> None:
        """Validate option_type/exercise_type and coerce strike."""
        if not isinstance(self.option_type, OptionType):
            raise ConfigurationError(
                f"option_type must be OptionType enum, got {type(self.option_type).__name__}"
            )
        if self.option_type not in (OptionType.CALL, OptionType.PUT):
            raise ValidationError(
                "OptionSpec.option_type must be OptionType.CALL or OptionType.PUT"
            )
        if not isinstance(self.exercise_type, ExerciseType):
            raise ConfigurationError(
                f"exercise_type must be ExerciseType enum, got {type(self.exercise_type).__name__}"
            )

        if self.strike is None:
            raise ValidationError("OptionSpec.strike must be provided")
        try:
            strike = float(self.strike)
        except (TypeError, ValueError) as exc:
            raise ConfigurationError("OptionSpec.strike must be numeric") from exc
        if not np.isfinite(strike):
            raise ValidationError("OptionSpec.strike must be finite")
        if strike < 0.0:
            raise ValidationError("OptionSpec.strike must be >= 0")
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
    payoff_fn: Callable[[np.ndarray | float], np.ndarray]
    currency: str | None = None
    contract_size: int | float = 100

    # Kept for compatibility with vanilla valuation interfaces
    strike: None = None

    def __post_init__(self) -> None:
        if not isinstance(self.exercise_type, ExerciseType):
            raise ConfigurationError(
                f"exercise_type must be ExerciseType enum, got {type(self.exercise_type).__name__}"
            )
        if not callable(self.payoff_fn):
            raise ConfigurationError("payoff_fn must be callable")

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
    currency : str, optional
        Currency denomination
    averaging_start : dt.datetime, optional
        Start of averaging period. If None, uses pricing date.
    num_steps : int, optional
        Number of equally spaced time steps within the averaging window.
        Required for analytical (BSM) pricing.  For Monte Carlo and Binomial the
        step count is determined by the simulation/tree time grid.
    exercise_type : ExerciseType
        Exercise style (EUROPEAN or AMERICAN). Default: EUROPEAN.
    contract_size : int | float
        Contract multiplier (default 100)
    observed_average : float, optional
        For seasoned Asians: the realised average price over the already-observed
        period.  Must be provided together with ``observed_count``.
    observed_count : int, optional
        For seasoned Asians: the number of already-observed fixings (n₁).
        Must be provided together with ``observed_average``.

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
    currency: str | None = None
    averaging_start: dt.datetime | None = None
    num_steps: int | None = None
    contract_size: int | float = 100
    exercise_type: ExerciseType = ExerciseType.EUROPEAN
    observed_average: float | None = None
    observed_count: int | None = None

    def __post_init__(self) -> None:
        """Validate Asian option specification."""
        if not isinstance(self.averaging, AsianAveraging):
            raise ConfigurationError(
                f"averaging must be AsianAveraging enum, got {type(self.averaging).__name__}"
            )
        if not isinstance(self.exercise_type, ExerciseType):
            raise ConfigurationError(
                f"exercise_type must be ExerciseType enum, got {type(self.exercise_type).__name__}"
            )

        if not isinstance(self.call_put, OptionType):
            raise ConfigurationError(
                f"call_put must be OptionType enum, got {type(self.call_put).__name__}"
            )
        if self.call_put not in (OptionType.CALL, OptionType.PUT):
            raise ValidationError(
                "AsianOptionSpec.call_put must be OptionType.CALL or OptionType.PUT"
            )

        if self.strike is None:
            raise ValidationError("AsianOptionSpec.strike must be provided")
        try:
            strike = float(self.strike)
        except (TypeError, ValueError) as exc:
            raise ConfigurationError("AsianOptionSpec.strike must be numeric") from exc
        if not np.isfinite(strike):
            raise ValidationError("AsianOptionSpec.strike must be finite")
        if strike < 0.0:
            raise ValidationError("AsianOptionSpec.strike must be >= 0")
        object.__setattr__(self, "strike", strike)

        if self.num_steps is not None:
            if not isinstance(self.num_steps, int) or self.num_steps < 1:
                raise ValidationError("num_steps must be a positive integer")

        # Seasoned Asian: observed_average and observed_count must be both set or both None
        if (self.observed_average is None) != (self.observed_count is None):
            raise ValidationError(
                "observed_average and observed_count must both be provided or both omitted."
            )
        if self.observed_average is not None:
            try:
                obs_avg = float(self.observed_average)
            except (TypeError, ValueError) as exc:
                raise ConfigurationError("observed_average must be numeric") from exc
            if not np.isfinite(obs_avg):
                raise ValidationError("observed_average must be finite")
            if obs_avg <= 0.0:
                raise ValidationError("observed_average must be > 0")
            object.__setattr__(self, "observed_average", obs_avg)

            if not isinstance(self.observed_count, int) or self.observed_count < 1:
                raise ValidationError("observed_count must be a positive integer")


@dataclass(frozen=True, slots=True)
class UnderlyingPricingData:
    """Minimal data container for option valuation underlying asset.

    Used when pricing with methods that don't require full stochastic process simulation
    (e.g., BSM, binomial trees, FD approximation to PDE).
    Contains only essential parameters: spot price, volatility, pricing date, discount curve,
    continuous dividend yield via dividend_curve,
    and optional discrete dividends as (ex_date, amount) pairs.
    """

    initial_value: float
    volatility: float
    market_data: MarketData
    discrete_dividends: Sequence[tuple[dt.datetime, float]] | None = None
    dividend_curve: DiscountCurve | None = None

    def __post_init__(self) -> None:
        if self.discrete_dividends is not None:
            cleaned: list[tuple[dt.datetime, float]] = []
            for ex_date, amount in self.discrete_dividends:
                if not isinstance(ex_date, dt.datetime):
                    raise ConfigurationError(
                        "discrete_dividends entries must be (datetime, amount) tuples"
                    )
                try:
                    amt = float(amount)
                except (TypeError, ValueError) as exc:
                    raise ConfigurationError("dividend amount must be numeric") from exc
                cleaned.append((ex_date, amt))
            object.__setattr__(
                self,
                "discrete_dividends",
                tuple(sorted(cleaned, key=lambda x: x[0])),
            )
        else:
            object.__setattr__(self, "discrete_dividends", ())

        if self.dividend_curve is not None and self.discrete_dividends:
            import warnings

            warnings.warn(
                "UnderlyingPricingData: both dividend_curve and discrete_dividends "
                "provided. The continuous yield will enter the drift and discrete "
                "dividends will be subtracted at each ex-date.",
                stacklevel=2,
            )

    @property
    def pricing_date(self) -> dt.datetime:
        return self.market_data.pricing_date

    @property
    def discount_curve(self) -> DiscountCurve:
        return self.market_data.discount_curve

    @property
    def currency(self) -> str:
        return self.market_data.currency

    def replace(self, **kwargs: object) -> "UnderlyingPricingData":
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
        return dc_replace(self, **kwargs)  # type: ignore[arg-type]


class OptionValuation:
    """Single-factor option valuation dispatcher.

    Routes to the appropriate pricing implementation based on pricing_method + exercise_type.
    Instances are effectively immutable once created — constructor arguments are exposed as
    read-only properties.
    """

    def __init__(
        self,
        underlying: PathSimulation | UnderlyingPricingData,
        spec: OptionSpec | PayoffSpec | AsianOptionSpec,
        pricing_method: PricingMethod,
        params: ValuationParams | None = None,
    ) -> None:
        # --- store private state ---
        self._underlying = underlying
        self._spec = spec

        # Validate pricing_method early — comparisons rely on enum identity
        if not isinstance(pricing_method, PricingMethod):
            raise ConfigurationError(
                f"pricing_method must be PricingMethod enum, got {type(pricing_method).__name__}"
            )
        self._pricing_method = pricing_method

        # Resolve option_type (best-effort across spec variants)
        if hasattr(spec, "option_type") and isinstance(spec.option_type, OptionType):
            self._option_type: OptionType | None = spec.option_type
        elif hasattr(spec, "call_put") and isinstance(spec.call_put, OptionType):
            self._option_type = spec.call_put
        else:
            self._option_type = None

        # Resolve params
        self._params: ValuationParams | None = self._resolve_params(
            pricing_method=pricing_method, params=params
        )

        # --- currency resolution & check (default match) ---

        self._currency = spec.currency or underlying.currency

        if spec.currency is not None and spec.currency != underlying.currency:
            raise UnsupportedFeatureError(
                "Cross-currency valuation is not supported. "
                "Option currency must match the underlying market currency."
            )

        # Strategy guardrails
        if pricing_method == PricingMethod.BSM and self._option_type not in (
            OptionType.CALL,
            OptionType.PUT,
        ):
            raise UnsupportedFeatureError(
                "BSM pricing is only available for vanilla CALL/PUT option types."
            )

        # Optional sanity check: maturity must be after pricing date
        if self.maturity <= self.pricing_date:
            raise ValidationError("Option maturity must be after pricing_date.")

        # Merge pricing_date + maturity into the simulation's observation_dates
        # via replace() so the caller's PathSimulation is never mutated.
        if isinstance(underlying, PathSimulation):
            merged = underlying.observation_dates | {self.pricing_date, self.maturity}
            if merged != underlying.observation_dates:
                underlying = underlying.replace(observation_dates=merged)
            self._underlying = underlying

        # Validate that MC requires PathSimulation
        if pricing_method == PricingMethod.MONTE_CARLO and not isinstance(
            self._underlying, PathSimulation
        ):
            raise ConfigurationError(
                "Monte Carlo pricing requires underlying to be a PathSimulation instance"
            )

        # Validate that deterministic methods don't receive PathSimulation
        if pricing_method in (
            PricingMethod.BINOMIAL,
            PricingMethod.BSM,
            PricingMethod.PDE_FD,
        ) and isinstance(self._underlying, PathSimulation):
            raise ConfigurationError(
                f"{pricing_method.name} pricing does not use stochastic path simulation. "
                "Pass an UnderlyingPricingData instance instead of PathSimulation."
            )

        # Dispatch to appropriate pricing method implementation
        self._impl = self._build_impl()

        # Resolve optional PV interceptor (e.g. seasoned Asian K* adjustment)
        tag = _resolve_interceptor(self._spec)
        method_name = _PV_INTERCEPTORS.get(tag) if tag else None  # type: ignore[arg-type]
        self._pv_interceptor: Callable[[], float] | None = (
            getattr(self, method_name) if method_name else None
        )

    # ──────────────────────────────
    # Public API (methods)
    # ──────────────────────────────

    def solve(
        self,
    ) -> float | np.ndarray | tuple[np.ndarray, np.ndarray] | tuple[float, np.ndarray, np.ndarray]:
        """Run the pricing method's core solver and return its raw output."""
        return self._impl.solve()

    def present_value(self) -> float:
        """Calculate present value of the derivative."""
        if self._pv_interceptor is not None:
            return float(self._pv_interceptor())

        base_pv = float(self._impl.present_value())
        if self._params is None or not getattr(self._params, "control_variate_european", False):
            return base_pv

        return float(self._apply_control_variate(base_pv))

    def present_value_pathwise(self) -> np.ndarray:
        """Return discounted pathwise present values (Monte Carlo only)."""
        pv_pathwise = getattr(self._impl, "present_value_pathwise", None)
        if pv_pathwise is None:
            raise UnsupportedFeatureError(
                "present_value_pathwise is only implemented for Monte Carlo valuation."
            )
        return pv_pathwise()

    def delta(
        self,
        epsilon: float | None = None,
        greek_calc_method: GreekCalculationMethod | None = None,
    ) -> float:
        method = self._resolve_greek_method(
            greek_calc_method,
            tree_capable=True,
            mc_analytic_capable=True,
        )
        if method == GreekCalculationMethod.PATHWISE:
            return float(self._impl.delta_pathwise())
        if method == GreekCalculationMethod.LIKELIHOOD_RATIO:
            return float(self._impl.delta_lr())
        if method != GreekCalculationMethod.NUMERICAL:
            return float(self._impl.delta())

        if epsilon is None:
            epsilon = self._underlying.initial_value / 100

        underlying_down = self._underlying.replace(
            initial_value=self._underlying.initial_value - epsilon
        )
        underlying_up = self._underlying.replace(
            initial_value=self._underlying.initial_value + epsilon
        )

        val_down = self._build_valuation(underlying=underlying_down)
        val_up = self._build_valuation(underlying=underlying_up)

        return (val_up.present_value() - val_down.present_value()) / (2 * epsilon)

    def gamma(
        self,
        epsilon: float | None = None,
        greek_calc_method: GreekCalculationMethod | None = None,
    ) -> float:
        method = self._resolve_greek_method(
            greek_calc_method,
            tree_capable=True,
            mc_analytic_capable=True,
        )
        if method == GreekCalculationMethod.PATHWISE:
            return float(self._impl.gamma_pathwise_fd(epsilon))
        if method == GreekCalculationMethod.LIKELIHOOD_RATIO:
            raise ValidationError(
                "likelihood_ratio is not available for gamma. "
                "Use PATHWISE (central-difference of pathwise delta) or NUMERICAL."
            )
        if method != GreekCalculationMethod.NUMERICAL:
            return float(self._impl.gamma())

        if epsilon is None:
            epsilon = self._underlying.initial_value / 100

        underlying_down = self._underlying.replace(
            initial_value=self._underlying.initial_value - epsilon
        )
        underlying_up = self._underlying.replace(
            initial_value=self._underlying.initial_value + epsilon
        )

        val_down = self._build_valuation(underlying=underlying_down)
        val_up = self._build_valuation(underlying=underlying_up)

        value_left = val_down.present_value()
        value_right = val_up.present_value()
        value_center = self.present_value()

        return (value_right - 2 * value_center + value_left) / (epsilon**2)

    def vega(
        self,
        epsilon: float = 0.01,
        greek_calc_method: GreekCalculationMethod | None = None,
    ) -> float:
        method = self._resolve_greek_method(greek_calc_method, mc_analytic_capable=True)
        if method == GreekCalculationMethod.PATHWISE:
            return float(self._impl.vega_pathwise())
        if method == GreekCalculationMethod.LIKELIHOOD_RATIO:
            return float(self._impl.vega_lr())
        if method == GreekCalculationMethod.ANALYTICAL:
            return float(self._impl.vega())

        underlying_down = self._underlying.replace(volatility=self._underlying.volatility - epsilon)
        underlying_up = self._underlying.replace(volatility=self._underlying.volatility + epsilon)

        val_down = self._build_valuation(underlying=underlying_down)
        val_up = self._build_valuation(underlying=underlying_up)

        vega = (val_up.present_value() - val_down.present_value()) / (2 * epsilon) / 100
        return vega

    def theta(
        self,
        greek_calc_method: GreekCalculationMethod | None = None,
        time_bump_days: float = 1.0,
    ) -> float:
        method = self._resolve_greek_method(greek_calc_method, tree_capable=True)
        if method != GreekCalculationMethod.NUMERICAL:
            return float(self._impl.theta())

        bumped_date = self.pricing_date + dt.timedelta(days=time_bump_days)
        if bumped_date >= self.maturity:
            return 0.0

        value_now = self.present_value()

        bumped_market = MarketData(
            pricing_date=bumped_date,
            discount_curve=self.discount_curve,
            currency=self._underlying.currency,
        )
        underlying_bumped = self._underlying.replace(market_data=bumped_market)

        value_bumped = OptionValuation(
            underlying=underlying_bumped,
            spec=self._spec,
            pricing_method=self._pricing_method,
            params=self._params,
        ).present_value()

        return (value_bumped - value_now) / time_bump_days

    def rho(
        self,
        greek_calc_method: GreekCalculationMethod | None = None,
        rate_bump: float = 0.01,
    ) -> float:
        method = self._resolve_greek_method(greek_calc_method)
        if method == GreekCalculationMethod.ANALYTICAL:
            return float(self._impl.rho())

        if self.discount_curve.flat_rate is None:
            raise UnsupportedFeatureError("Numerical rho requires a flat discount curve.")

        rate_up = self.discount_curve.flat_rate + rate_bump / 2
        rate_down = self.discount_curve.flat_rate - rate_bump / 2

        ttm = calculate_year_fraction(self.pricing_date, self.maturity)
        curve_up = DiscountCurve.flat(rate_up, end_time=ttm)
        curve_down = DiscountCurve.flat(rate_down, end_time=ttm)

        md_up = MarketData(self.pricing_date, curve_up, currency=self.currency)
        md_down = MarketData(self.pricing_date, curve_down, currency=self.currency)

        underlying_up = self._underlying.replace(market_data=md_up)
        underlying_down = self._underlying.replace(market_data=md_down)

        val_up = self._build_valuation(underlying=underlying_up)
        val_down = self._build_valuation(underlying=underlying_down)

        return (val_up.present_value() - val_down.present_value()) / rate_bump * 0.01

    # ──────────────────────────────
    # Read-only properties (public)
    # ──────────────────────────────

    @property
    def underlying(self) -> PathSimulation | UnderlyingPricingData:
        return self._underlying

    @property
    def spec(self) -> OptionSpec | PayoffSpec | AsianOptionSpec:
        return self._spec

    @property
    def pricing_method(self) -> PricingMethod:
        return self._pricing_method

    @property
    def params(self) -> ValuationParams | None:
        return self._params

    @property
    def option_type(self) -> OptionType | None:
        return self._option_type

    @property
    def maturity(self) -> dt.datetime:
        return self._spec.maturity

    @property
    def strike(self) -> float | None:
        return self._spec.strike

    @property
    def currency(self) -> str:
        # effective currency resolved in __init__
        return self._currency

    @property
    def exercise_type(self) -> ExerciseType:
        return self._spec.exercise_type

    @property
    def contract_size(self) -> int | float:
        return self._spec.contract_size

    @property
    def pricing_date(self) -> dt.datetime:
        return self._underlying.pricing_date

    @property
    def discount_curve(self) -> DiscountCurve:
        return self._underlying.discount_curve

    # ──────────────────────────────
    # Private API (helpers)
    # ──────────────────────────────

    def _build_impl(self):
        spec = self._spec

        if isinstance(spec, AsianOptionSpec):
            impl_cls = _ASIAN_REGISTRY.get((self._pricing_method, spec.exercise_type))
            if impl_cls is None:
                raise ValidationError(
                    f"Asian options with {spec.exercise_type.name} exercise "
                    f"do not support {self._pricing_method.name} pricing."
                )
            return impl_cls(self)

        impl_cls = _VANILLA_REGISTRY.get((self._pricing_method, spec.exercise_type))
        if impl_cls is None:
            if self._pricing_method is PricingMethod.BSM:
                raise UnsupportedFeatureError(
                    "BSM is only applicable to European option valuation. "
                    "Select a different pricing method for American options such as Binomial, "
                    "PDE_FD or MONTE_CARLO."
                )
            raise ValidationError(
                f"{self._pricing_method.name} does not support {spec.exercise_type.name} exercise."
            )
        return impl_cls(self)

    @staticmethod
    def _resolve_params(
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
                raise ConfigurationError(
                    "pricing_method=MONTE_CARLO requires params=MonteCarloParams"
                )
            return params

        if pricing_method == PricingMethod.BINOMIAL:
            if not isinstance(params, BinomialParams):
                raise ConfigurationError("pricing_method=BINOMIAL requires params=BinomialParams")
            return params

        if pricing_method == PricingMethod.PDE_FD:
            if not isinstance(params, PDEParams):
                raise ConfigurationError("pricing_method=PDE_FD requires params=PDEParams")
            return params

        raise ConfigurationError(
            f"pricing_method={pricing_method.name} does not accept valuation params"
        )

    def _apply_control_variate(self, base_pv: float) -> float:
        if self._spec.exercise_type is not ExerciseType.AMERICAN:
            raise ValidationError(
                "control_variate_european is only valid for options with American exercise."
            )

        if isinstance(self._spec, AsianOptionSpec):
            return self._apply_asian_control_variate(base_pv)

        if self._pricing_method not in (
            PricingMethod.BINOMIAL,
            PricingMethod.PDE_FD,
            PricingMethod.MONTE_CARLO,
        ):
            raise UnsupportedFeatureError(
                "control_variate_european is only supported for BINOMIAL, PDE_FD, "
                "and MONTE_CARLO pricing."
            )
        if not isinstance(self._spec, OptionSpec):
            raise UnsupportedFeatureError(
                "Vanilla control_variate_european requires spec to be of type OptionSpec. "
                "PayoffSpec is not supported."
            )
        if self._option_type not in (OptionType.CALL, OptionType.PUT):
            raise UnsupportedFeatureError(
                "Vanilla control_variate_european requires a CALL or PUT option type."
            )

        euro_spec = dc_replace(self._spec, exercise_type=ExerciseType.EUROPEAN)

        cv_params = dc_replace(self._params, control_variate_european=False)
        euro_num = OptionValuation(
            underlying=self._underlying,
            spec=euro_spec,
            pricing_method=self._pricing_method,
            params=cv_params,
        ).present_value()

        bsm_underlying = self._as_underlying_data()

        euro_bsm = OptionValuation(
            underlying=bsm_underlying,
            spec=euro_spec,
            pricing_method=PricingMethod.BSM,
        ).present_value()

        return base_pv + (euro_bsm - euro_num)

    def _as_underlying_data(self) -> UnderlyingPricingData:
        """Return an UnderlyingPricingData instance, extracting from PathSimulation if needed."""
        if isinstance(self._underlying, PathSimulation):
            return UnderlyingPricingData(
                initial_value=self._underlying.initial_value,
                volatility=self._underlying.volatility,
                market_data=self._underlying.market_data,
                dividend_curve=self._underlying.dividend_curve,
                discrete_dividends=self._underlying.discrete_dividends or None,
            )
        return self._underlying

    def _apply_asian_control_variate(self, base_pv: float) -> float:
        if self._pricing_method not in (PricingMethod.BINOMIAL, PricingMethod.MONTE_CARLO):
            raise UnsupportedFeatureError(
                "Asian control_variate_european is only supported for "
                "BINOMIAL and MONTE_CARLO pricing."
            )
        spec = self._spec
        assert isinstance(spec, AsianOptionSpec)
        if spec.averaging not in (AsianAveraging.GEOMETRIC, AsianAveraging.ARITHMETIC):
            raise UnsupportedFeatureError(
                "Asian control_variate_european requires GEOMETRIC or ARITHMETIC averaging "
            )

        params = self._params

        if self._pricing_method is PricingMethod.BINOMIAL:
            if not isinstance(params, BinomialParams):
                raise ConfigurationError("Expected BinomialParams for binomial pricing.")
            if params.asian_tree_averages is None:
                raise UnsupportedFeatureError(
                    "Asian control_variate_european requires Hull tree averages "
                    "(set asian_tree_averages on BinomialParams)."
                )

        cv_params = dc_replace(params, control_variate_european=False)

        euro_spec = dc_replace(spec, exercise_type=ExerciseType.EUROPEAN)
        euro_num = OptionValuation(
            underlying=self._underlying,
            spec=euro_spec,
            pricing_method=self._pricing_method,
            params=cv_params,
        ).present_value()

        bsm_underlying = self._as_underlying_data()
        if isinstance(self._underlying, PathSimulation):
            n_steps = len(self._underlying.time_grid) - 1
            bsm_spec = dc_replace(euro_spec, num_steps=n_steps)  # type: ignore[arg-type]
        else:
            bsm_spec = dc_replace(euro_spec, num_steps=params.num_steps)  # type: ignore[union-attr, arg-type]

        euro_analytical = OptionValuation(
            underlying=bsm_underlying,
            spec=bsm_spec,
            pricing_method=PricingMethod.BSM,
        ).present_value()

        logger.debug(
            "Asian CV: american=%.6f euro_num=%.6f euro_analytical=%.6f adj=%.6f",
            base_pv,
            euro_num,
            euro_analytical,
            euro_analytical - euro_num,
        )

        return base_pv + (euro_analytical - euro_num)

    # ── Seasoned Asian ───────────────────────────────────────────────────

    def _seasoned_asian_future_obs(self) -> int:
        """Return the number of *future* averaging observations (n₂)."""
        spec = self._spec
        assert isinstance(spec, AsianOptionSpec)

        if self._pricing_method is PricingMethod.BSM:
            if spec.num_steps is None:
                raise ValidationError(
                    "num_steps is required on AsianOptionSpec for analytical (BSM) pricing."
                )
            return spec.num_steps + 1

        if self._pricing_method is PricingMethod.BINOMIAL:
            assert isinstance(self._params, BinomialParams)
            return self._params.num_steps + 1

        if self._pricing_method is PricingMethod.MONTE_CARLO:
            assert isinstance(self._underlying, PathSimulation)
            self._underlying._ensure_time_grid()
            return len(self._underlying.time_grid)

        raise UnsupportedFeatureError(
            f"Seasoned Asian pricing is not supported for {self._pricing_method.name}."
        )

    def _seasoned_asian_pv(self) -> float:
        """Price a seasoned Asian using Hull's adjusted-strike reduction.
        When part of the averaging window has elapsed, the payoff of an
        average-price call is::

            max((n₁·S̄ + n₂·S_avg_future) / (n₁+n₂) − K, 0)

        which equals ``(n₂/(n₁+n₂)) · max(S_avg_future − K*, 0)`` where::

            K* = ((n₁+n₂)/n₂) · K  −  (n₁/n₂) · S̄

        When K* > 0 this is a newly-issued Asian with strike K* scaled by
        n₂/(n₁+n₂).  When K* ≤ 0 the option is certain to be exercised and
        its value is that of a forward contract on the remaining average.

        See Hull, *Options, Futures, and Other Derivatives*, Section 26.13."""
        spec = self._spec
        assert isinstance(spec, AsianOptionSpec)
        assert spec.observed_average is not None and spec.observed_count is not None

        n1 = spec.observed_count
        n2 = self._seasoned_asian_future_obs()
        n_total = n1 + n2
        S_bar = spec.observed_average
        K = spec.strike

        K_star = (n_total / n2) * K - (n1 / n2) * S_bar
        scale = n2 / n_total

        logger.debug(
            "Seasoned Asian: n1=%d n2=%d S_bar=%.4f K=%.4f K*=%.4f scale=%.4f",
            n1,
            n2,
            S_bar,
            K,
            K_star,
            scale,
        )

        if K_star > 0.0:
            fresh_spec = dc_replace(spec, strike=K_star, observed_average=None, observed_count=None)
            fresh_pv = OptionValuation(
                underlying=self._underlying,
                spec=fresh_spec,
                pricing_method=self._pricing_method,
                params=self._params,
            ).present_value()
            return scale * fresh_pv

        # K* <= 0: option is certain to be exercised → value as forward contract.
        # For a call:  scale · [M₁·e^{-rT} − K*·e^{-rT}]
        # For a put:   scale · [K*·e^{-rT} − M₁·e^{-rT}]  (always 0 when K*<=0)
        # M₁ is the forward of the average over the remaining period.  Rather than
        # recompute the exact first moment, we price a fresh Asian with strike=0
        # (deep ITM) which equals the discounted expected average, then apply the
        # K* offset.

        ttm = calculate_year_fraction(self.pricing_date, self.maturity)
        df = float(self.discount_curve.df(ttm))

        fresh_spec_zero = dc_replace(
            spec,
            strike=0.0,
            observed_average=None,
            observed_count=None,
        )
        # A zero-strike Asian call equals e^{-rT} · E[S_avg] = discounted M₁
        disc_M1 = OptionValuation(
            underlying=self._underlying,
            spec=dc_replace(fresh_spec_zero, call_put=OptionType.CALL),
            pricing_method=self._pricing_method,
            params=self._params,
        ).present_value()

        if spec.call_put is OptionType.CALL:
            return scale * (disc_M1 - K_star * df)
        # Put with K*<=0: max(K* - S_avg, 0) is 0 when K*<=0 and S_avg>0
        return 0.0

    def _resolve_greek_method(
        self,
        greek_calc_method: GreekCalculationMethod | None,
        *,
        tree_capable: bool = False,
        mc_analytic_capable: bool = False,
    ) -> GreekCalculationMethod:
        if greek_calc_method is not None and not isinstance(
            greek_calc_method, GreekCalculationMethod
        ):
            raise ConfigurationError(
                f"greek_calc_method must be GreekCalculationMethod enum, "
                f"got {type(greek_calc_method).__name__}"
            )

        if greek_calc_method is None:
            if self._pricing_method == PricingMethod.BSM:
                return GreekCalculationMethod.ANALYTICAL
            if tree_capable and self._pricing_method == PricingMethod.BINOMIAL:
                return GreekCalculationMethod.TREE
            return GreekCalculationMethod.NUMERICAL

        if (
            greek_calc_method == GreekCalculationMethod.ANALYTICAL
            and self._pricing_method != PricingMethod.BSM
        ):
            raise ValidationError(
                "Analytical greeks are only available for BSM pricing method. "
                "Use GreekCalculationMethod.NUMERICAL (or .TREE for binomial delta/gamma/theta)."
            )

        if greek_calc_method == GreekCalculationMethod.TREE:
            if self._pricing_method != PricingMethod.BINOMIAL:
                raise ValidationError("Tree greeks are only available for BINOMIAL pricing method.")
            if not tree_capable:
                raise ValidationError(
                    "Tree extraction is not available for this greek. "
                    "Only delta, gamma, and theta support GreekCalculationMethod.TREE."
                )

        if greek_calc_method in (
            GreekCalculationMethod.PATHWISE,
            GreekCalculationMethod.LIKELIHOOD_RATIO,
        ):
            if self._pricing_method != PricingMethod.MONTE_CARLO:
                raise ValidationError(
                    f"{greek_calc_method.value} greeks are only available for "
                    "MONTE_CARLO pricing method."
                )
            if not mc_analytic_capable:
                raise ValidationError(
                    f"{greek_calc_method.value} is not available for this greek. "
                    "Only delta, gamma, and vega support PATHWISE; "
                    "only delta and vega support LIKELIHOOD_RATIO."
                )
            if not isinstance(self._spec, OptionSpec):
                raise ValidationError(
                    f"{greek_calc_method.value} greeks are only implemented for "
                    "vanilla European options (OptionSpec)."
                )
            if self._underlying.discrete_dividends:
                raise UnsupportedFeatureError(
                    "Pathwise and likelihood-ratio MC Greeks are not supported "
                    "with discrete dividends."
                )

        return greek_calc_method

    def _build_valuation(self, *, underlying) -> "OptionValuation":
        return OptionValuation(
            underlying=underlying,
            spec=self._spec,
            pricing_method=self._pricing_method,
            params=self._params,
        )
