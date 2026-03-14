"""Top-level exports for common portfolio_analytics types."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("portfolio_analytics")
except PackageNotFoundError:
    __version__ = "0+unknown"

from .exceptions import (
    DerivativesAnalyticsError,
    ValidationError,
    ConfigurationError,
    UnsupportedFeatureError,
    NumericalError,
    ArbitrageViolationError,
    ConvergenceError,
    StabilityError,
)
from .market_environment import MarketData, CorrelationContext
from .rates import DiscountCurve
from .stochastic_processes import (
    SimulationConfig,
    GBMParams,
    JDParams,
    SRDParams,
    GBMProcess,
    JDProcess,
    SRDProcess,
    PathSimulation,
)
from .enums import (
    OptionType,
    ExerciseType,
    PricingMethod,
    PDEMethod,
    PDESpaceGrid,
    PDEEarlyExercise,
    DayCountConvention,
    AsianAveraging,
    GreekCalculationMethod,
    ImpliedVolMethod,
)
from .valuation import (
    VanillaSpec,
    OptionValuation,
    UnderlyingData,
    PayoffSpec,
    AsianSpec,
    MonteCarloParams,
    BinomialParams,
    PDEParams,
)
from .valuation.implied_volatility import implied_volatility

__all__ = [
    # Market data
    "MarketData",
    "CorrelationContext",
    "DiscountCurve",
    # Stochastic processes
    "SimulationConfig",
    "GBMParams",
    "JDParams",
    "SRDParams",
    "GBMProcess",
    "JDProcess",
    "SRDProcess",
    "PathSimulation",
    # Enums
    "OptionType",
    "ExerciseType",
    "PricingMethod",
    "PDEMethod",
    "PDESpaceGrid",
    "PDEEarlyExercise",
    "DayCountConvention",
    "AsianAveraging",
    "GreekCalculationMethod",
    "ImpliedVolMethod",
    # Valuation
    "VanillaSpec",
    "OptionValuation",
    "UnderlyingData",
    "PayoffSpec",
    "AsianSpec",
    "MonteCarloParams",
    "BinomialParams",
    "PDEParams",
    "implied_volatility",
    # Exceptions
    "DerivativesAnalyticsError",
    "ValidationError",
    "ConfigurationError",
    "UnsupportedFeatureError",
    "NumericalError",
    "ArbitrageViolationError",
    "ConvergenceError",
    "StabilityError",
]
