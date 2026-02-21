"""Top-level exports for common portfolio_analytics types."""

from importlib.metadata import version

__version__ = version("portfolio_analytics")

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
from .market_environment import MarketData
from .rates import DiscountCurve
from .stochastic_processes import (
    SimulationConfig,
    GBMParams,
    JDParams,
    SRDParams,
    GeometricBrownianMotion,
    JumpDiffusion,
    SquareRootDiffusion,
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
    PositionSide,
    GreekCalculationMethod,
    ImpliedVolMethod,
)
from .strategies import CondorSpec
from .valuation import (
    OptionSpec,
    OptionValuation,
    UnderlyingPricingData,
    PayoffSpec,
    AsianOptionSpec,
    MonteCarloParams,
    BinomialParams,
    PDEParams,
)
from .valuation.implied_volatility import implied_volatility

__all__ = [
    # Market data
    "MarketData",
    "DiscountCurve",
    # Stochastic processes
    "SimulationConfig",
    "GBMParams",
    "JDParams",
    "SRDParams",
    "GeometricBrownianMotion",
    "JumpDiffusion",
    "SquareRootDiffusion",
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
    "PositionSide",
    "GreekCalculationMethod",
    "ImpliedVolMethod",
    # Valuation
    "OptionSpec",
    "OptionValuation",
    "UnderlyingPricingData",
    "PayoffSpec",
    "AsianOptionSpec",
    "MonteCarloParams",
    "BinomialParams",
    "PDEParams",
    "implied_volatility",
    # Strategies
    "CondorSpec",
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
