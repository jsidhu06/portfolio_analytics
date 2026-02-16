"""Enums for option valuation."""

from enum import Enum

__all__ = [
    "OptionType",
    "AsianAveraging",
    "PositionSide",
    "ExerciseType",
    "PricingMethod",
    "GreekCalculationMethod",
    "ImpliedVolMethod",
    "PDEMethod",
    "PDESpaceGrid",
    "PDEEarlyExercise",
    "DayCountConvention",
]


class OptionType(Enum):
    CALL = "call"
    PUT = "put"


class AsianAveraging(Enum):
    ARITHMETIC = "arithmetic"
    GEOMETRIC = "geometric"


class PositionSide(Enum):
    LONG = "long"
    SHORT = "short"


class ExerciseType(Enum):
    EUROPEAN = "european"
    AMERICAN = "american"


class PricingMethod(Enum):
    MONTE_CARLO = "monte_carlo"
    BINOMIAL = "binomial"
    BSM = "bsm"
    PDE_FD = "pde_fd"


class GreekCalculationMethod(Enum):
    ANALYTICAL = "analytical"
    NUMERICAL = "numerical"


class ImpliedVolMethod(Enum):
    NEWTON_RAPHSON = "newton_raphson"
    BISECTION = "bisection"
    BRENTQ = "brentq"


class PDEMethod(Enum):
    IMPLICIT = "implicit"
    EXPLICIT = "explicit"
    CRANK_NICOLSON = "crank_nicolson"


class PDESpaceGrid(Enum):
    SPOT = "spot"
    LOG_SPOT = "log_spot"


class PDEEarlyExercise(Enum):
    INTRINSIC = "intrinsic"
    GAUSS_SEIDEL = "gauss_seidel"


class DayCountConvention(Enum):
    ACT_360 = "ACT/360"
    ACT_365F = "ACT/365F"
    ACT_365_25 = "ACT/365.25"
    THIRTY_360_US = "30/360 US"
