"""Enums for option valuation."""

from enum import Enum


class OptionType(Enum):
    CALL = "call"
    PUT = "put"
    CONDOR = "condor"
    CUSTOM = "custom"
    ASIAN_ARITHMETIC = "asian_arithmetic"
    ASIAN_GEOMETRIC = "asian_geometric"


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
