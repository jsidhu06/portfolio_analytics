"""Enums for option valuation."""

from enum import Enum


class OptionType(Enum):
    CALL = "call"
    PUT = "put"


class AsianAveraging(Enum):
    ARITHMETIC = "arithmetic"
    GEOMETRIC = "geometric"


class BarrierType(Enum):
    UP_AND_IN = "up_and_in"
    UP_AND_OUT = "up_and_out"
    DOWN_AND_IN = "down_and_in"
    DOWN_AND_OUT = "down_and_out"


class BarrierMonitoring(Enum):
    CONTINUOUS = "continuous"
    DISCRETE = "discrete"


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
    HESTON = "heston"


class GreekCalculationMethod(Enum):
    ANALYTICAL = "analytical"
    NUMERICAL = "numerical"
