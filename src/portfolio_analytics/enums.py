"""Enums for option valuation."""

from enum import Enum


class OptionType(Enum):
    CALL = "call"
    PUT = "put"
    CONDOR = "condor"


class PositionSide(Enum):
    LONG = "long"
    SHORT = "short"


class ExerciseType(Enum):
    EUROPEAN = "european"
    AMERICAN = "american"


class PricingMethod(Enum):
    MONTE_CARLO = "monte_carlo"
    BINOMIAL = "binomial"
    BSM_CONTINUOUS = "bsm_continuous"
    BSM_DISCRETE = "bsm_discrete"


class GreekCalculationMethod(Enum):
    ANALYTICAL = "analytical"
    NUMERICAL = "numerical"
