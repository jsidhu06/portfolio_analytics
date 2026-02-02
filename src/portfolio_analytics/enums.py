"""Enums for option valuation."""

from enum import Enum


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


class DayCountConvention(Enum):
    ACT_360 = "ACT/360"
    ACT_365F = "ACT/365F"
    ACT_365_25 = "ACT/365.25"
    THIRTY_360_US = "30/360 US"
