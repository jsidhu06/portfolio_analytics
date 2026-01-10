# pylint: disable=too-few-public-methods, missing-function-docstring
"""Model market environments for valuation."""

from dataclasses import dataclass
import datetime as dt
import numpy as np
from .rates import ConstantShortRate


class MarketEnvironment:
    """Class to model a market environment relevant for valuation.

    Attributes
    ==========
    name: string
        name of the market environment
    pricing_date: datetime object
        date of the market environment

    Methods
    =======
    add_constant:
        adds a constant (e.g. model parameter)
    get_constant:
        gets a constant
    add_list:
        adds a list (e.g. underlyings)
    get_list:
        gets a list
    add_curve:
        adds a market curve (e.g. yield curve)
    get_curve:
        gets a market curve
    add_environment:
        adds and overwrites whole market environments
        with constants, lists, and curves

    NOTE: get methods raise KeyError if key not found.
    """

    def __init__(self, name, pricing_date):
        self.name = name
        self.pricing_date = pricing_date
        self.constants = {}
        self.lists = {}
        self.curves = {}

    def add_constant(self, key, constant):
        self.constants[key] = constant

    def get_constant(self, key):
        return self.constants[key]

    def add_list(self, key, list_object):
        self.lists[key] = list_object

    def get_list(self, key):
        return self.lists[key]

    def add_curve(self, key, curve):
        self.curves[key] = curve

    def get_curve(self, key):
        return self.curves[key]

    def add_environment(self, env):
        # overwrites existing values, if they exist
        self.constants.update(env.constants)
        self.lists.update(env.lists)
        self.curves.update(env.curves)


@dataclass(frozen=True, slots=True)
class MarketData:
    """Market data required for valuation/simulation."""

    pricing_date: dt.datetime
    discount_curve: ConstantShortRate
    currency: str


@dataclass(frozen=True, slots=True)
class CorrelationContext:
    """Shared correlation/scenario context for multi-asset simulation."""

    cholesky_matrix: np.ndarray  # shape (n_assets, n_assets)
    random_numbers: np.ndarray  # shape (n_assets, n_time_intervals, n_paths)
    rn_set: dict[str, int]  # maps asset name -> index in random_numbers


@dataclass(frozen=True, slots=True)
class ValuationEnvironment:
    """Portfolio-level valuation inputs.

    This class is intentionally *not* a catch-all for portfolio-derived artifacts.
    Dates like a portfolio simulation start/end, a time grid, or special dates
    depend on the positions held and should be derived by the portfolio/scheduler.

    Attributes
    ==========
    market_data: MarketData
        Market data container.
    paths:
        Number of Monte Carlo paths.
    frequency:
        Time grid frequency (e.g. 'D', 'W', 'M').
    day_count_convention:
        Day count basis (default 365).
    """

    market_data: MarketData
    paths: int
    frequency: str
    day_count_convention: int = 365
