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


@dataclass(slots=True)
class ValuationEnvironment:
    """Valuation environment for derivatives portfolio.

    Contains configuration parameters for Monte Carlo simulation and valuation,
    along with generated time grid and special dates.

    Attributes
    ==========
    pricing_date: datetime
        valuation/pricing date
    discount_curve: ConstantShortRate
        discount rate curve for present value calculations
    frequency: str
        time grid frequency (e.g., 'W' for weekly, 'D' for daily)
    paths: int
        number of Monte Carlo simulation paths
    starting_date: datetime
        start date of simulation period
    final_date: datetime
        end date of simulation period
    day_count_convention: int
        day count basis (default 365)
    time_grid: np.ndarray, optional
        array of datetime objects in simulation time grid
    special_dates: list[datetime], optional
        important dates that must be included in time grid (e.g., maturity dates)
    """

    pricing_date: dt.datetime
    discount_curve: ConstantShortRate
    frequency: str
    paths: int
    starting_date: dt.datetime
    final_date: dt.datetime
    day_count_convention: int = 365
    time_grid: np.ndarray | None = None
    special_dates: list[dt.datetime] | None = None
