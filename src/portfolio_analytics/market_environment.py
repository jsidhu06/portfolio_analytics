# pylint: disable=too-few-public-methods, missing-function-docstring
"""Model market environments for valuation."""

from dataclasses import dataclass
import datetime as dt
import numpy as np
from .enums import DayCountConvention
from .rates import ConstantShortRate


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
        Day count basis (default ACT/365F).
    """

    market_data: MarketData
    paths: int
    frequency: str
    day_count_convention: DayCountConvention = DayCountConvention.ACT_365F
