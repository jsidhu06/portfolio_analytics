"""Market data containers for valuation and simulation."""

from dataclasses import dataclass
import datetime as dt
import numpy as np
from .rates import DiscountCurve


@dataclass(frozen=True, slots=True)
class MarketData:
    """Market data required for valuation/simulation."""

    pricing_date: dt.datetime
    discount_curve: DiscountCurve
    currency: str


@dataclass(frozen=True, slots=True)
class CorrelationContext:
    """Shared correlation/scenario context for multi-asset simulation."""

    cholesky_matrix: np.ndarray  # shape (n_assets, n_assets)
    random_numbers: np.ndarray  # shape (n_assets, n_time_intervals, n_paths)
    rn_set: dict[str, int]  # maps asset name -> index in random_numbers
