"""Market data containers for valuation and simulation."""

from dataclasses import dataclass
import datetime as dt
import numpy as np
from .rates import DiscountCurve
from .exceptions import ValidationError


@dataclass(frozen=True, slots=True)
class MarketData:
    """Market data required for valuation/simulation."""

    pricing_date: dt.datetime
    discount_curve: DiscountCurve
    currency: str

    def __post_init__(self) -> None:
        if not isinstance(self.pricing_date, dt.datetime):
            raise ValidationError(
                f"pricing_date must be a datetime, got {type(self.pricing_date).__name__}"
            )
        if not isinstance(self.discount_curve, DiscountCurve):
            raise ValidationError(
                f"discount_curve must be a DiscountCurve, got {type(self.discount_curve).__name__}"
            )
        if not isinstance(self.currency, str) or not self.currency:
            raise ValidationError("currency must be a non-empty string")


@dataclass(frozen=True, slots=True)
class CorrelationContext:
    """Shared correlation/scenario context for multi-asset simulation."""

    cholesky_matrix: np.ndarray  # shape (n_assets, n_assets)
    random_numbers: np.ndarray  # shape (n_assets, n_time_intervals, n_paths)
    rn_set: dict[str, int]  # maps asset name -> index in random_numbers
