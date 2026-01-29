"""Simulation module for hedging and PnL attribution.

This module provides:
- DeltaHedgingSimulator: Simulate delta hedging strategies
- PnL attribution: Decompose PnL into Greeks contributions
"""

from .delta_hedging import DeltaHedgingSimulator, HedgingParams, HedgingResult
from .pnl_attribution import PnLAttribution, attribute_pnl_taylor_expansion

__all__ = [
    "DeltaHedgingSimulator",
    "HedgingParams",
    "HedgingResult",
    "PnLAttribution",
    "attribute_pnl_taylor_expansion",
]
