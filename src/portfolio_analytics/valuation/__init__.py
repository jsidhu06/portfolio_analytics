"""Option valuation and pricing engines.

This module provides a unified interface for pricing vanilla and custom options
using various methods: Monte Carlo simulation, Binomial trees, Black-Scholes-Merton
analytical formulas, and PDE finite difference methods.

Public API
----------
Core classes:
    OptionValuation: Main dispatcher for option pricing
    OptionSpec: Contract specification for vanilla options
    PayoffSpec: Contract specification for custom payoffs
    UnderlyingPricingData: Minimal underlying data container

Parameter classes:
    MonteCarloParams: Configuration for Monte Carlo pricing
    BinomialParams: Configuration for Binomial tree pricing
    PDEParams: Configuration for PDE finite difference pricing
    ValuationParams: Union type for all parameter classes
"""

from .core import (
    OptionValuation,
    OptionSpec,
    PayoffSpec,
    UnderlyingPricingData,
)
from .params import (
    MonteCarloParams,
    BinomialParams,
    PDEParams,
    ValuationParams,
)

__all__ = [
    # Core valuation classes
    "OptionValuation",
    "OptionSpec",
    "PayoffSpec",
    "UnderlyingPricingData",
    # Parameter classes
    "MonteCarloParams",
    "BinomialParams",
    "PDEParams",
    "ValuationParams",
]
