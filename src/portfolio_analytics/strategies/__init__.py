"""Example option strategy specifications.

This package contains lightweight *strategy specs* (e.g., condor) that are useful for
notebooks and experimentation.

Core valuation (`OptionValuation`) intentionally focuses on single-instrument contracts
(vanilla options, custom single-contract payoffs) rather than multi-leg strategy objects.
"""

from .condor import CondorSpec

__all__ = ["CondorSpec"]
