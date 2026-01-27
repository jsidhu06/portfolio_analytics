"""Parameter classes for method-specific valuation configuration.

Each pricing method (Monte Carlo, Binomial, etc.) has its own parameter class
that explicitly documents the configuration options available for that method.
"""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class MonteCarloParams:
    """Parameters for Monte Carlo option valuation.

    Attributes
    ==========
    random_seed:
        Random seed for reproducibility. If None, uses random state.
    deg:
        Polynomial degree for Longstaff-Schwartz regression (American only).
        Typical range: 2-5. Default: 2.
    """

    random_seed: int | None = None
    deg: int = 2

    def __post_init__(self):
        if self.deg is not None and self.deg < 1:
            raise ValueError(f"deg must be >= 1, got {self.deg}")


@dataclass(frozen=True, slots=True)
class BinomialParams:
    """Parameters for binomial tree option valuation.

    Attributes
    ==========
    num_steps:
        Number of time steps in the binomial tree.
        More steps increase accuracy but also computation time.
        Default: 500.
    """

    num_steps: int = 500

    def __post_init__(self):
        if self.num_steps < 1:
            raise ValueError(f"num_steps must be >= 1, got {self.num_steps}")


@dataclass(frozen=True, slots=True)
class PDEParams:
    """Parameters for PDE finite difference option valuation.

    Attributes:
        smax_mult: Multiplier for maximum spot price in grid.
                   Grid extends from 0 to smax_mult * max(spot, strike).
                   Default: 4.0
           spot_steps: Number of spatial (spot price) steps in the grid.
                    More steps improve accuracy. Typical: 200-500. Default: 200.
           time_steps: Number of time steps. More steps improve stability and accuracy.
                    Typical: 200-500. Default: 200.
        omega: SOR relaxation parameter for American options (PSOR algorithm).
               Range: (1.0, 2.0). Values > 1 accelerate convergence.
                Default: 1.5
        tol: Convergence tolerance for PSOR iterations (American only).
               Default: 1e-6
        max_iter: Maximum PSOR iterations per time step (American only).
                   Default: 20000
    """

    smax_mult: float = 4.0
    spot_steps: int = 200
    time_steps: int = 200
    omega: float = 1.5
    tol: float = 1e-6
    max_iter: int = 20_000

    def __post_init__(self):
        if self.smax_mult <= 0:
            raise ValueError(f"smax_mult must be positive, got {self.smax_mult}")
        if self.spot_steps < 3:
            raise ValueError(f"spot_steps must be >= 3, got {self.spot_steps}")
        if self.time_steps < 1:
            raise ValueError(f"time_steps must be >= 1, got {self.time_steps}")
        if not (1.0 < self.omega < 2.0):
            raise ValueError(f"omega must be in (1.0, 2.0), got {self.omega}")
        if self.tol <= 0:
            raise ValueError(f"tol must be positive, got {self.tol}")
        if self.max_iter < 1:
            raise ValueError(f"max_iter must be >= 1, got {self.max_iter}")


# Type alias for any valuation parameters
ValuationParams = MonteCarloParams | BinomialParams | PDEParams
