"""Parameter classes for method-specific valuation configuration.

Each pricing method (Monte Carlo, Binomial, etc.) has its own parameter class
that explicitly documents the configuration options available for that method.
"""

from dataclasses import dataclass
import warnings

from ..enums import PDEEarlyExercise, PDEMethod, PDESpaceGrid


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
    mc_paths:
        Number of Monte Carlo paths when sampling the binomial tree
        (used for path-dependent payoffs like Asian options). If None,
        Monte Carlo sampling is disabled and Hull-style averages are used.
    random_seed:
        Random seed for binomial-tree Monte Carlo sampling. This param is ignored
        if mc_paths is None.
    asian_tree_averages:
        Number of representative averages per node for Hull-style Asian
        binomial tree valuation. Used when mc_paths is None.
        Practical guidance: keep asian_tree_averages on the same order as
        num_steps (roughly 0.5x to 1.0x) to reduce interpolation bias.
        Larger values improve stability but increase memory usage as
        O(asian_tree_averages * num_steps^2).
    """

    num_steps: int = 500
    mc_paths: int | None = None
    random_seed: int | None = None
    asian_tree_averages: int | None = None

    def __post_init__(self):
        if self.num_steps < 1:
            raise ValueError(f"num_steps must be >= 1, got {self.num_steps}")
        if self.mc_paths is not None and self.asian_tree_averages is not None:
            raise ValueError("Only one of mc_paths and asian_tree_averages can be set, got both")
        if self.mc_paths is not None and self.mc_paths < 1:
            raise ValueError(f"mc_paths must be >= 1, got {self.mc_paths}")
        if self.asian_tree_averages is not None and self.asian_tree_averages < 1:
            raise ValueError(f"asian_tree_averages must be >= 1, got {self.asian_tree_averages}")
        if self.asian_tree_averages is not None:
            ratio = self.asian_tree_averages / self.num_steps
            if ratio < 0.5:
                warnings.warn(
                    "asian_tree_averages is small relative to num_steps; "
                    "Hull-style Asian valuation may be biased upward. "
                    "Consider asian_tree_averages >= 0.5 * num_steps.",
                    RuntimeWarning,
                )
            if ratio > 2.0:
                warnings.warn(
                    "asian_tree_averages is large relative to num_steps; "
                    "memory usage may be high with limited accuracy gains.",
                    RuntimeWarning,
                )
            est_bytes = self.asian_tree_averages * (self.num_steps + 1) ** 2 * 8
            if est_bytes > 1_000_000_000:
                est_gib = est_bytes / (1024**3)
                warnings.warn(
                    f"Estimated memory for Hull Asian grid is ~{est_gib:.2f} GiB; "
                    "consider reducing num_steps or asian_tree_averages.",
                    RuntimeWarning,
                )


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
        method: Time-stepping scheme for the FD solver.
        space_grid: Spatial discretization grid in spot or log-spot space.
        american_solver: Early exercise handling for American options.
    """

    smax_mult: float = 4.0
    spot_steps: int = 200
    time_steps: int = 200
    omega: float = 1.5
    tol: float = 1e-6
    max_iter: int = 20_000
    method: PDEMethod | str = PDEMethod.CRANK_NICOLSON
    space_grid: PDESpaceGrid | str = PDESpaceGrid.SPOT
    american_solver: PDEEarlyExercise | str = PDEEarlyExercise.GAUSS_SEIDEL

    def __post_init__(self):
        if isinstance(self.method, str):
            object.__setattr__(self, "method", PDEMethod(self.method))
        if isinstance(self.space_grid, str):
            object.__setattr__(self, "space_grid", PDESpaceGrid(self.space_grid))
        if isinstance(self.american_solver, str):
            object.__setattr__(self, "american_solver", PDEEarlyExercise(self.american_solver))
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
        if not isinstance(self.method, PDEMethod):
            raise ValueError(f"method must be a PDEMethod, got {self.method}")
        if not isinstance(self.space_grid, PDESpaceGrid):
            raise ValueError(f"space_grid must be a PDESpaceGrid, got {self.space_grid}")
        if not isinstance(self.american_solver, PDEEarlyExercise):
            raise ValueError(
                f"american_solver must be a PDEEarlyExercise, got {self.american_solver}"
            )


# Type alias for any valuation parameters
ValuationParams = MonteCarloParams | BinomialParams | PDEParams
