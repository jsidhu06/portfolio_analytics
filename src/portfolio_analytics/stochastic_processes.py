"Path simulation classes for various stochastic processes"

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace as dc_replace
import copy
import datetime as dt
import numpy as np
import pandas as pd
from .market_environment import MarketData, CorrelationContext
from .enums import DayCountConvention
from .utils import calculate_year_fraction, sn_random_numbers


@dataclass(frozen=True)
class SimulationConfig:
    """Configuration for a path simulation run.

    Notes
    -----
        - The simulation **start date** is taken from ``MarketData.pricing_date`` (the process pricing date).
            Therefore this config only specifies the horizon **end** (``end_date``) and the discretization.

        Grid specification
        ------------------
        Exactly one of the following modes must be used:

        1) Explicit grid: provide ``time_grid`` (and do not provide ``end_date``, ``frequency``, or ``num_steps``).
        2) Calendar grid: provide ``end_date`` + ``frequency``.
        3) Uniform-step grid: provide ``end_date`` + ``num_steps``.
    """

    paths: int
    frequency: str | None = None
    end_date: dt.datetime | None = None
    num_steps: int | None = None
    day_count_convention: DayCountConvention = DayCountConvention.ACT_365F
    time_grid: np.ndarray | None = None  # optional portfolio override
    special_dates: set[dt.datetime] = field(default_factory=set)

    def __post_init__(self):
        if self.paths is None or int(self.paths) <= 0:
            raise ValueError("SimulationConfig.paths must be a positive integer")

        has_end_date = self.end_date is not None
        has_time_grid = self.time_grid is not None
        has_frequency = self.frequency is not None
        has_num_steps = self.num_steps is not None

        # Explicit grid mode
        if has_time_grid:
            if has_end_date or has_frequency or has_num_steps:
                raise ValueError(
                    "When time_grid is provided, end_date, frequency, and num_steps must be omitted."
                )
            if len(self.time_grid) == 0:
                raise ValueError("SimulationConfig.time_grid must be non-empty when provided")
            return

        # Otherwise we require an end_date and one discretization knob.
        if not has_end_date:
            raise ValueError("SimulationConfig.end_date must be provided when time_grid is not set")

        if has_frequency == has_num_steps:
            raise ValueError(
                "SimulationConfig requires exactly one of frequency or num_steps when end_date is provided."
            )

        if has_frequency:
            if not isinstance(self.frequency, str) or not self.frequency.strip():
                raise ValueError("SimulationConfig.frequency must be a non-empty string")

        if has_num_steps:
            try:
                steps = int(self.num_steps)
            except (TypeError, ValueError) as exc:
                raise TypeError("SimulationConfig.num_steps must be an integer") from exc
            if steps <= 0:
                raise ValueError("SimulationConfig.num_steps must be a positive integer")


@dataclass(frozen=True, slots=True, kw_only=True)
class GBMParams:
    initial_value: float
    volatility: float
    dividend_yield: float = 0.0
    discrete_dividends: list[tuple[dt.datetime, float]] | None = None

    def __post_init__(self):
        if self.initial_value is None:
            raise ValueError("GBMParams requires initial_value to be not None")
        if self.volatility is None:
            raise ValueError("GBMParams requires volatility to be not None")
        if not np.isfinite(float(self.initial_value)):
            raise ValueError("GBMParams requires initial_value to be finite")
        if not np.isfinite(float(self.volatility)):
            raise ValueError("GBMParams requires volatility to be finite")
        if float(self.volatility) < 0.0:
            raise ValueError("GBMParams requires volatility to be >= 0")
        if self.discrete_dividends is not None and self.dividend_yield != 0.0:
            raise ValueError(
                "Provide either dividend_yield or discrete_dividends in GBMParams, not both"
            )


@dataclass(frozen=True, slots=True, kw_only=True)
class JDParams:
    initial_value: float
    volatility: float
    lambd: float  # lambda (per year)
    mu: float  # mu_J (mean of log jump size)
    delta: float  # delta_J (std of log jump size)
    dividend_yield: float = 0.0
    discrete_dividends: list[tuple[dt.datetime, float]] | None = None

    def __post_init__(self):
        if self.initial_value is None:
            raise ValueError("JDParams requires initial_value to be not None")
        if self.volatility is None:
            raise ValueError("JDParams requires volatility to be not None")
        if not np.isfinite(float(self.initial_value)):
            raise ValueError("JDParams requires initial_value to be finite")
        if not np.isfinite(float(self.volatility)):
            raise ValueError("JDParams requires volatility to be finite")
        if float(self.volatility) < 0.0:
            raise ValueError("JDParams requires volatility to be >= 0")

        if self.lambd is None or self.mu is None or self.delta is None:
            raise ValueError("JDParams requires lambd, mu, and delta to be not None")

        if not np.isfinite(float(self.lambd)):
            raise ValueError("JDParams requires lambd to be finite")
        if not np.isfinite(float(self.mu)):
            raise ValueError("JDParams requires mu to be finite")
        if not np.isfinite(float(self.delta)):
            raise ValueError("JDParams requires delta to be finite")
        if float(self.lambd) < 0.0:
            raise ValueError("JDParams requires lambd to be >= 0")
        if float(self.delta) < 0.0:
            raise ValueError("JDParams requires delta to be >= 0")
        if self.discrete_dividends is not None and self.dividend_yield != 0.0:
            raise ValueError(
                "Provide either dividend_yield or discrete_dividends in JDParams, not both"
            )


@dataclass(frozen=True, slots=True, kw_only=True)
class SRDParams:
    initial_value: float
    volatility: float
    kappa: float  # mean reversion speed
    theta: float  # long-run mean

    def __post_init__(self):
        if self.initial_value is None:
            raise ValueError("SRDParams requires initial_value to be not None")
        if self.volatility is None:
            raise ValueError("SRDParams requires volatility to be not None")
        if not np.isfinite(float(self.initial_value)):
            raise ValueError("SRDParams requires initial_value to be finite")
        if not np.isfinite(float(self.volatility)):
            raise ValueError("SRDParams requires volatility to be finite")
        if float(self.volatility) < 0.0:
            raise ValueError("SRDParams requires volatility to be >= 0")

        if self.kappa is None or self.theta is None:
            raise ValueError("SRDParams requires kappa and theta to be not None")

        if not np.isfinite(float(self.kappa)):
            raise ValueError("SRDParams requires kappa to be finite")
        if not np.isfinite(float(self.theta)):
            raise ValueError("SRDParams requires theta to be finite")
        if float(self.kappa) < 0.0:
            raise ValueError("SRDParams requires kappa to be >= 0")
        if float(self.theta) < 0.0:
            raise ValueError("SRDParams requires theta to be >= 0")


class PathSimulation(ABC):
    """Providing base methods for simulation classes.

    Attributes
    ==========
    name: str
        Name of the simulation object (typically the underlying identifier).
    market_data: MarketData
        Market data required for simulation (pricing date, discount curve, currency).
    process_params:
        Model-specific parameters for the stochastic process
        (e.g. GBMParams, JDParams, SRDParams).
    sim: SimulationConfig
        Simulation configuration (paths, frequency, time grid, special dates).
    correlation_context: CorrelationContext or None
        Shared correlation/scenario context used in multi-asset simulations.
        If None, the process is simulated independently.

    Methods
    =======
    generate_time_grid:
        returns time grid for simulation
    get_instrument_values:
        returns the current instrument values (array)
    """

    def __init__(
        self,
        name: str,
        market_data: MarketData,
        process_params: GBMParams | JDParams | SRDParams,
        sim: SimulationConfig,
        corr: CorrelationContext | None = None,
    ):
        self.name = name
        self.market_data = market_data

        self.initial_value = process_params.initial_value
        self.volatility = process_params.volatility
        if isinstance(process_params, (GBMParams, JDParams)):
            self.dividend_yield = process_params.dividend_yield
            self.discrete_dividends = list(process_params.discrete_dividends or [])
        else:
            self.discrete_dividends = []

        self.paths = sim.paths
        self.frequency = sim.frequency
        self.num_steps = sim.num_steps
        self.day_count_convention = sim.day_count_convention
        self.time_grid = sim.time_grid
        self.special_dates = set(sim.special_dates)
        if self.discrete_dividends:
            for ex_date, _ in self.discrete_dividends:
                self.special_dates.add(ex_date)

        # horizon / end_date logic
        if self.time_grid is not None:
            self.end_date = max(self.time_grid)
        else:
            self.end_date = sim.end_date

        self.instrument_values = None
        self.correlation_context = corr

        if corr is not None:
            # only needed in a portfolio context when
            # risk factors are correlated
            self.cholesky_matrix = corr.cholesky_matrix
            self.rn_set = corr.rn_set[self.name]
            self.random_numbers = corr.random_numbers

    def generate_time_grid(self) -> None:
        "Generate time grid for simulation of stochastic process"
        start = self.pricing_date
        end = self.end_date
        if self.num_steps is not None:
            time_grid = list(
                pd.date_range(start=start, end=end, periods=int(self.num_steps) + 1).to_pydatetime()
            )
        else:
            time_grid = list(
                pd.date_range(start=start, end=end, freq=self.frequency).to_pydatetime()
            )
        # enhance time_grid by start, end, and special_dates
        if start not in time_grid:
            time_grid.insert(0, start)
            # insert start date if not in list
        if end not in time_grid:
            time_grid.append(end)
            # insert end date if not in list
        if self.special_dates:
            # add all special dates
            time_grid.extend(self.special_dates)
            # delete duplicates and sort
            time_grid = sorted(set(time_grid))
        self.time_grid = np.array(time_grid)

    @property
    def pricing_date(self) -> dt.datetime:
        return self.market_data.pricing_date

    @property
    def currency(self) -> str:
        return self.market_data.currency

    @property
    def discount_curve(self):
        return self.market_data.discount_curve

    def replace(self, **kwargs) -> "PathSimulation":
        """Return a shallow-cloned instance with selected attributes replaced.

        This avoids in-place mutation and clears cached instrument values by default,
        which is important for bump-and-revalue workflows and thread safety.
        """
        allowed_fields = {
            "name",
            "market_data",
            "initial_value",
            "volatility",
            "dividend_yield",
            "discrete_dividends",
            "paths",
            "frequency",
            "num_steps",
            "day_count_convention",
            "time_grid",
            "special_dates",
            "end_date",
            "pricing_date",
            "discount_curve",
            "currency",
        }
        unknown = set(kwargs).difference(allowed_fields)
        if unknown:
            raise ValueError(
                f"replace() only supports overriding {sorted(allowed_fields)}; "
                f"unknown: {sorted(unknown)}"
            )

        cloned = copy.copy(self)

        # Detach mutable containers to avoid shared state across clones.
        cloned.market_data = self.market_data  # immutable so fine
        cloned.special_dates = set(self.special_dates)
        cloned.discrete_dividends = list(self.discrete_dividends)
        cloned.time_grid = None if self.time_grid is None else np.array(self.time_grid, copy=True)

        # Always reset cached paths unless explicitly overridden.
        cloned.instrument_values = None

        # Apply replacements (skip market_data & market_data-derived keys)
        deferred = {"market_data", "pricing_date", "discount_curve", "currency"}
        for key, value in kwargs.items():
            if key in deferred:
                continue
            setattr(cloned, key, value)

        # Normalize list-like overrides
        if "special_dates" in kwargs:
            cloned.special_dates = set(kwargs["special_dates"] or [])

        if "discrete_dividends" in kwargs:
            cloned.discrete_dividends = list(kwargs["discrete_dividends"] or [])

        if "time_grid" in kwargs:
            cloned.time_grid = (
                None if kwargs["time_grid"] is None else np.array(kwargs["time_grid"], copy=True)
            )
            if cloned.time_grid is not None and len(cloned.time_grid) > 0:
                cloned.end_date = max(cloned.time_grid)
        else:
            if any(
                key in kwargs
                for key in ("pricing_date", "end_date", "frequency", "num_steps", "special_dates")
            ):
                cloned.time_grid = None

        # Update market_data if any related fields were overridden
        if "market_data" in kwargs:
            cloned.market_data = kwargs["market_data"]
        else:
            updates = {}
            if "pricing_date" in kwargs:
                updates["pricing_date"] = kwargs["pricing_date"]
            if "discount_curve" in kwargs:
                updates["discount_curve"] = kwargs["discount_curve"]
            if "currency" in kwargs:
                updates["currency"] = kwargs["currency"]
            if updates:
                cloned.market_data = dc_replace(cloned.market_data, **updates)

        # Ensure discrete dividend dates are included as special dates.
        if cloned.discrete_dividends:
            for ex_date, _ in cloned.discrete_dividends:
                cloned.special_dates.add(ex_date)

        return cloned

    def get_instrument_values(self, random_seed: int | None = None) -> np.ndarray:
        """Generate paths and get instrument values matrix

        Parameters
        ==========
        random_seed: int, optional
            random seed for path generation

        Returns
        =======
        instrument_values: np.ndarray
            simulated instrument value paths
        """

        self.instrument_values = self.generate_paths(random_seed=random_seed)
        return self.instrument_values

    @abstractmethod
    def generate_paths(
        self,
        random_seed: int | None = None,
    ) -> None:
        """Generate paths for the stochastic process.

        Subclasses must implement this method to define their specific
        path generation logic.

        Parameters
        ==========
        random_seed: int, optional
            random seed for reproducibility

        Raises
        ======
        NotImplementedError
            if subclass does not implement this method
        """
        raise NotImplementedError("Subclasses must implement generate_paths method")


class GeometricBrownianMotion(PathSimulation):
    """Class to generate simulated paths based on
    the Black-Scholes-Merton geometric Brownian motion model.
    """

    def generate_paths(self, random_seed: int | None = None) -> None:
        """Generate geometric Brownian motion paths.

        Implements the classic Black-Scholes-Merton model:
        dS_t = mu * S_t * dt + sigma * S_t * dW_t

        Parameters
        ==========
        random_seed: int, optional
            random seed for reproducibility

        Notes
        =====
        The drift term (mu) is derived from the risk-free rate
        to ensure the model is calibrated to the discount curve.
        """
        if self.time_grid is None:
            # method from generic simulation class
            self.generate_time_grid()
        # number of dates (timesteps) for time grid
        M = len(self.time_grid)
        # number of paths
        num_paths = self.paths
        # ndarray initialization for path simulation
        paths = np.zeros((M, num_paths), dtype=float)
        # initialize first date with initial_value
        paths[0] = self.initial_value
        if self.correlation_context is None:
            # if not correlated, generate random numbers
            rand = sn_random_numbers(
                (1, M, num_paths), random_seed=random_seed
            )  # shape (M,num_paths)
        else:
            # if correlated, use random number object as provided
            # in market environment
            rand = self.random_numbers

        # get short rate for drift of process
        short_rate = self.discount_curve.short_rate

        dividend_by_date: dict[dt.datetime, float] = {}
        if self.discrete_dividends:
            time_set = set(self.time_grid)
            for ex_date, amount in self.discrete_dividends:
                if ex_date in time_set:
                    dividend_by_date[ex_date] = dividend_by_date.get(ex_date, 0.0) + float(amount)

        for t in range(1, M):
            # select the right time slice from the relevant
            # random number set
            if self.correlation_context is None:
                ran = rand[t]  # shape (num_paths,)
            else:
                ran = np.dot(self.cholesky_matrix, rand[:, t, :])
                ran = ran[self.rn_set]  # shape (num_paths,)

            # difference between two dates as year fraction
            delta_t = calculate_year_fraction(
                self.time_grid[t - 1],
                self.time_grid[t],
                day_count_convention=self.day_count_convention,
            )

            drift = (short_rate - self.dividend_yield - 0.5 * self.volatility**2) * delta_t

            diffusion = self.volatility * np.sqrt(delta_t) * ran
            # generate simulated values for the respective date
            paths[t] = paths[t - 1] * np.exp(drift + diffusion)

            div_amt = dividend_by_date.get(self.time_grid[t])
            if div_amt is not None:
                paths[t] = np.maximum(paths[t] - div_amt, 0.0)

        return paths


class JumpDiffusion(PathSimulation):
    """Merton (1976) jump diffusion (lognormal jumps).

    Risk-neutral discretisation (per step Δt):
        S_{t+Δt} = S_t * exp((r - λk - 0.5σ^2)Δt + σ√Δt Z)
                        * exp( Σ_{i=1}^{N} Y_i )

    where:
        N ~ Poisson(λΔt)
        Y_i ~ Normal(μ, δ^2)
        k = E[e^Y - 1] = exp(μ + 0.5δ^2) - 1
    """

    def __init__(
        self,
        name: str,
        market_data: MarketData,
        process_params,  # expects: initial_value, volatility, lambd, mu, delta
        sim: SimulationConfig,
        corr: CorrelationContext | None = None,
    ):
        super().__init__(name, market_data, process_params, sim, corr=corr)
        self.lambd = process_params.lambd  # lambda (average number of jumps per year)
        self.mu = process_params.mu  # mu_J (mean of log jump size)
        self.delta = process_params.delta  # delta_J (std of log jump size)

    def generate_paths(self, random_seed: int | None = None) -> np.ndarray:
        # TO DO: Check these calcs
        if self.time_grid is None:
            self.generate_time_grid()

        M = len(self.time_grid)
        num_paths = self.paths

        # ndarray initialization for path simulation
        paths = np.zeros((M, num_paths), dtype=float)
        # initialize first date with initial_value
        paths[0] = self.initial_value
        # risk-free rate (adjust if your curve API differs)
        r = float(self.discount_curve.short_rate)

        lam = self.lambd
        mu_j = self.mu
        sig_j = self.delta
        vol = self.volatility

        # compensator k = E[e^Y - 1]
        k = np.exp(mu_j + 0.5 * sig_j**2) - 1.0

        rng = np.random.default_rng(random_seed)

        dividend_by_date: dict[dt.datetime, float] = {}
        if self.discrete_dividends:
            time_set = set(self.time_grid)
            for ex_date, amount in self.discrete_dividends:
                if ex_date in time_set:
                    dividend_by_date[ex_date] = dividend_by_date.get(ex_date, 0.0) + float(amount)

        for t in range(1, M):
            delta_t = calculate_year_fraction(
                self.time_grid[t - 1],
                self.time_grid[t],
                day_count_convention=self.day_count_convention,
            )

            # Diffusion shock Z
            if self.correlation_context is not None:
                # random_numbers shape is (n_assets, M, num_paths)
                z = self.correlation_context.random_numbers[:, t, :]  # (N,I)
                ran_all = self.correlation_context.cholesky_matrix @ z  # (N,I)
                Z = ran_all[self.rn_set]  # (I,)
            else:
                Z = rng.standard_normal(num_paths)  # (I,)

            # Diffusion multiplier
            drift = (r - self.dividend_yield - lam * k - 0.5 * vol**2) * delta_t
            diffusion = vol * np.sqrt(delta_t) * Z
            diffusion_multiplier = np.exp(drift + diffusion)

            # Jump multiplier: exp(sum of jump sizes)
            N = rng.poisson(lam * delta_t, size=num_paths)  # number of jumps in (t-1, t]
            # sum of N normals: Normal(N*mu, N*delta^2)
            jump_sum = np.where(
                N > 0,
                N * mu_j + np.sqrt(N) * sig_j * rng.standard_normal(num_paths),
                0.0,
            )
            jump_multiplier = np.exp(jump_sum)

            paths[t] = paths[t - 1] * diffusion_multiplier * jump_multiplier

            div_amt = dividend_by_date.get(self.time_grid[t])
            if div_amt is not None:
                paths[t] = np.maximum(paths[t] - div_amt, 0.0)

        return paths


class SquareRootDiffusion(PathSimulation):
    """Cox–Ingersoll–Ross (1985) square-root diffusion (CIR/SRD).

    Model (under chosen measure):
        dX_t = kappa*(theta - X_t) dt + sigma*sqrt(X_t) dW_t

    Uses full truncation Euler to preserve non-negativity.
    """

    def __init__(
        self,
        name: str,
        market_data: MarketData,
        process_params,  # should provide initial_value, volatility, kappa, theta
        sim: SimulationConfig,
        corr: CorrelationContext | None = None,
    ):
        super().__init__(name, market_data, process_params, sim, corr=corr)
        self.kappa = process_params.kappa
        self.theta = process_params.theta

    def generate_paths(self, random_seed: int | None = None) -> np.ndarray:
        """Generate Cox-Ingersoll-Ross (square-root diffusion) paths."""
        if self.time_grid is None:
            self.generate_time_grid()

        M = len(self.time_grid)
        num_paths = self.paths

        paths = np.zeros((M, num_paths), dtype=float)
        paths_hat = np.zeros_like(paths)

        paths[0] = self.initial_value
        paths_hat[0] = self.initial_value

        if self.correlation_context is None:
            rand = sn_random_numbers((1, M, num_paths), random_seed=random_seed)
        else:
            rand = self.random_numbers

        for t in range(1, M):
            delta_t = calculate_year_fraction(
                self.time_grid[t - 1],
                self.time_grid[t],
                day_count_convention=self.day_count_convention,
            )
            if self.correlation_context is None:
                ran = rand[t]
            else:
                ran = np.dot(self.cholesky_matrix, rand[:, t, :])
                ran = ran[self.rn_set]

            # full truncation Euler discretization
            mean_reversion = self.kappa * (self.theta - paths[t - 1, :]) * delta_t
            diffusion = np.sqrt(paths[t - 1, :]) * self.volatility * np.sqrt(delta_t) * ran
            paths_hat[t] = paths_hat[t - 1] + mean_reversion + diffusion
            paths[t] = np.maximum(0, paths_hat[t])

        return paths
