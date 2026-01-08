"Path simulation classes for various stochastic processes"

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Protocol
import datetime as dt
import numpy as np
import pandas as pd
from .market_environment import MarketData, CorrelationContext
from .utils import calculate_year_fraction, sn_random_numbers


@dataclass(frozen=True)
class SimulationConfig:
    paths: int
    frequency: str
    final_date: dt.datetime
    day_count_convention: int | float = 365
    time_grid: np.ndarray | None = None  # optional portfolio override
    special_dates: list[dt.datetime] = field(default_factory=list)


class DiffusionParams(Protocol):
    initial_value: int | float
    volatility: float


@dataclass(frozen=True, slots=True, kw_only=True)
class GBMParams:
    initial_value: float
    volatility: float


@dataclass(frozen=True, slots=True)
class JDParams:
    initial_value: float
    volatility: float
    jump_intensity: float  # lambda (per year)
    jump_mean: float  # mu_J (mean of log jump size)
    jump_std: float  # delta_J (std of log jump size)


@dataclass(frozen=True, slots=True)
class SRDParams:
    initial_value: float
    volatility: float
    kappa: float  # mean reversion speed
    theta: float  # long-run mean


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
        process_params: DiffusionParams,
        sim: SimulationConfig,
        corr: CorrelationContext | None = None,
    ):
        self.name = name
        self.pricing_date = market_data.pricing_date
        self.currency = market_data.currency
        self.discount_curve = market_data.discount_curve

        self.initial_value = process_params.initial_value
        self.volatility = process_params.volatility

        self.paths = sim.paths
        self.frequency = sim.frequency
        self.day_count_convention = sim.day_count_convention
        self.time_grid = sim.time_grid
        self.special_dates = sim.special_dates

        # horizon / final_date logic
        if self.time_grid is not None:
            self.final_date = max(self.time_grid)
        else:
            self.final_date = sim.final_date

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
        end = self.final_date
        # pandas date_range function; see
        # https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#timeseries-offset-aliases
        # for frequencies
        time_grid = list(pd.date_range(start=start, end=end, freq=self.frequency).to_pydatetime())
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
            # delete duplicates
            time_grid = list(set(time_grid))
            # sort list
            time_grid.sort()
        self.time_grid = np.array(time_grid)

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

        instrument_values = self.generate_paths(random_seed=random_seed)
        return instrument_values

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

            drift = (short_rate - 0.5 * self.volatility**2) * delta_t
            diffusion = self.volatility * np.sqrt(delta_t) * ran
            # generate simulated values for the respective date
            paths[t] = paths[t - 1] * np.exp(drift + diffusion)

        self.instrument_values = paths
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

        self.instrument_values = paths
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
        process_params,  # expects: initial_value, volatility, lamb, mu, delta
        sim: SimulationConfig,
        corr: CorrelationContext | None = None,
    ):
        super().__init__(name, market_data, process_params, sim, corr=corr)
        self.lamb = process_params.lamb  # jump intensity (per year)
        self.mu = process_params.mu  # mean of log jump size
        self.delta = process_params.delta  # std of log jump size

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

        lam = self.lamb
        mu_j = self.mu
        sig_j = self.delta
        vol = self.volatility

        # compensator k = E[e^Y - 1]
        k = np.exp(mu_j + 0.5 * sig_j**2) - 1.0

        rng = np.random.default_rng(random_seed)

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
            drift = (r - lam * k - 0.5 * vol**2) * delta_t
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

        self.instrument_values = paths
        return paths
