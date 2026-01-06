"Path simulation classes for various stochastic processes"

from typing import Optional, Union
from abc import ABC, abstractmethod
import datetime as dt
import numpy as np
import pandas as pd
from .market_environment import MarketEnvironment
from .utils import calculate_year_fraction, sn_random_numbers


class PathSimulation(ABC):
    """Providing base methods for simulation classes.

    Attributes
    ==========
    name: str
        name of the object
    mar_env: MarketEnvironment
        market environment data for simulation
    corr: bool
        True if correlated with other model object

    Methods
    =======
    generate_time_grid:
        returns time grid for simulation
    get_instrument_values:
        returns the current instrument values (array)
    """

    def __init__(self, name: str, mar_env: MarketEnvironment, corr: bool = False):
        self.name = name
        self.pricing_date = mar_env.pricing_date
        self.initial_value = mar_env.get_constant("initial_value")
        self.volatility = mar_env.get_constant("volatility")
        self.final_date = mar_env.get_constant("final_date")
        self.currency = mar_env.get_constant("currency")
        self.frequency = mar_env.get_constant("frequency")
        self.paths = mar_env.get_constant("paths")
        self.discount_curve = mar_env.get_curve("discount_curve")
        # if time_grid in mar_env take that object
        # (for portfolio valuation)
        self.time_grid = mar_env.get_list("time_grid") if "time_grid" in mar_env.lists else None
        self.special_dates = (
            mar_env.get_list("special_dates") if "special_dates" in mar_env.lists else []
        )
        self.instrument_values = None
        self.correlated = corr

        if corr is True:
            # only needed in a portfolio context when
            # risk factors are correlated
            self.cholesky_matrix = mar_env.get_list("cholesky_matrix")
            self.rn_set = mar_env.get_list("rn_set")[self.name]
            self.random_numbers = mar_env.get_list("random_numbers")

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

    def get_instrument_values(
        self, random_seed: Optional[int] = None, day_count_convention: float = 365.0
    ) -> np.ndarray:
        """Get instrument values matrix; generate paths if not yet available.

        Parameters
        ==========
        random_seed: int, optional
            random seed for path generation
        day_count_convention: float, default 365.0
            day count convention (365, 360, etc.)

        Returns
        =======
        instrument_values: np.ndarray
            simulated instrument value paths
        """
        if self.instrument_values is None:
            # only initiate simulation if there are no instrument values
            self.generate_paths(random_seed=random_seed, day_count_convention=day_count_convention)
        elif random_seed is not None:
            # also initiate resimulation when random_seed is not None
            self.generate_paths(random_seed=random_seed, day_count_convention=day_count_convention)
        return self.instrument_values

    @abstractmethod
    def generate_paths(
        self, random_seed: Optional[int] = None, day_count_convention: float = 365.0
    ) -> None:
        """Generate paths for the stochastic process.

        Subclasses must implement this method to define their specific
        path generation logic.

        Parameters
        ==========
        random_seed: int, optional
            random seed for reproducibility
        day_count_convention: float, default 365.0
            day count convention for time calculations

        Raises
        ======
        NotImplementedError
            if subclass does not implement this method
        """
        raise NotImplementedError("Subclasses must implement generate_paths method")


class GeometricBrownianMotion(PathSimulation):
    """Class to generate simulated paths based on
    the Black-Scholes-Merton geometric Brownian motion model.

    Attributes
    ==========
    name: string
        name of the object
    mar_env: MarketEnvironment
        market environment data for simulation
    corr: Boolean
        True if correlated with other model simulation object

    Methods
    =======
    update:
        updates parameters
    generate_paths:
        returns Monte Carlo paths given the market environment
    """

    def update(
        self,
        initial_value: Optional[float] = None,
        volatility: Optional[float] = None,
        final_date: Optional[dt.datetime] = None,
    ) -> None:
        """Update parameters of the GBM model."""
        if initial_value is not None:
            self.initial_value = initial_value
        if volatility is not None:
            self.volatility = volatility
        if final_date is not None:
            self.final_date = final_date

        # reset instrument values to None
        self.instrument_values = None

    def generate_paths(
        self, random_seed: Optional[int] = None, day_count_convention: Union[int, float] = 365
    ) -> None:
        """Generate geometric Brownian motion paths.

        Implements the classic Black-Scholes-Merton model:
        dS_t = mu * S_t * dt + sigma * S_t * dW_t

        Parameters
        ==========
        random_seed: int, optional
            random seed for reproducibility
        day_count_convention: int or float, default 365
            day count convention for time calculations

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
        paths = np.zeros((M, num_paths))
        # initialize first date with initial_value
        paths[0] = self.initial_value
        if self.correlated is False:
            # if not correlated, generate random numbers
            rand = sn_random_numbers(
                (1, M, num_paths), random_seed=random_seed
            )  # shape (M,num_paths)
        else:
            # if correlated, use random number object as provided
            # in market environment
            rand = self.random_numbers
        short_rate = self.discount_curve.short_rate
        # get short rate for drift of process
        for t in range(1, len(self.time_grid)):
            # select the right time slice from the relevant
            # random number set
            if not self.correlated:
                ran = rand[t]  # shape (num_paths,)
            else:
                ran = np.dot(self.cholesky_matrix, rand[:, t, :])
                ran = ran[self.rn_set]

            # difference between two dates as year fraction
            delta_t = calculate_year_fraction(
                self.time_grid[t - 1], self.time_grid[t], day_count_convention=day_count_convention
            )

            paths[t] = paths[t - 1] * np.exp(
                (short_rate - 0.5 * self.volatility**2) * delta_t
                + self.volatility * np.sqrt(delta_t) * ran
            )
            # generate simulated values for the respective date
        self.instrument_values = paths


class SquareRootDiffusion(PathSimulation):
    """Class to generate simulated paths based on
    the Cox-Ingersoll-Ross (1985) square-root diffusion model.

    Attributes
    ==========
    name : string
        name of the object
    mar_env : MarketEnvironment
        market environment data for simulation
    corr : Boolean
        True if correlated with other model object

    Methods
    =======
    update :
        updates parameters
    generate_paths :
        returns Monte Carlo paths given the market environment
    """

    def __init__(self, name, mar_env, corr=False):
        super().__init__(name, mar_env, corr)
        # additional parameters needed
        self.kappa = mar_env.get_constant("kappa")
        self.theta = mar_env.get_constant("theta")

    def update(
        self, initial_value=None, volatility=None, kappa=None, theta=None, final_date=None
    ) -> None:
        "Update parameters"
        if initial_value is not None:
            self.initial_value = initial_value
        if volatility is not None:
            self.volatility = volatility
        if kappa is not None:
            self.kappa = kappa
        if theta is not None:
            self.theta = theta
        if final_date is not None:
            self.final_date = final_date
        self.instrument_values = None

    def generate_paths(
        self, random_seed: Optional[int] = None, day_count_convention: Union[int, float] = 365
    ) -> None:
        """Generate Cox-Ingersoll-Ross (square-root diffusion) paths.

        Implements the CIR model for mean-reverting interest rates:
        dr_t = kappa * (theta - r_t) * dt + sigma * sqrt(r_t) * dW_t

        Parameters
        ==========
        random_seed: int, optional
            random seed for reproducibility
        day_count_convention: int or float, default 365
            day count convention for time calculations

        Notes
        =====
        Uses full truncation Euler discretization to ensure paths
        remain non-negative (consistent with interest rate interpretation).
        """
        if self.time_grid is None:
            self.generate_time_grid()
        M = len(self.time_grid)
        num_paths = self.paths
        paths = np.zeros((M, num_paths))
        paths_ = np.zeros_like(paths)
        paths[0] = self.initial_value
        paths_[0] = self.initial_value

        if self.correlated is False:
            rand = sn_random_numbers((1, M, num_paths), random_seed=random_seed)
        else:
            rand = self.random_numbers

        for t in range(1, len(self.time_grid)):
            delta_t = calculate_year_fraction(
                self.time_grid[t - 1], self.time_grid[t], day_count_convention=day_count_convention
            )
            if self.correlated is False:
                ran = rand[t]
            else:
                ran = np.dot(self.cholesky_matrix, rand[:, t, :])
                ran = ran[self.rn_set]

            # full truncation Euler discretization
            mean_reversion = self.kappa * (self.theta - paths[t - 1, :]) * delta_t
            diffusion = np.sqrt(paths[t - 1, :]) * self.volatility * np.sqrt(delta_t) * ran
            paths_[t] = paths_[t - 1] + mean_reversion + diffusion
            paths[t] = np.maximum(0, paths_[t])

        self.instrument_values = paths


class JumpDiffusion(PathSimulation):
    """Class to generate simulated paths based on
    the Merton (1976) jump diffusion model.

    Attributes
    ==========
    name: str
        name of the object
    mar_env: MarketEnvironment
        market environment data for simulation
    corr: bool
        True if correlated with other model object

    Methods
    =======
    update:
        updates parameters
    generate_paths:
        returns Monte Carlo paths given the market environment
    """

    def __init__(self, name, mar_env, corr=False):
        super().__init__(name, mar_env, corr)
        # additional parameters needed
        self.lamb = mar_env.get_constant("lambda")
        self.mu = mar_env.get_constant("mu")
        self.delt = mar_env.get_constant("delta")

    def update(
        self, initial_value=None, volatility=None, lamb=None, mu=None, delta=None, final_date=None
    ):
        if initial_value is not None:
            self.initial_value = initial_value
        if volatility is not None:
            self.volatility = volatility
        if lamb is not None:
            self.lamb = lamb
        if mu is not None:
            self.mu = mu
        if delta is not None:
            self.delt = delta
        if final_date is not None:
            self.final_date = final_date
        self.instrument_values = None

    def generate_paths(
        self, random_seed: Optional[int] = None, day_count_convention: Union[int, float] = 365
    ) -> None:
        """Generate Merton jump diffusion paths.

        Implements the Merton (1976) model that combines geometric
        Brownian motion with random jump events:
        dS_t / S_t = (mu - lambda * E[J - 1]) * dt + sigma * dW_t + (J - 1) * dN_t

        Parameters
        ==========
        random_seed: int, optional
            random seed for reproducibility
        day_count_convention: int or float, default 365
            day count convention for time calculations

        Notes
        =====
        Requires parameters lambda (jump intensity), mu (jump mean),
        and delta (jump volatility) to be set in market environment.
        """
        if self.time_grid is None:
            # method from generic simulation class
            self.generate_time_grid()
        # number of dates for time grid
        M = len(self.time_grid)
        # number of paths
        num_paths = self.paths
        # ndarray initialization for path simulation
        paths = np.zeros((M, num_paths))
        # initialize first date with initial_value
        paths[0] = self.initial_value
        if self.correlated is False:
            # if not correlated, generate random numbers
            sn1 = sn_random_numbers((1, M, num_paths), random_seed=random_seed)
        else:
            # if correlated, use random number object as provided
            # in market environment
            sn1 = self.random_numbers

        # standard normally distributed pseudo-random numbers
        # for the jump component
        sn2 = sn_random_numbers((1, M, num_paths), random_seed=random_seed)

        rj = self.lamb * (np.exp(self.mu + 0.5 * self.delt**2) - 1)

        short_rate = self.discount_curve.short_rate

        for t in range(1, len(self.time_grid)):
            # select the right time slice from the relevant
            # random number set
            if self.correlated is False:
                ran = sn1[t]
            else:
                # only with correlation in portfolio context
                ran = np.dot(self.cholesky_matrix, sn1[:, t, :])
                ran = ran[self.rn_set]

            # difference between two dates as year fraction
            delta_t = calculate_year_fraction(
                self.time_grid[t - 1], self.time_grid[t], day_count_convention=day_count_convention
            )
            # Poisson-distributed pseudorandom numbers for jump component
            poi = np.random.poisson(self.lamb * delta_t, num_paths)
            drift = (short_rate - rj - 0.5 * self.volatility**2) * delta_t
            diffusion = self.volatility * np.sqrt(delta_t) * ran
            diffusion_factor = np.exp(drift + diffusion)

            jump_size = self.mu + self.delt * sn2[t]
            jump_factor = (np.exp(jump_size) - 1.0) * poi
            paths[t] = paths[t - 1] * (diffusion_factor + jump_factor)

        self.instrument_values = paths
