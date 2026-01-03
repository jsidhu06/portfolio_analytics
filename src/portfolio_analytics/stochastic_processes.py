from typing import Optional
from abc import ABC, abstractmethod
import datetime as dt
import numpy as np
import pandas as pd
from .utils import calculate_year_fraction, sn_random_numbers


class PathSimulation(ABC):
    """Providing base methods for simulation classes.

    Attributes
    ==========
    name: str
        name of the object
    mar_env: instance of market_environment
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

    def __init__(self, name: str, mar_env, corr: bool):
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

    def generate_time_grid(self):
        start = self.pricing_date
        end = self.final_date
        # pandas date_range function
        # freq = e.g. 'B' for Business Day,
        # 'W' for Weekly, 'M' for Monthly
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

    def get_instrument_values(self, random_seed: Optional[int] = None, day_count=365.0):
        if self.instrument_values is None:
            # only initiate simulation if there are no instrument values
            self.generate_paths(random_seed=random_seed, day_count=day_count)
        elif random_seed is not None:
            # also initiate resimulation when random_seed is not None
            self.generate_paths(random_seed=random_seed, day_count=day_count)
        return self.instrument_values

    @abstractmethod
    def generate_paths(self, random_seed: Optional[int] = None, day_count: float = 365.0) -> None:
        raise NotImplementedError("Subclasses must implement generate_paths method")


class GeometricBrownianMotion(PathSimulation):
    """Class to generate simulated paths based on
    the Black-Scholes-Merton geometric Brownian motion model.

    Attributes
    ==========
    name: string
        name of the object
    mar_env: instance of market_environment
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

    def __init__(self, name: str, mar_env, corr: bool = False):
        super().__init__(name, mar_env, corr)

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

    def generate_paths(self, random_seed: Optional[int] = None, day_count=365.0) -> None:
        "Generate geometric Brownian motion paths."
        if self.time_grid is None:
            # method from generic simulation class
            self.generate_time_grid()
        # number of dates (timesteps) for time grid
        M = len(self.time_grid)
        # number of paths
        I = self.paths  # noqa:E741
        # ndarray initialization for path simulation
        paths = np.zeros((M, I))
        # initialize first date with initial_value
        paths[0] = self.initial_value
        if not self.correlated:
            # if not correlated, generate random numbers
            rand = sn_random_numbers((1, M, I), random_seed=random_seed)  # shape (M,I)
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
                ran = rand[t]  # shape (I,)
            else:
                ran = np.dot(self.cholesky_matrix, rand[:, t, :])
                ran = ran[self.rn_set]

            delta_t = calculate_year_fraction(
                self.time_grid[t - 1], self.time_grid[t], day_count=day_count
            )
            # difference between two dates as year fraction
            paths[t] = paths[t - 1] * np.exp(
                (short_rate - 0.5 * self.volatility**2) * delta_t
                + self.volatility * np.sqrt(delta_t) * ran
            )
            # generate simulated values for the respective date
        self.instrument_values = paths
