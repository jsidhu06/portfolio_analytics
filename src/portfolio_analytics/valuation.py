from abc import ABC, abstractmethod
from typing import Tuple, Optional
import numpy as np


class OptionValuation(ABC):
    """Basic class for single-factor option valuation.

    Attributes
    ==========
    name: str
        name of the object
    underlying: instance of simulation class
        object modeling the single risk factor
    mar_env: instance of market_environment
        market environment data for valuation
    side: str
        'call' or 'put' for the option type

    Methods
    =======
    update:
        updates selected valuation parameters
    delta:
        returns the Delta of the derivative
    vega:
        returns the Vega of the derivative
    """

    def __init__(self, name, underlying, mar_env, side: str):
        if side not in ("call", "put"):
            raise ValueError(f"side must be 'call' or 'put', received '{side}'")
        self.name = name
        self.side = side
        self.pricing_date = mar_env.pricing_date
        self.strike = mar_env.constants.get("strike")  # strike is optional
        self.maturity = mar_env.get_constant("maturity")
        self.currency = mar_env.get_constant("currency")

        # simulation parameters and discount curve from simulation object
        self.frequency = underlying.frequency
        self.paths = underlying.paths
        self.discount_curve = underlying.discount_curve
        self.underlying = underlying
        # provide pricing_date and maturity to underlying
        self.underlying.special_dates.extend([self.pricing_date, self.maturity])

    def update(self, initial_value=None, volatility=None, strike=None, maturity=None):
        if initial_value is not None:
            self.underlying.update(initial_value=initial_value)
        if volatility is not None:
            self.underlying.update(volatility=volatility)
        if strike is not None:
            self.strike = strike
        if maturity is not None:
            self.maturity = maturity
            # add new maturity date if not in time_grid
            if maturity not in self.underlying.time_grid:
                self.underlying.special_dates.append(maturity)
                self.underlying.instrument_values = None

    @abstractmethod
    def generate_payoff(self, random_seed: Optional[int] = None) -> None:
        """
        Parameters
        ==========
        random_seed: int, optional
            random seed for path generation
        """
        raise NotImplementedError("Method generate_payoff() not implemented")

    @abstractmethod
    def present_value(
        self, random_seed: Optional[int] = None, full: bool = False
    ) -> Tuple[float, np.ndarray] | float:
        """
        Parameters
        ==========
        random_seed: int, optional
            random seed for path generation
        full: bool, optional
            set to True to return full discount present value payoff vector

        Returns
        =======
        present_value: float
            present value of the derivative
        """
        raise NotImplementedError("Method present_value() not implemented")

    def delta(self, epsilon: Optional[float] = None, random_seed: Optional[int] = None):
        if epsilon is None:
            epsilon = self.underlying.initial_value / 100
        # central difference approximation
        initial_spot = self.underlying.initial_value
        try:
            # calculate left value for numerical Delta
            self.underlying.initial_value -= epsilon
            value_left = self.present_value(random_seed=random_seed)
            # numerical underlying value for right value
            self.underlying.initial_value += 2 * epsilon
            # calculate right value for numerical delta
            value_right = self.present_value(random_seed=random_seed)
        finally:
            # reset the initial_value of the simulation object
            self.underlying.initial_value = initial_spot

        delta = (value_right - value_left) / (2 * epsilon)
        # correct for potential numerical errors
        if delta < -1.0:
            return -1.0
        if delta > 1.0:
            return 1.0
        return delta

    def vega(self, epsilon: float = 0.01, random_seed: Optional[int] = None):
        if epsilon < self.underlying.volatility / 50.0:
            epsilon = self.underlying.volatility / 50.0
        # central-difference approximation
        initial_vol = self.underlying.volatility
        try:
            # calculate the left value for numerical Vega
            self.underlying.volatility -= epsilon
            value_left = self.present_value(random_seed=random_seed)
            # numerical volatility value for right value
            # update the simulation object
            self.underlying.volatility += 2 * epsilon
            # calculate the right value for numerical Vega
            value_right = self.present_value(random_seed=random_seed)
        finally:
            # reset volatility value of simulation object
            self.underlying.volatility = initial_vol
        vega = (value_right - value_left) / (2 * epsilon)
        return vega


# PayoffFunc = Callable[[Dict[str, np.ndarray | float | int]], np.ndarray]


class ValuationMCSEuropean(OptionValuation):
    """Monte Carlo Simulation European option valuation class.

    Attributes
    ==========
    name: str
        name of the object
    underlying: instance of simulation class
        object modeling the single risk factor
    mar_env: instance of market_environment
        market environment data for valuation

    Methods
    =======
    generate_payoff:
        generates the payoff at maturity
    present_value:
        returns the present value of the derivative
    """

    def generate_payoff(self, random_seed: Optional[int] = None) -> None:
        """
        Parameters
        ==========
        random_seed: int, optional
            random seed for path generation
        """

        paths = self.underlying.get_instrument_values(random_seed=random_seed)
        time_grid = self.underlying.time_grid
        time_index_start = int(np.where(time_grid == self.pricing_date)[0][0])
        time_index_end = int(np.where(time_grid == self.maturity)[0][0])
        instrument_values = paths[time_index_start : time_index_end + 1]
        maturity_value = paths[time_index_end]

        if self.side == "call":
            payoff = np.maximum(maturity_value - self.strike, 0)
        else:
            payoff = np.maximum(self.strike - maturity_value, 0)

        return instrument_values, payoff, time_index_start, time_index_end

    def present_value(
        self, random_seed: Optional[int] = None, full: bool = False
    ) -> Tuple[float, np.ndarray] | float:
        """
        Parameters
        ==========
        accuracy: int
            number of decimals in returned result
        random_seed: int, optional
            random seed for path generation
        full: bool
            return also full 1d array of present values
        """
        # for european optionss we only need the payoff
        cash_flow = self.generate_payoff(random_seed=random_seed)[1]
        discount_factor = self.discount_curve.get_discount_factors(
            (self.pricing_date, self.maturity)
        )[-1, 1]
        result = discount_factor * np.sum(cash_flow) / len(cash_flow)
        if full:
            return result, discount_factor * cash_flow
        else:
            return result


class ValuationMCSAmerican(OptionValuation):
    """Monte Carlo Simulation American option valuation class.

    Attributes
    ==========
    name: str
        name of the object
    underlying: instance of simulation class
        object modeling the single risk factor
    mar_env: instance of market_environment
        market environment data for valuation
    side: str
        'call' or 'put' for the option type

    Methods
    =======
    generate_payoff:
        generates the payoff at maturity
    present_value:
        returns the present value of the derivative
    """

    def generate_payoff(self, random_seed: Optional[int] = None) -> None:
        """
        Parameters
        ==========
        random_seed: int, optional
            random seed for path generation
        """
        paths = self.underlying.get_instrument_values(random_seed=random_seed)
        time_grid = self.underlying.time_grid
        time_index_start = int(np.where(time_grid == self.pricing_date)[0][0])
        time_index_end = int(np.where(time_grid == self.maturity)[0][0])
        instrument_values = paths[time_index_start : time_index_end + 1]

        if self.side == "call":
            payoff = np.maximum(instrument_values - self.strike, 0)
        else:
            payoff = np.maximum(self.strike - instrument_values, 0)

        return instrument_values, payoff, time_index_start, time_index_end

    def present_value(
        self, random_seed: Optional[int] = None, deg: int = 2, full: bool = False
    ) -> Tuple[float, np.ndarray] | float:
        """
        Parameters
        ==========
        random_seed: int, optional
            random seed for path generation
        deg: int
            degree of polynomial for regression
        full: bool
            return also full 1d array of present values
        """
        instrument_values, intrinsic_values, time_index_start, time_index_end = (
            self.generate_payoff(random_seed=random_seed)
        )
        time_list = self.underlying.time_grid[time_index_start : time_index_end + 1]
        discount_factors = self.discount_curve.get_discount_factors(time_list, dtobjects=True)
        V = np.zeros_like(intrinsic_values)
        V[-1] = intrinsic_values[-1]
        for t in range(len(time_list) - 2, 0, -1):
            df = discount_factors[t + 1, 1] / discount_factors[t, 1]
            itm = intrinsic_values[t] > 0
            # itm = [True for _ in range(intrinsic_values.shape[1])]  # consider all paths
            X = instrument_values[t][itm]
            Y = df * V[t + 1][itm]
            if len(X) > 0:
                coefficients = np.polyfit(X, Y, deg=deg)
            else:
                coefficients = np.zeros(deg + 1)
            predicted_cv = np.zeros_like(instrument_values[t])
            predicted_cv[itm] = np.polyval(coefficients, instrument_values[t][itm])
            V[t] = np.where(intrinsic_values[t] > predicted_cv, intrinsic_values[t], df * V[t + 1])
        df = discount_factors[1, 1] / discount_factors[0, 1]
        result = df * np.mean(V[1])
        if full:
            return result, df * np.mean(V[1])
        return result
