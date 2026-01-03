# pylint: disable=missing-docstring, invalid-name, too-many-arguments
import numpy as np
import pandas as pd

SECONDS_IN_DAY = 86400.0


def calculate_year_fraction(start_date, end_date, day_count=365.0):
    """Calculate year fraction between two dates.

    Parameters
    ==========
    start_date: datetime object
        starting date
    end_date: datetime object
        ending date
    day_count: float
        number of days for a year
        (to account for different conventions)

    Results
    =======
    year_fraction: float
        year fraction between start_date and end_date
    """
    delta_days = (end_date - start_date).total_seconds() / SECONDS_IN_DAY
    year_fraction = delta_days / day_count
    return year_fraction


def get_year_deltas(date_list, day_count=365.0):
    """Return vector of floats with day deltas in year fractions.
    Initial value normalized to zero.

    Parameters
    ==========
    date_list: list or array
        collection of datetime objects
    day_count: float
        number of days for a year
        (to account for different conventions)

    Results
    =======
    delta_list: array
        year fractions
    """

    start = min(date_list)

    delta_list = [calculate_year_fraction(start, date, day_count) for date in date_list]
    return np.array(delta_list)


def sn_random_numbers(shape, antithetic=True, moment_matching=True, random_seed=None):
    """Returns an ndarray object of shape shape with (pseudo)random numbers
    that are standard normally distributed.

    Parameters
    ==========
    shape: tuple (o, n, m)
        generation of array with shape (o, n, m)
    antithetic: Boolean
        generation of antithetic variates
    moment_matching: Boolean
        matching of first and second moments
    random_seed: Boolean
        flag to fix the seed

    Results
    =======
    ran: (o, n, m) array of (pseudo)random numbers if o>1,
    else (n, m) array
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    if antithetic:
        ran = np.random.standard_normal((shape[0], shape[1], shape[2] // 2))
        ran = np.concatenate((ran, -ran), axis=2)
    else:
        ran = np.random.standard_normal(shape)
    if moment_matching:
        ran = (ran - np.mean(ran)) / np.std(ran)
    if shape[0] == 1:
        return ran[0]
    return ran


class ConstantShortRate:
    """Class for constant short rate discounting.

    Attributes
    ==========
    name: string
        name of the object
    short_rate: float (positive)
        constant rate for discounting

    Methods
    =======
    get_discount_factors:
        get discount factors given a list/array of datetime objects
        or year fractions
    """

    def __init__(self, name, short_rate):
        self.name = name
        self.short_rate = short_rate
        if short_rate < 0:
            raise ValueError("Short rate negative.")
            # this is debatable given recent market realities

    def _ensure_ascending(self, x):
        if np.any(np.diff(x) < 0):
            raise ValueError("date_list must be sorted ascending")

    def get_discount_factors(self, date_list, dtobjects=True):
        """Get discount factors for given date list."""
        self._ensure_ascending(date_list)

        if dtobjects is True:
            dlist = get_year_deltas(date_list)
        else:
            dlist = np.array(date_list)
        dflist = np.exp(-self.short_rate * dlist)
        return pd.DataFrame({"discount_factor": dflist}, index=date_list)


class MarketEnvironment:
    """Class to model a market environment relevant for valuation.

    Attributes
    ==========
    name: string
        name of the market environment
    pricing_date: datetime object
        date of the market environment

    Methods
    =======
    add_constant:
        adds a constant (e.g. model parameter)
    get_constant:
        gets a constant
    add_list:
        adds a list (e.g. underlyings)
    get_list:
        gets a list
    add_curve:
        adds a market curve (e.g. yield curve)
    get_curve:
        gets a market curve
    add_environment:
        adds and overwrites whole market environments
        with constants, lists, and curves
    """

    def __init__(self, name, pricing_date):
        self.name = name
        self.pricing_date = pricing_date
        self.constants = {}
        self.lists = {}
        self.curves = {}

    def add_constant(self, key, constant):
        self.constants[key] = constant

    def get_constant(self, key):
        return self.constants[key]

    def add_list(self, key, list_object):
        self.lists[key] = list_object

    def get_list(self, key):
        return self.lists[key]

    def add_curve(self, key, curve):
        self.curves[key] = curve

    def get_curve(self, key):
        return self.curves[key]

    def add_environment(self, env):
        # overwrites existing values, if they exist
        self.constants.update(env.constants)
        self.lists.update(env.lists)
        self.curves.update(env.curves)
