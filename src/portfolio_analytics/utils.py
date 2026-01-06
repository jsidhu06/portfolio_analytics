"Helper functions and classes for deriatives valuation" ""
import numpy as np

SECONDS_IN_DAY = 86400


def calculate_year_fraction(start_date, end_date, day_count=365):
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


def get_year_deltas(date_list, day_count=365):
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
    random_seed: int, optional
        random number generator seed

    Results
    =======
    ran: (o, n, m) array of (pseudo)random numbers if o>1, else (n, m) array
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    if antithetic:
        ran = np.random.standard_normal((shape[0], shape[1], shape[2] // 2))
        ran = np.concatenate((ran, -ran), axis=2)
    else:
        ran = np.random.standard_normal(shape)
    if moment_matching:
        ran = (ran - np.mean(ran)) / np.std(ran)  # note this is population std dev
    if shape[0] == 1:
        return ran[0]
    return ran
