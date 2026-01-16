"""Helper functions and classes for derivatives valuation."""

from datetime import datetime
import numpy as np

SECONDS_IN_DAY = 86400


def calculate_year_fraction(start_date, end_date, day_count_convention: int | float = 365) -> float:
    """Calculate year fraction between two dates.

    This is a fundamental calculation in finance for time-value of money,
    discount factors, and accrued interest calculations.

    Parameters
    ==========
    start_date: datetime
        starting date
    end_date: datetime
        ending date
    day_count_convention: int or float, default 365
        number of days per year (day count convention):
        - 365: Actual/365 Fixed
        - 360: 30/360 (US)
        - 365.25: Actual/Actual (approximate)

    Returns
    =======
    year_fraction: float
        year fraction between start_date and end_date

    Examples
    ========
    >>> from datetime import datetime
    >>> start = datetime(2025, 1, 1)
    >>> end = datetime(2025, 1, 2)
    >>> calculate_year_fraction(start, end)  # doctest: +SKIP
    0.00273972...
    """
    delta_days = (end_date - start_date).total_seconds() / SECONDS_IN_DAY
    year_fraction = delta_days / day_count_convention
    return year_fraction


def get_year_deltas(
    date_list: list[datetime], day_count_convention: int | float = 365
) -> np.ndarray:
    """Return vector of floats with day deltas in year fractions.

    Initial value is normalized to zero. Useful for discount factor
    calculations and time grid generation.

    Parameters
    ==========
    date_list: list or array-like
        collection of datetime objects
    day_count_convention: int or float, default 365
        number of days per year (day count convention)

    Returns
    =======
    delta_list: np.ndarray
        array of year fractions, first element is always 0
    """
    start = min(date_list)
    delta_list = [calculate_year_fraction(start, date, day_count_convention) for date in date_list]
    return np.array(delta_list)


def sn_random_numbers(
    shape: tuple[int, int, int],
    antithetic: bool = True,
    moment_matching: bool = True,
    random_seed: int | None = None,
) -> np.ndarray:
    """Return array of standard normally distributed pseudo-random numbers.

    Supports antithetic variates and moment matching for variance reduction
    in Monte Carlo simulations.

    Parameters
    ==========
    shape: tuple (o, n, m)
        array shape (# simulations, # time steps, # paths)
    antithetic: bool, default True
        if True, use antithetic variates for variance reduction
        (generates n/2 randoms and appends their negatives)
    moment_matching: bool, default True
        if True, rescale to match sample mean=0 and std=1
    random_seed: int, optional
        random number generator seed for reproducibility

    Returns
    =======
    ran: np.ndarray
        shape (o, n, m) if o > 1, else shape (n, m)
        standard normally distributed random numbers
    """
    rng = np.random.default_rng(random_seed)
    if antithetic:
        ran = rng.standard_normal((shape[0], shape[1], shape[2] // 2))
        ran = np.concatenate((ran, -ran), axis=2)
    else:
        ran = rng.standard_normal(shape)
    if moment_matching:
        ran = (ran - np.mean(ran)) / np.std(ran)  # note this is population std dev
    if shape[0] == 1:
        return ran[0]
    return ran
