"""Helper functions and classes for derivatives valuation."""

from datetime import datetime
from math import comb
from typing import Callable, Literal
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


def pv_discrete_dividends(
    dividends: list[tuple[datetime, float]],
    pricing_date: datetime,
    maturity: datetime,
    short_rate: float,
    day_count_convention: int | float = 365,
) -> float:
    """Present value of discrete cash dividends between pricing_date and maturity.

    Only dividends with pricing_date < ex_date <= maturity are included.
    """
    if not dividends:
        return 0.0

    pv = 0.0
    for ex_date, amount in dividends:
        if pricing_date < ex_date <= maturity:
            t = calculate_year_fraction(pricing_date, ex_date, day_count_convention)
            pv += float(amount) * np.exp(-short_rate * t)
    return float(pv)


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


def binomial_pmf(k: np.ndarray | int, n: int, p: float) -> np.ndarray:
    """Binomial(n, p) probability mass function.

    Parameters
    ==========
    k:
        Success count(s). Can be an int or a numpy array of ints.
    n:
        Number of trials (>= 0).
    p:
        Success probability in [0, 1].
    """
    if n < 0:
        raise ValueError("n must be >= 0")
    if not (0.0 <= float(p) <= 1.0):
        raise ValueError("p must be in [0, 1]")

    k_arr = np.asarray(k, dtype=int)
    if np.any((k_arr < 0) | (k_arr > n)):
        # Out-of-support values have pmf 0
        out = np.zeros_like(k_arr, dtype=float)
        in_support = (k_arr >= 0) & (k_arr <= n)
        if np.any(in_support):
            ks = k_arr[in_support]
            out[in_support] = np.array([comb(n, int(kk)) for kk in ks], dtype=float) * (
                (p**ks) * ((1.0 - p) ** (n - ks))
            )
        return out

    return np.array([comb(n, int(kk)) for kk in k_arr], dtype=float) * (
        (p**k_arr) * ((1.0 - p) ** (n - k_arr))
    )


def expected_binomial(
    n: int,
    p: float,
    f: Callable[[np.ndarray], np.ndarray],
) -> float:
    """Compute $\mathbb{E}[f(K)]$ where $K \sim \text{Binomial}(n, p)$.

    This is a small convenience wrapper around the explicit sum
    $\sum_{k=0}^n \binom{n}{k} p^k (1-p)^{n-k} f(k)$.
    """
    ks = np.arange(n + 1)
    pmf = binomial_pmf(ks, n=n, p=p)
    vals = np.asarray(f(ks), dtype=float)
    if vals.shape != ks.shape:
        raise ValueError("f(k) must return an array with same shape as k")
    return float(np.dot(pmf, vals))


def expected_binomial_payoff(
    *,
    S0: float,
    n: int,
    T: float,
    side: Literal["call", "put"],
    K: float,
    r: float,
    q: float,
    u: float | None = None,
) -> float:
    """Expected vanilla payoff under a binomial terminal distribution.

    Notes
    -----
    In binomial-tree style models, the terminal spot is
    $S_T(k) = S_0 u^k d^{n-k}$

    Parameters
    ==========
    S0:
        Initial spot.
    n, p:
        Binomial distribution parameters.
    side:
        "call" or "put".
    K:
        Strike.
    u, d:
        Optional up/down multipliers used to map k -> terminal spot. If omitted,
        the function returns the (k-invariant) payoff at S0.
    """
    side_l = str(side).lower()
    if side_l not in ("call", "put"):
        raise ValueError("side must be 'call' or 'put'")

    S0 = float(S0)
    K = float(K)

    u = float(u)
    d = float(1 / u)

    def payoff_from_k(ks: np.ndarray) -> np.ndarray:
        ST = S0 * (u**ks) * (d ** (n - ks))
        if side_l == "call":
            return np.maximum(ST - K, 0.0)
        return np.maximum(K - ST, 0.0)

    delta_t = T / n
    p = (np.exp((r - q) * delta_t) - d) / (u - d)

    return expected_binomial(n=n, p=p, f=payoff_from_k)
