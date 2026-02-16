"""Helper functions and classes for derivatives valuation."""

from contextlib import contextmanager
from datetime import datetime
from math import comb
from typing import Callable, Protocol
from collections.abc import Iterator
import time
import numpy as np

from .enums import DayCountConvention, OptionType

__all__ = [
    "ForwardCurve",
    "log_timing",
    "calculate_year_fraction",
    "get_year_deltas",
    "pv_discrete_dividends",
    "forward_price",
    "put_call_parity_rhs",
    "put_call_parity_gap",
    "sn_random_numbers",
    "binomial_pmf",
    "expected_binomial",
    "expected_binomial_payoff",
]

SECONDS_IN_DAY = 86400


class ForwardCurve(Protocol):
    """Minimal interface for a deterministic discount/forward curve."""

    def df(self, t: float | np.ndarray) -> np.ndarray: ...

    def forward_rate(self, t0: float, t1: float) -> float: ...


@contextmanager
def log_timing(logger, label: str, enabled: bool) -> Iterator[None]:
    """Log timing for a code block when enabled is True."""
    if not enabled:
        yield
        return
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        logger.debug("Timing %s: %.6fs", label, elapsed)


def _day_count_30_360_us(start_date: datetime, end_date: datetime) -> float:
    """30/360 (US) day-count fraction between two dates."""
    y1, m1, d1 = start_date.year, start_date.month, start_date.day
    y2, m2, d2 = end_date.year, end_date.month, end_date.day

    if d1 == 31:
        d1 = 30
    if d2 == 31 and d1 in (30, 31):
        d2 = 30

    return (360 * (y2 - y1) + 30 * (m2 - m1) + (d2 - d1)) / 360.0


def calculate_year_fraction(
    start_date,
    end_date,
    day_count_convention: DayCountConvention = DayCountConvention.ACT_365F,
) -> float:
    """Calculate year fraction between two dates.

    Parameters
    ==========
    start_date: datetime
        starting date
    end_date: datetime
        ending date
    day_count_convention: DayCountConvention, default DayCountConvention.ACT_365F
        Day-count basis. Supported:
        - DayCountConvention.ACT_365F
        - DayCountConvention.ACT_360
        - DayCountConvention.ACT_365_25
        - DayCountConvention.THIRTY_360_US

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
    if day_count_convention is DayCountConvention.THIRTY_360_US:
        return _day_count_30_360_us(start_date, end_date)
    if day_count_convention is DayCountConvention.ACT_360:
        denom = 360.0
    elif day_count_convention is DayCountConvention.ACT_365_25:
        denom = 365.25
    elif day_count_convention is DayCountConvention.ACT_365F:
        denom = 365.0
    else:
        raise ValueError(f"Unsupported day_count_convention: {day_count_convention}")

    delta_days = (end_date - start_date).total_seconds() / SECONDS_IN_DAY
    year_fraction = delta_days / denom
    return year_fraction


def get_year_deltas(
    date_list: list[datetime],
    day_count_convention: DayCountConvention = DayCountConvention.ACT_365F,
) -> np.ndarray:
    """Return vector of floats with day deltas in year fractions.

    Initial value is normalized to zero. Useful for discount factor
    calculations and time grid generation.

    Parameters
    ==========
    date_list: list or array-like
        collection of datetime objects
    day_count_convention: DayCountConvention, default DayCountConvention.ACT_365F
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
    curve_date: datetime,
    end_date: datetime,
    discount_curve: ForwardCurve,
    start_date: datetime | None = None,
    day_count_convention: DayCountConvention = DayCountConvention.ACT_365F,
) -> float:
    """Present value of discrete cash dividends between start_date and end_date.

    Only dividends with start_date < ex_date <= end_date are included.
    """
    if not dividends:
        return 0.0

    start_date = curve_date if start_date is None else start_date
    t_start = calculate_year_fraction(curve_date, start_date, day_count_convention)
    df_start = float(discount_curve.df(t_start))

    pv = 0.0
    for ex_date, amount in dividends:
        if start_date < ex_date <= end_date:
            t_ex = calculate_year_fraction(curve_date, ex_date, day_count_convention)
            df_ex = float(discount_curve.df(t_ex))
            df = df_ex / df_start
            pv += float(amount) * df
    return float(pv)


def forward_price(
    *,
    spot: float,
    pricing_date: datetime,
    maturity: datetime,
    short_rate: float,
    discount_curve: ForwardCurve | None = None,
    dividend_curve: ForwardCurve | None = None,
    discrete_dividends: list[tuple[datetime, float]] | None = None,
    day_count_convention: DayCountConvention = DayCountConvention.ACT_365F,
) -> float:
    """Compute the no-arbitrage forward price for an equity-style underlying.

    Supports either continuous dividend curve or discrete dividends (not both).
    """
    if dividend_curve is not None and discrete_dividends:
        raise ValueError(
            "Provide either a continuous dividend_curve or discrete_dividends, not both."
        )

    t = calculate_year_fraction(pricing_date, maturity, day_count_convention)
    if t <= 0:
        raise ValueError("maturity must be after pricing_date")

    spot = float(spot)
    short_rate = float(short_rate)

    if discrete_dividends:
        if discount_curve is None:
            raise ValueError("discount_curve is required for discrete_dividends.")
        pv_divs = pv_discrete_dividends(
            discrete_dividends,
            curve_date=pricing_date,
            end_date=maturity,
            discount_curve=discount_curve,
            day_count_convention=day_count_convention,
        )
        prepaid_forward = spot - pv_divs
        return float(prepaid_forward * np.exp(short_rate * t))

    df_q = 1.0 if dividend_curve is None else float(dividend_curve.df(t))
    return float(spot * np.exp(short_rate * t) * df_q)


def put_call_parity_rhs(
    *,
    spot: float,
    strike: float,
    pricing_date: datetime,
    maturity: datetime,
    short_rate: float,
    discount_curve: ForwardCurve | None = None,
    dividend_curve: ForwardCurve | None = None,
    discrete_dividends: list[tuple[datetime, float]] | None = None,
    day_count_convention: DayCountConvention = DayCountConvention.ACT_365F,
) -> float:
    """Compute the RHS of put-call parity for European options.

    Returns C - P implied by no-arbitrage (i.e., PV(F - K)).
    """
    t = calculate_year_fraction(pricing_date, maturity, day_count_convention)
    if t <= 0:
        raise ValueError("maturity must be after pricing_date")

    fwd = forward_price(
        spot=spot,
        pricing_date=pricing_date,
        maturity=maturity,
        short_rate=short_rate,
        discount_curve=discount_curve,
        dividend_curve=dividend_curve,
        discrete_dividends=discrete_dividends,
        day_count_convention=day_count_convention,
    )
    return float(np.exp(-short_rate * t) * (fwd - float(strike)))


def put_call_parity_gap(
    *,
    call_price: float,
    put_price: float,
    spot: float,
    strike: float,
    pricing_date: datetime,
    maturity: datetime,
    short_rate: float,
    discount_curve: ForwardCurve | None = None,
    dividend_curve: ForwardCurve | None = None,
    discrete_dividends: list[tuple[datetime, float]] | None = None,
    day_count_convention: DayCountConvention = DayCountConvention.ACT_365F,
) -> float:
    """Return call-put parity residual: (C - P) - RHS."""
    rhs = put_call_parity_rhs(
        spot=spot,
        strike=strike,
        pricing_date=pricing_date,
        maturity=maturity,
        short_rate=short_rate,
        discount_curve=discount_curve,
        dividend_curve=dividend_curve,
        discrete_dividends=discrete_dividends,
        day_count_convention=day_count_convention,
    )
    return float(call_price - put_price - rhs)


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

    p = float(p)
    if not (0.0 <= p <= 1.0):
        raise ValueError("p must be in [0, 1]")

    k_arr = np.asarray(k, dtype=int)
    out = np.zeros_like(k_arr, dtype=float)

    in_support = (k_arr >= 0) & (k_arr <= n)
    if not np.any(in_support):
        return out

    ks = k_arr[in_support]
    combs = np.array([comb(n, int(kk)) for kk in ks], dtype=float)
    out[in_support] = combs * (p**ks) * ((1.0 - p) ** (n - ks))
    return out


def expected_binomial(
    n: int,
    p: float,
    f: Callable[[np.ndarray], np.ndarray],
) -> float:
    """Compute $\mathbb{E}[f(K)]$ where $K \sim \text{Binomial}(n, p)$.

    This is a small convenience wrapper around the explicit sum
    $\sum_{k=0}^n \binom{n}{k} p^k (1-p)^{n-k} f(k)$.
    """
    if n < 0:
        raise ValueError("n must be >= 0")
    p = float(p)
    if not (0.0 <= p <= 1.0):
        raise ValueError("p must be in [0, 1]")

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
    option_type: OptionType,
    K: float,
    r: float,
    q: float,
    u: float | None = None,
) -> float:
    """Expected vanilla payoff under a binomial terminal distribution.

    Parameters
    ==========
    S0:
        Initial spot.
    n:
        Number of steps.
    T:
        Time to maturity (years).
    option_type: OptionType
        Option type (Call or Put).
    K:
        Strike.
    u:
        Up multiplier for one step. Down multiplier is set to d = 1 / u.

    Notes
    -----
    In a CRR-style model, the terminal spot is $S_T(k) = S_0 u^k d^{n-k}$
    with risk-neutral probability
    $p = (e^{(r-q)\Delta t} - d) / (u - d)$.
    This function returns the expected terminal payoff (not discounted).
    """
    if option_type not in (OptionType.CALL, OptionType.PUT):
        raise ValueError("option_type must be OptionType.CALL or OptionType.PUT")

    if n < 1:
        raise ValueError("n must be >= 1")
    if T <= 0:
        raise ValueError("T must be positive")

    S0 = float(S0)
    K = float(K)
    r = float(r)
    q = float(q)

    if u is None:
        raise ValueError("u must be provided")
    u = float(u)
    if u <= 0:
        raise ValueError("u must be positive")
    d = 1.0 / u

    def payoff_from_k(ks: np.ndarray) -> np.ndarray:
        ST = S0 * (u**ks) * (d ** (n - ks))
        if option_type is OptionType.CALL:
            return np.maximum(ST - K, 0.0)
        return np.maximum(K - ST, 0.0)

    delta_t = T / n
    p = (np.exp((r - q) * delta_t) - d) / (u - d)
    if not (0.0 <= p <= 1.0):
        raise ValueError("risk-neutral probability p must be in [0, 1]")

    return expected_binomial(n=n, p=p, f=payoff_from_k)
