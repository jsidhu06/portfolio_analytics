"""Implicit finite difference solver for the Black–Scholes–Merton PDE.

This module is for teaching purposes and mirrors the structure of pde.py but uses
fully implicit time stepping (backward Euler). It supports vanilla European and
American options (with PSOR for early exercise).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ..enums import OptionType
from ..utils import calculate_year_fraction
from ..enums import DayCountConvention
from .params import PDEParams, BinomialParams

if TYPE_CHECKING:
    from .core import OptionValuation


def _solve_tridiagonal_thomas(
    lower: np.ndarray,
    diag: np.ndarray,
    upper: np.ndarray,
    rhs: np.ndarray,
) -> np.ndarray:
    """Solve a tridiagonal system Ax = rhs via the Thomas algorithm."""
    n = diag.size
    if rhs.size != n:
        raise ValueError("rhs length must match diag length")
    if lower.size != n - 1 or upper.size != n - 1:
        raise ValueError("lower/upper must have length n-1")

    c = upper.astype(float, copy=True)
    d = diag.astype(float, copy=True)
    b = lower.astype(float, copy=True)
    y = rhs.astype(float, copy=True)

    for i in range(1, n):
        w = b[i - 1] / d[i - 1]
        d[i] -= w * c[i - 1]
        y[i] -= w * y[i - 1]

    x = np.empty(n, dtype=float)
    x[-1] = y[-1] / d[-1]
    for i in range(n - 2, -1, -1):
        x[i] = (y[i] - c[i] * x[i + 1]) / d[i]
    return x


def _boundary_values(
    *,
    option_type: OptionType,
    strike: float,
    smax: float,
    tau: float,
    r: float,
    q: float,
    early_exercise: bool,
) -> tuple[float, float]:
    if option_type is OptionType.PUT:
        left = strike if early_exercise else strike * np.exp(-r * tau)
        right = 0.0
    else:
        left = 0.0
        continuation = smax * np.exp(-q * tau) - strike * np.exp(-r * tau)
        intrinsic = smax - strike
        right = max(continuation, intrinsic) if early_exercise else max(continuation, 0.0)
    return float(left), float(right)


def _vanilla_fd_implicit_core(
    *,
    spot: float,
    strike: float,
    time_to_maturity: float,
    risk_free_rate: float,
    volatility: float,
    dividend_yield: float,
    option_type: OptionType,
    smax_mult: float,
    spot_steps: int,
    time_steps: int,
    early_exercise: bool,
) -> tuple[float, np.ndarray, np.ndarray]:
    if option_type not in (OptionType.CALL, OptionType.PUT):
        raise NotImplementedError("Implicit FD supports only vanilla CALL/PUT.")
    if time_to_maturity <= 0:
        raise ValueError("time_to_maturity must be positive")
    if spot_steps < 3:
        raise ValueError("spot_steps must be >= 3")
    if time_steps < 1:
        raise ValueError("time_steps must be >= 1")
    if volatility <= 0:
        raise ValueError("volatility must be positive")

    smax = float(smax_mult * max(spot, strike))
    dS = smax / spot_steps
    S = np.linspace(0.0, smax, spot_steps + 1)
    j = np.arange(1, spot_steps)
    Sj = S[j]

    if option_type is OptionType.PUT:
        payoff = np.maximum(strike - S, 0.0)
    else:
        payoff = np.maximum(S - strike, 0.0)

    V = payoff.copy()  # V at tau=0 (maturity)
    intrinsic = payoff if early_exercise else None

    dt = time_to_maturity / time_steps

    a = (
        0.5
        * dt
        * ((risk_free_rate - dividend_yield) * Sj / dS - (volatility**2) * (Sj**2) / (dS**2))
    )
    b = 1.0 + dt * ((volatility**2) * (Sj**2) / (dS**2) + risk_free_rate)
    c = (
        -0.5
        * dt
        * ((risk_free_rate - dividend_yield) * Sj / dS + (volatility**2) * (Sj**2) / (dS**2))
    )

    for n in range(1, time_steps + 1):
        tau = float(n * dt)

        left, right = _boundary_values(
            option_type=option_type,
            strike=strike,
            smax=smax,
            tau=tau,
            r=risk_free_rate,
            q=dividend_yield,
            early_exercise=early_exercise,
        )
        V[0] = left
        V[-1] = right

        rhs = V[j].copy()
        rhs[0] -= a[0] * V[0]
        rhs[-1] -= c[-1] * V[-1]

        x = _solve_tridiagonal_thomas(a[1:], b, c[:-1], rhs)
        if early_exercise:
            V[j] = np.maximum(x, intrinsic[j])
        else:
            V[j] = x

    price = np.interp(spot, S, V)
    return price, S, V


def _european_vanilla_fd_implicit(
    *,
    spot: float,
    strike: float,
    time_to_maturity: float,
    risk_free_rate: float,
    volatility: float,
    dividend_yield: float,
    option_type: OptionType,
    smax_mult: float,
    spot_steps: int,
    time_steps: int,
) -> tuple[float, np.ndarray, np.ndarray]:
    """European vanilla option price via implicit FD (backward Euler)."""
    return _vanilla_fd_implicit_core(
        spot=spot,
        strike=strike,
        time_to_maturity=time_to_maturity,
        risk_free_rate=risk_free_rate,
        volatility=volatility,
        dividend_yield=dividend_yield,
        option_type=option_type,
        smax_mult=smax_mult,
        spot_steps=spot_steps,
        time_steps=time_steps,
        early_exercise=False,
    )


def _american_vanilla_fd_implicit_simple(
    *,
    spot: float,
    strike: float,
    time_to_maturity: float,
    risk_free_rate: float,
    volatility: float,
    dividend_yield: float,
    option_type: OptionType,
    smax_mult: float,
    spot_steps: int,
    time_steps: int,
    omega: float,
    tol: float,
    max_iter: int,
) -> tuple[float, np.ndarray, np.ndarray]:
    """American vanilla option price via implicit FD with immediate exercise projection."""
    return _vanilla_fd_implicit_core(
        spot=spot,
        strike=strike,
        time_to_maturity=time_to_maturity,
        risk_free_rate=risk_free_rate,
        volatility=volatility,
        dividend_yield=dividend_yield,
        option_type=option_type,
        smax_mult=smax_mult,
        spot_steps=spot_steps,
        time_steps=time_steps,
        early_exercise=True,
    )


class _FDEuropeanImplicitValuation:
    """European option valuation using implicit FD (teaching helper)."""

    def __init__(self, parent: "OptionValuation"):
        self.parent = parent

    def solve(self, params: PDEParams) -> tuple[float, np.ndarray, np.ndarray]:
        pv, S, V = self._solve(params)
        return pv, S, V

    def _solve(self, params: PDEParams) -> tuple[float, np.ndarray, np.ndarray]:
        spot = float(self.parent.underlying.initial_value)
        strike = self.parent.strike
        if strike is None:
            raise ValueError("strike is required for PDE FD valuation")

        volatility = float(self.parent.underlying.volatility)
        risk_free_rate = float(self.parent.discount_curve.short_rate)
        dividend_yield = float(self.parent.underlying.dividend_yield)

        time_to_maturity = calculate_year_fraction(
            self.parent.pricing_date,
            self.parent.maturity,
            day_count_convention=DayCountConvention.ACT_365F,
        )

        return _european_vanilla_fd_implicit(
            spot=spot,
            strike=float(strike),
            time_to_maturity=float(time_to_maturity),
            risk_free_rate=risk_free_rate,
            volatility=volatility,
            dividend_yield=dividend_yield,
            option_type=self.parent.option_type,
            smax_mult=float(params.smax_mult),
            spot_steps=int(params.spot_steps),
            time_steps=int(params.time_steps),
        )

    def present_value(self, params: PDEParams) -> float:
        pv, *_ = self._solve(params)
        return float(pv)


class _FDAmericanImplicitValuation:
    """American option valuation using implicit FD with early-exercise projection."""

    def __init__(self, parent: "OptionValuation"):
        self.parent = parent

    def solve(self, params: PDEParams) -> tuple[float, np.ndarray, np.ndarray]:
        pv, S, V = self._solve(params)
        return pv, S, V

    def _solve(self, params: PDEParams) -> tuple[float, np.ndarray, np.ndarray]:
        spot = float(self.parent.underlying.initial_value)
        strike = self.parent.strike
        if strike is None:
            raise ValueError("strike is required for PDE FD valuation")

        volatility = float(self.parent.underlying.volatility)
        risk_free_rate = float(self.parent.discount_curve.short_rate)
        dividend_yield = float(self.parent.underlying.dividend_yield)

        time_to_maturity = calculate_year_fraction(
            self.parent.pricing_date,
            self.parent.maturity,
            day_count_convention=DayCountConvention.ACT_365F,
        )

        return _american_vanilla_fd_implicit_simple(
            spot=spot,
            strike=float(strike),
            time_to_maturity=float(time_to_maturity),
            risk_free_rate=risk_free_rate,
            volatility=volatility,
            dividend_yield=dividend_yield,
            option_type=self.parent.option_type,
            smax_mult=float(params.smax_mult),
            spot_steps=int(params.spot_steps),
            time_steps=int(params.time_steps),
            omega=float(params.omega),
            tol=float(params.tol),
            max_iter=int(params.max_iter),
        )

    def present_value(self, params: PDEParams) -> float:
        pv, *_ = self._solve(params)
        return float(pv)


if __name__ == "__main__":
    import datetime as dt

    from ..market_environment import MarketData
    from ..rates import ConstantShortRate
    from .core import UnderlyingPricingData
    from types import SimpleNamespace

    pricing_date = dt.datetime(2025, 1, 1)
    maturity = pricing_date + dt.timedelta(days=365 * 5 / 12)  # 5 months later

    spot = 50.0
    strike = 50.0
    r = 0.10
    vol = 0.40
    dividend_yield = 0.0

    # Grid parameters: M=20 (spot steps), N=10 (time steps), dS=5 => Smax=100
    smax = 20 * 5.0
    smax_mult = smax / max(spot, strike)

    market_data = MarketData(
        pricing_date=pricing_date, discount_curve=ConstantShortRate("r", r), currency="USD"
    )

    underlying = UnderlyingPricingData(
        initial_value=spot,
        volatility=vol,
        market_data=market_data,
        dividend_yield=dividend_yield,
    )

    params = PDEParams(smax_mult=smax_mult, spot_steps=20, time_steps=10)

    parent = SimpleNamespace(
        underlying=underlying,
        strike=strike,
        pricing_date=pricing_date,
        maturity=maturity,
        discount_curve=underlying.discount_curve,
        option_type=OptionType.PUT,
    )

    price = _FDEuropeanImplicitValuation(parent).present_value(params)
    print(f"European put (implicit FD) price: {price:.6f}")

    american_price = _FDAmericanImplicitValuation(parent).present_value(params)
    print(f"American put (implicit FD) price: {american_price:.6f}")

    # parent = OptionValuation('eur_put', underlying, contract_spec)

    from .pde import _FDEuropeanValuation, _FDAmericanValuation

    params = PDEParams()
    european_price_cn = _FDEuropeanValuation(parent).present_value(params)
    print(f"European put (Crank-Nicolson FD) price: {european_price_cn:.6f}")

    american_price_cn = _FDAmericanValuation(parent).present_value(params)
    print(f"American put (Crank-Nicolson FD) price: {american_price_cn:.6f}")

    from .binomial import _BinomialEuropeanValuation, _BinomialAmericanValuation

    params = BinomialParams()
    binom_eur_price = _BinomialEuropeanValuation(parent).present_value(params)
    print(f"European put (binomial) price: {binom_eur_price:.6f}")

    binom_amer_price = _BinomialAmericanValuation(parent).present_value(params)
    print(f"American put (binomial) price: {binom_amer_price:.6f}")
