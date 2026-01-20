"""Finite difference (PDE) valuation implementations.

This module follows the same structure as other valuation modules:
- an internal implementation class that plugs into OptionValuation
- thin convenience wrappers for direct function-style pricing

Current scope
-------------
PDE via finite differences (Crank–Nicolson) for vanilla EUROPEAN call/put.
PDE via finite differences (Crank–Nicolson + PSOR) for vanilla AMERICAN call/put.
"""

from typing import TYPE_CHECKING

import numpy as np

from .enums import OptionType
from .utils import calculate_year_fraction

if TYPE_CHECKING:
    from .valuation import OptionValuation


def _solve_tridiagonal_thomas(
    lower: np.ndarray,
    diag: np.ndarray,
    upper: np.ndarray,
    rhs: np.ndarray,
) -> np.ndarray:
    """Solve a tridiagonal system Ax = rhs via the Thomas algorithm.

    A has:
      - lower: subdiagonal (length n-1)  -> A[i, i-1]
      - diag:  main diagonal (length n)  -> A[i, i]
      - upper: superdiagonal (length n-1)-> A[i, i+1]
    """
    n = diag.size
    if rhs.size != n:
        raise ValueError("rhs length must match diag length")
    if lower.size != n - 1 or upper.size != n - 1:
        raise ValueError("lower/upper must have length n-1")

    # Copy to avoid mutating inputs
    c = upper.astype(float, copy=True)
    d = diag.astype(float, copy=True)
    b = lower.astype(float, copy=True)
    y = rhs.astype(float, copy=True)

    # Forward elimination
    for i in range(1, n):
        w = b[i - 1] / d[i - 1]
        d[i] -= w * c[i - 1]
        y[i] -= w * y[i - 1]

    # Back substitution
    x = np.empty(n, dtype=float)
    x[-1] = y[-1] / d[-1]
    for i in range(n - 2, -1, -1):
        x[i] = (y[i] - c[i] * x[i + 1]) / d[i]
    return x


def _european_vanilla_fd_cn(
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
    """European vanilla option price via Crank–Nicolson FD.

    Solves in the transformed time variable tau (time remaining):
    - at maturity: tau = 0
    - at pricing date: tau = T

    Returns:
      (price_at_spot, spot_grid_S, values_V_at_tau=T)
    """
    if option_type not in (OptionType.CALL, OptionType.PUT):
        raise NotImplementedError("FD PDE valuation supports only vanilla CALL/PUT.")
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
    dt = time_to_maturity / time_steps

    S = np.linspace(0.0, smax, spot_steps + 1)
    i = np.arange(1, spot_steps)  # interior indices 1..M-1
    Si = S[i]

    if option_type is OptionType.PUT:
        payoff = np.maximum(strike - S, 0.0)
    else:
        payoff = np.maximum(S - strike, 0.0)

    V = payoff.copy()  # V at tau=0 (maturity)

    def boundary_values(tau: float) -> tuple[float, float]:
        if option_type is OptionType.PUT:
            left = strike * np.exp(-risk_free_rate * tau)
            right = 0.0
        else:
            left = 0.0
            right = smax * np.exp(-dividend_yield * tau) - strike * np.exp(-risk_free_rate * tau)
            right = max(right, 0.0)
        return float(left), float(right)

    # CN coefficients
    a = (
        0.25
        * dt
        * (volatility**2 * (Si**2) / (dS**2) - (risk_free_rate - dividend_yield) * Si / dS)
    )
    b = -0.5 * dt * (volatility**2 * (Si**2) / (dS**2) + risk_free_rate)
    c = (
        0.25
        * dt
        * (volatility**2 * (Si**2) / (dS**2) + (risk_free_rate - dividend_yield) * Si / dS)
    )

    # LHS: (I - A); RHS: (I + A)
    # Interior system size = spot_steps - 1
    L_lower = -a[1:]  # length n-1
    L_diag = 1.0 - b  # length n
    L_upper = -c[:-1]  # length n-1

    R_lower = a[1:]
    R_diag = 1.0 + b
    R_upper = c[:-1]

    # March forward in tau: 0 -> T (equivalently backward in calendar time)
    for n in range(time_steps):
        tau = (n + 1) * dt
        left, right = boundary_values(tau)

        # Apply boundary values at current tau
        V[0] = left
        V[-1] = right

        V_old = V.copy()

        # RHS for interior nodes
        rhs = (
            R_diag * V_old[i]
            + np.concatenate(([0.0], R_lower)) * V_old[i - 1]  # align lengths safely
            + np.concatenate((R_upper, [0.0])) * V_old[i + 1]
        )

        # Boundary adjustments (interior equation references V[0] and V[-1])
        rhs[0] += a[0] * V_old[0]  # i=1 uses V[0]
        rhs[-1] += c[-1] * V_old[-1]  # i=M-1 uses V[M]

        # Solve tridiagonal system for interior values at new tau
        x = _solve_tridiagonal_thomas(L_lower, L_diag, L_upper, rhs)
        V[i] = x

    price = np.interp(spot, S, V)
    return price, S, V


def _american_vanilla_fd_cn_psor(
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
    """American vanilla option price via CN + PSOR.

    Solves in the transformed time variable tau (time remaining):
    - at maturity: tau = 0
    - at pricing date: tau = T
    """
    if option_type not in (OptionType.CALL, OptionType.PUT):
        raise NotImplementedError("FD PDE valuation supports only vanilla CALL/PUT.")
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
    dt = time_to_maturity / time_steps

    S = np.linspace(0.0, smax, spot_steps + 1)
    i = np.arange(1, spot_steps)  # interior indices
    Si = S[i]

    if option_type is OptionType.PUT:
        payoff = np.maximum(strike - S, 0.0)
    else:
        payoff = np.maximum(S - strike, 0.0)

    V = payoff.copy()  # V at tau=0
    intrinsic = payoff  # intrinsic is time-invariant on this grid

    def boundary_values(tau: float) -> tuple[float, float]:
        if option_type is OptionType.PUT:
            left = strike * np.exp(-risk_free_rate * tau)
            right = 0.0
        else:
            left = 0.0
            right = smax * np.exp(-dividend_yield * tau) - strike * np.exp(-risk_free_rate * tau)
            right = max(right, 0.0)
        return float(left), float(right)

    a = (
        0.25
        * dt
        * (volatility**2 * (Si**2) / (dS**2) - (risk_free_rate - dividend_yield) * Si / dS)
    )
    b = -0.5 * dt * (volatility**2 * (Si**2) / (dS**2) + risk_free_rate)
    c = (
        0.25
        * dt
        * (volatility**2 * (Si**2) / (dS**2) + (risk_free_rate - dividend_yield) * Si / dS)
    )

    # LHS: (I - A); RHS: (I + A)
    L_lower = -a
    L_diag = 1.0 - b
    L_upper = -c

    R_lower = a
    R_diag = 1.0 + b
    R_upper = c
    # March forward in tau: 0 -> T (equivalently backward in calendar time)
    for n in range(time_steps):
        tau = (n + 1) * dt
        left, right = boundary_values(tau)
        V[0] = left
        V[-1] = right

        V_old = V.copy()
        rhs = R_lower * V_old[i - 1] + R_diag * V_old[i] + R_upper * V_old[i + 1]

        x = V_old[i].copy()  # initial guess
        exercise_i = intrinsic[i]

        for _ in range(max_iter):
            x_prev = x.copy()

            for k in range(x.size):
                left_val = x[k - 1] if k > 0 else V[0]
                right_val = x[k + 1] if k < x.size - 1 else V[-1]

                gs = (rhs[k] - L_lower[k] * left_val - L_upper[k] * right_val) / L_diag[k]
                sor = x[k] + omega * (gs - x[k])
                x[k] = max(sor, exercise_i[k])

            if np.max(np.abs(x - x_prev)) < tol:
                break

        V[i] = x

    price = np.interp(spot, S, V)
    return price, S, V


class _FDEuropeanValuation:
    """European option valuation using PDE finite differences (Crank–Nicolson)."""

    def __init__(self, parent: "OptionValuation"):
        self.parent = parent

    def generate_payoff(self, **kwargs) -> tuple[float, np.ndarray, np.ndarray]:
        """Generate the full FD solution on the spot grid at pricing time."""
        pv, S, V = self._solve(**kwargs)
        return pv, S, V

    def _solve(self, **kwargs) -> tuple[float, np.ndarray, np.ndarray]:
        spot = float(self.parent.underlying.initial_value)
        strike = self.parent.strike
        if strike is None:
            raise ValueError("strike is required for PDE FD valuation")

        volatility = float(self.parent.underlying.volatility)
        risk_free_rate = float(self.parent.discount_curve.short_rate)
        dividend_yield = float(getattr(self.parent.underlying, "dividend_yield", 0.0))

        time_to_maturity = calculate_year_fraction(self.parent.pricing_date, self.parent.maturity)

        smax_mult = float(kwargs.get("smax_mult", 4.0))
        spot_steps = int(kwargs.get("spot_steps", kwargs.get("M", 400)))
        time_steps = int(kwargs.get("time_steps", kwargs.get("N", 400)))

        return _european_vanilla_fd_cn(
            spot=spot,
            strike=float(strike),
            time_to_maturity=float(time_to_maturity),
            risk_free_rate=risk_free_rate,
            volatility=volatility,
            dividend_yield=dividend_yield,
            option_type=self.parent.option_type,
            smax_mult=smax_mult,
            spot_steps=spot_steps,
            time_steps=time_steps,
        )

    def present_value(
        self, full: bool = False, **kwargs
    ) -> float | tuple[float, np.ndarray, np.ndarray]:
        pv, S, V = self._solve(**kwargs)
        if full:
            return pv, S, V
        return pv


class _FDAmericanValuation:
    """American option valuation using PDE finite differences (CN + PSOR)."""

    def __init__(self, parent: "OptionValuation"):
        self.parent = parent

    def generate_payoff(self, **kwargs) -> tuple[float, np.ndarray, np.ndarray]:
        """Generate the full FD solution on the spot grid at pricing time."""
        pv, S, V = self._solve(**kwargs)
        return pv, S, V

    def _solve(self, **kwargs) -> tuple[float, np.ndarray, np.ndarray]:
        spot = float(self.parent.underlying.initial_value)
        strike = self.parent.strike
        if strike is None:
            raise ValueError("strike is required for PDE FD valuation")

        volatility = float(self.parent.underlying.volatility)
        risk_free_rate = float(self.parent.discount_curve.short_rate)
        dividend_yield = float(getattr(self.parent.underlying, "dividend_yield", 0.0))

        time_to_maturity = calculate_year_fraction(self.parent.pricing_date, self.parent.maturity)

        smax_mult = float(kwargs.get("smax_mult", 4.0))
        spot_steps = int(kwargs.get("spot_steps", kwargs.get("M", 400)))
        time_steps = int(kwargs.get("time_steps", kwargs.get("N", 400)))
        omega = float(kwargs.get("omega", 1.2))
        tol = float(kwargs.get("tol", 1e-8))
        max_iter = int(kwargs.get("max_iter", 50_000))

        return _american_vanilla_fd_cn_psor(
            spot=spot,
            strike=float(strike),
            time_to_maturity=float(time_to_maturity),
            risk_free_rate=risk_free_rate,
            volatility=volatility,
            dividend_yield=dividend_yield,
            option_type=self.parent.option_type,
            smax_mult=smax_mult,
            spot_steps=spot_steps,
            time_steps=time_steps,
            omega=omega,
            tol=tol,
            max_iter=max_iter,
        )

    def present_value(
        self, full: bool = False, **kwargs
    ) -> float | tuple[float, np.ndarray, np.ndarray]:
        pv, S, V = self._solve(**kwargs)
        if full:
            return pv, S, V
        return pv
