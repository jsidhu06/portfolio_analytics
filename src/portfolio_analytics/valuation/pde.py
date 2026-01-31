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

from ..enums import OptionType
from ..utils import calculate_year_fraction
from .params import PDEParams

if TYPE_CHECKING:
    from .core import OptionValuation


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


def _build_tau_grid(
    time_to_maturity: float,
    time_steps: int,
    dividend_taus: list[float],
) -> np.ndarray:
    """Build a tau (time remaining) grid that includes dividend dates."""
    base = np.linspace(0.0, time_to_maturity, time_steps + 1)
    if not dividend_taus:
        return base
    grid = np.unique(np.concatenate([base, np.array(dividend_taus, dtype=float)]))
    grid.sort()
    return grid


def _dividend_tau_schedule(
    *,
    discrete_dividends: list[tuple],
    pricing_date,
    maturity,
) -> list[tuple[float, float]]:
    """Return list of (tau, amount) for dividends strictly before maturity."""
    if not discrete_dividends:
        return []

    ttm = calculate_year_fraction(pricing_date, maturity)
    schedule: dict[float, float] = {}
    for ex_date, amount in discrete_dividends:
        if pricing_date < ex_date < maturity:
            t = calculate_year_fraction(pricing_date, ex_date)
            tau = ttm - t
            if 0.0 < tau < ttm:
                key = round(float(tau), 12)
                schedule[key] = schedule.get(key, 0.0) + float(amount)
    return sorted(schedule.items())


def _apply_dividend_jump(
    values: np.ndarray,
    spot_grid: np.ndarray,
    amount: float,
) -> np.ndarray:
    """Apply the cash dividend jump condition V(S,t^-)=V(S-D,t^+)."""
    if amount == 0.0:
        return values
    shifted = np.interp(
        spot_grid - amount,
        spot_grid,
        values,
        left=values[0],
        right=values[-1],
    )
    values[:] = shifted
    return values


def _european_vanilla_fd_cn(
    *,
    spot: float,
    strike: float,
    time_to_maturity: float,
    risk_free_rate: float,
    volatility: float,
    dividend_yield: float,
    dividend_schedule: list[tuple[float, float]] | None,
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

    schedule = dividend_schedule or []
    dividend_taus = [tau for tau, _ in schedule]
    tau_grid = _build_tau_grid(time_to_maturity, time_steps, dividend_taus)
    dividend_map = {tau: amount for tau, amount in schedule}

    # March forward in tau: 0 -> T (equivalently backward in calendar time)
    for n in range(1, tau_grid.size):
        dt = tau_grid[n] - tau_grid[n - 1]
        tau = float(tau_grid[n])

        # CN coefficients (depend on dt)
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

        # Apply discrete dividend jump at tau if needed
        if dividend_map:
            amount = dividend_map.get(round(tau, 12))
            if amount is not None:
                _apply_dividend_jump(V, S, amount)

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
    dividend_schedule: list[tuple[float, float]] | None,
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

    schedule = dividend_schedule or []
    dividend_taus = [tau for tau, _ in schedule]
    tau_grid = _build_tau_grid(time_to_maturity, time_steps, dividend_taus)
    dividend_map = {tau: amount for tau, amount in schedule}

    # March forward in tau: 0 -> T (equivalently backward in calendar time)
    for n in range(1, tau_grid.size):
        dt = tau_grid[n] - tau_grid[n - 1]
        tau = float(tau_grid[n])

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
        left, right = boundary_values(tau)
        V[0] = left
        V[-1] = right

        V_old = V.copy()
        rhs = R_lower * V_old[i - 1] + R_diag * V_old[i] + R_upper * V_old[i + 1]

        exercise_i = intrinsic[i]

        # Warm-start: solve the unconstrained CN system (European step) using Thomas,
        # then project onto the early-exercise constraint.
        #
        # The linear system for interior unknowns x has boundary terms embedded via V[0], V[-1].
        rhs_adj = rhs.copy()
        rhs_adj[0] -= L_lower[0] * V[0]
        rhs_adj[-1] -= L_upper[-1] * V[-1]

        th_lower = L_lower[1:]
        th_diag = L_diag
        th_upper = L_upper[:-1]
        x = _solve_tridiagonal_thomas(th_lower, th_diag, th_upper, rhs_adj)
        x = np.maximum(x, exercise_i)

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

        # Apply discrete dividend jump at tau if needed
        if dividend_map:
            amount = dividend_map.get(round(tau, 12))
            if amount is not None:
                _apply_dividend_jump(V, S, amount)
                V[:] = np.maximum(V, intrinsic)

    price = np.interp(spot, S, V)
    return price, S, V


class _FDEuropeanValuation:
    """European option valuation using PDE finite differences (Crank–Nicolson)."""

    def __init__(self, parent: "OptionValuation"):
        self.parent = parent

    def solve(self, params: PDEParams) -> tuple[float, np.ndarray, np.ndarray]:
        """Compute the full FD solution on the spot grid at pricing time."""
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
        discrete_dividends = self.parent.underlying.discrete_dividends

        time_to_maturity = calculate_year_fraction(self.parent.pricing_date, self.parent.maturity)

        dividend_schedule = _dividend_tau_schedule(
            discrete_dividends=discrete_dividends,
            pricing_date=self.parent.pricing_date,
            maturity=self.parent.maturity,
        )

        smax_mult = float(params.smax_mult)
        spot_steps = int(params.spot_steps)
        time_steps = int(params.time_steps)

        return _european_vanilla_fd_cn(
            spot=spot,
            strike=float(strike),
            time_to_maturity=float(time_to_maturity),
            risk_free_rate=risk_free_rate,
            volatility=volatility,
            dividend_yield=dividend_yield,
            dividend_schedule=dividend_schedule,
            option_type=self.parent.option_type,
            smax_mult=smax_mult,
            spot_steps=spot_steps,
            time_steps=time_steps,
        )

    def present_value(self, params: PDEParams) -> float:
        pv, *_ = self._solve(params)
        return float(pv)


class _FDAmericanValuation:
    """American option valuation using PDE finite differences (CN + PSOR)."""

    def __init__(self, parent: "OptionValuation"):
        self.parent = parent

    def solve(self, params: PDEParams) -> tuple[float, np.ndarray, np.ndarray]:
        """Compute the full FD solution on the spot grid at pricing time."""
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
        discrete_dividends = self.parent.underlying.discrete_dividends

        # Special-case: American CALL with ~0 dividends has no early-exercise premium.
        # Avoid the PSOR loop entirely and price as European via CN.
        if (
            self.parent.option_type is OptionType.CALL
            and abs(dividend_yield) < 1e-12
            and not discrete_dividends
        ):
            time_to_maturity = calculate_year_fraction(
                self.parent.pricing_date, self.parent.maturity
            )

            dividend_schedule = _dividend_tau_schedule(
                discrete_dividends=discrete_dividends,
                pricing_date=self.parent.pricing_date,
                maturity=self.parent.maturity,
            )

            smax_mult = float(params.smax_mult)
            spot_steps = int(params.spot_steps)
            time_steps = int(params.time_steps)

            return _european_vanilla_fd_cn(
                spot=spot,
                strike=float(strike),
                time_to_maturity=float(time_to_maturity),
                risk_free_rate=risk_free_rate,
                volatility=volatility,
                dividend_yield=dividend_yield,
                dividend_schedule=dividend_schedule,
                option_type=self.parent.option_type,
                smax_mult=smax_mult,
                spot_steps=spot_steps,
                time_steps=time_steps,
            )

        time_to_maturity = calculate_year_fraction(self.parent.pricing_date, self.parent.maturity)

        dividend_schedule = _dividend_tau_schedule(
            discrete_dividends=discrete_dividends,
            pricing_date=self.parent.pricing_date,
            maturity=self.parent.maturity,
        )

        smax_mult = float(params.smax_mult)
        spot_steps = int(params.spot_steps)
        time_steps = int(params.time_steps)
        omega = float(params.omega)
        tol = float(params.tol)
        max_iter = int(params.max_iter)

        return _american_vanilla_fd_cn_psor(
            spot=spot,
            strike=float(strike),
            time_to_maturity=float(time_to_maturity),
            risk_free_rate=risk_free_rate,
            volatility=volatility,
            dividend_yield=dividend_yield,
            dividend_schedule=dividend_schedule,
            option_type=self.parent.option_type,
            smax_mult=smax_mult,
            spot_steps=spot_steps,
            time_steps=time_steps,
            omega=omega,
            tol=tol,
            max_iter=max_iter,
        )

    def present_value(self, params: PDEParams) -> float:
        pv, *_ = self._solve(params)
        return float(pv)
