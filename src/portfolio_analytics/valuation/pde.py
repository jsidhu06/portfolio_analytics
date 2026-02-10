"""Finite difference (PDE) valuation implementations.

This module follows the same structure as other valuation modules:
- an internal implementation class that plugs into OptionValuation
- thin convenience wrappers for direct function-style pricing

Current scope
-------------
PDE via finite differences for vanilla European and American call/put:
- time stepping: implicit, explicit, or Crank–Nicolson
- spatial grids: spot or log-spot
- American handling: intrinsic projection or Gauss-Seidel/PSOR
"""

from typing import TYPE_CHECKING

import math

import numpy as np

from ..enums import PDEEarlyExercise, PDEMethod, PDESpaceGrid, OptionType
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
    grid: np.ndarray,
    amount: float,
    *,
    space_grid: PDESpaceGrid,
) -> None:
    """Apply the cash dividend jump condition V(S,t^-)=V(S-D,t^+)."""
    if amount == 0.0:
        return
    if space_grid is PDESpaceGrid.LOG_SPOT:
        spot_grid = np.exp(grid)
    else:
        spot_grid = grid
    shifted = np.interp(
        spot_grid - amount,
        spot_grid,
        values,
        left=values[0],
        right=values[-1],
    )
    values[:] = shifted


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


def _build_log_grid(
    *,
    spot: float,
    strike: float,
    time_to_maturity: float,
    volatility: float,
    smax_mult: float,
    spot_steps: int,
    time_steps: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Build log-spot grid using Hull's dz heuristic when possible."""
    dt = time_to_maturity / time_steps
    dz_hull = volatility * np.sqrt(3.0 * dt)

    smax = float(smax_mult * max(spot, strike))
    smin = float(max(max(spot, strike) / smax_mult, 1.0e-8))
    zmin_target = np.log(smin)
    zmax_target = np.log(smax)

    grid_width = spot_steps * dz_hull
    if (zmax_target - zmin_target) > grid_width:
        dz = (zmax_target - zmin_target) / spot_steps
        zmin = zmin_target
        zmax = zmax_target
    else:
        dz = dz_hull
        center = np.log(spot)
        zmin = center - 0.5 * grid_width
        zmax = center + 0.5 * grid_width
        if zmin > zmin_target:
            shift = zmin_target - zmin
            zmin += shift
            zmax += shift
        if zmax < zmax_target:
            shift = zmax_target - zmax
            zmin += shift
            zmax += shift

    Z = np.linspace(zmin, zmax, spot_steps + 1)
    S = np.exp(Z)
    return Z, S, dz


def _spot_operator_coeffs(
    *,
    spot_values: np.ndarray,
    dS: float,
    risk_free_rate: float,
    dividend_yield: float,
    volatility: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    diffusion = (volatility**2) * (spot_values**2) / (dS**2)
    drift = (risk_free_rate - dividend_yield) * spot_values / dS
    gamma = 0.5 * (diffusion - drift)
    beta = -(diffusion + risk_free_rate)
    alpha = 0.5 * (diffusion + drift)
    return gamma, beta, alpha


def _log_operator_coeffs(
    *,
    dz: float,
    risk_free_rate: float,
    dividend_yield: float,
    volatility: float,
) -> tuple[float, float, float]:
    mu = risk_free_rate - dividend_yield - 0.5 * volatility**2
    diffusion = (volatility**2) / (dz**2)
    drift = mu / dz
    gamma = 0.5 * (diffusion - drift)
    beta = -(diffusion + risk_free_rate)
    alpha = 0.5 * (diffusion + drift)
    return gamma, beta, alpha


def _scaled_operator_coeffs(
    *,
    gamma: np.ndarray | float,
    beta: np.ndarray | float,
    alpha: np.ndarray | float,
    dt: float,
) -> tuple[np.ndarray | float, np.ndarray | float, np.ndarray | float]:
    a = -dt * gamma
    b = -dt * beta
    c = -dt * alpha
    return a, b, c


def _as_array(coeff: np.ndarray | float, size: int) -> np.ndarray:
    if isinstance(coeff, np.ndarray):
        return coeff
    return np.full(size, float(coeff))


def _vanilla_fd_core(
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
    early_exercise: bool,
    method: PDEMethod,
    space_grid: PDESpaceGrid,
    american_solver: PDEEarlyExercise,
    omega: float | None = None,
    tol: float | None = None,
    max_iter: int | None = None,
) -> tuple[float, np.ndarray, np.ndarray]:
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
    if early_exercise and american_solver is PDEEarlyExercise.GAUSS_SEIDEL:
        if omega is None or tol is None or max_iter is None:
            raise ValueError("PSOR params (omega/tol/max_iter) are required for early exercise")
    if method is PDEMethod.EXPLICIT and american_solver is PDEEarlyExercise.GAUSS_SEIDEL:
        raise ValueError("GAUSS_SEIDEL is not supported with explicit time stepping")

    smax = float(smax_mult * max(spot, strike))
    if space_grid is PDESpaceGrid.SPOT:
        grid = np.linspace(0.0, smax, spot_steps + 1)
        S = grid
        dS = smax / spot_steps
        gamma, beta, alpha = _spot_operator_coeffs(
            spot_values=S[1:-1],
            dS=dS,
            risk_free_rate=risk_free_rate,
            dividend_yield=dividend_yield,
            volatility=volatility,
        )
    else:
        grid, S, dz = _build_log_grid(
            spot=spot,
            strike=strike,
            time_to_maturity=time_to_maturity,
            volatility=volatility,
            smax_mult=smax_mult,
            spot_steps=spot_steps,
            time_steps=time_steps,
        )
        gamma, beta, alpha = _log_operator_coeffs(
            dz=dz,
            risk_free_rate=risk_free_rate,
            dividend_yield=dividend_yield,
            volatility=volatility,
        )

    j = np.arange(1, spot_steps)  # interior indices 1..M-1

    if option_type is OptionType.PUT:
        payoff = np.maximum(strike - S, 0.0)
    else:
        payoff = np.maximum(S - strike, 0.0)

    V = payoff.copy()  # V at tau=0 (maturity)
    intrinsic = payoff if early_exercise else None

    schedule = dividend_schedule or []
    dividend_taus = [tau for tau, _ in schedule]
    tau_grid = _build_tau_grid(time_to_maturity, time_steps, dividend_taus)
    dividend_map = {round(tau, 12): amount for tau, amount in schedule}

    if method is PDEMethod.EXPLICIT and space_grid is PDESpaceGrid.SPOT:
        drift = abs(risk_free_rate - dividend_yield)
        if drift > volatility**2:
            raise ValueError(
                "Explicit spot-grid scheme unstable: |r-q| must be <= sigma^2. "
                "Use log-spot or implicit/CN."
            )

        denom = (volatility**2) * (smax**2) / (dS**2) + max(risk_free_rate, 0.0)
        if denom > 0.0:
            dt_max = 1.0 / denom
            max_dt = float(np.max(np.diff(tau_grid)))
            if max_dt > dt_max:
                min_steps = int(math.ceil(time_to_maturity / dt_max))
                raise ValueError(
                    "Explicit spot-grid scheme unstable: time step too large. "
                    f"max_dt={max_dt:.4g} exceeds dt_max={dt_max:.4g}. "
                    f"Increase time_steps to >= {min_steps} or use log-spot/implicit/CN."
                )

    # March forward in tau: 0 -> T (equivalently backward in calendar time)
    for n in range(1, tau_grid.size):
        dt = tau_grid[n] - tau_grid[n - 1]
        tau = float(tau_grid[n])

        left, right = _boundary_values(
            option_type=option_type,
            strike=strike,
            smax=float(S[-1]),
            tau=tau,
            r=risk_free_rate,
            q=dividend_yield,
            early_exercise=early_exercise,
        )

        V_old = V.copy()

        a, b, c = _scaled_operator_coeffs(gamma=gamma, beta=beta, alpha=alpha, dt=dt)

        if method is PDEMethod.EXPLICIT:
            V_new = V_old.copy()
            V_new[j] = -a * V_old[j - 1] + (1.0 - b) * V_old[j] - c * V_old[j + 1]
            V_new[0] = left
            V_new[-1] = right
            if early_exercise:
                V_new[:] = np.maximum(V_new, intrinsic)
            V = V_new
        else:
            if method is PDEMethod.CRANK_NICOLSON:
                a *= 0.5
                b *= 0.5
                c *= 0.5

            # A = -dt * L (L being the tridiagonal matrix of gamma/beta/alpha).
            A_lower = _as_array(a, spot_steps - 1)
            A_diag = _as_array(1.0 + b, spot_steps - 1)
            A_upper = _as_array(c, spot_steps - 1)

            if method is PDEMethod.IMPLICIT:
                rhs = V_old[j].copy()
            else:
                rhs = -a * V_old[j - 1] + (1.0 - b) * V_old[j] - c * V_old[j + 1]

            V[0] = left
            V[-1] = right

            rhs_adj = rhs.copy()
            rhs_adj[0] -= A_lower[0] * V[0]
            rhs_adj[-1] -= A_upper[-1] * V[-1]

            x = _solve_tridiagonal_thomas(A_lower[1:], A_diag, A_upper[:-1], rhs_adj)
            if not early_exercise:
                V[j] = x
            else:
                exercise_j = intrinsic[j]
                if american_solver is PDEEarlyExercise.INTRINSIC:
                    V[j] = np.maximum(x, exercise_j)
                else:
                    x = np.maximum(x, exercise_j)
                    for _ in range(int(max_iter)):
                        x_prev = x.copy()
                        for k in range(x.size):
                            left_val = x[k - 1] if k > 0 else V[0]
                            right_val = x[k + 1] if k < x.size - 1 else V[-1]
                            gs = (rhs[k] - A_lower[k] * left_val - A_upper[k] * right_val) / A_diag[
                                k
                            ]
                            sor = x[k] + float(omega) * (gs - x[k])
                            x[k] = max(sor, exercise_j[k])
                        if np.max(np.abs(x - x_prev)) < float(tol):
                            break
                    V[j] = x

        # Apply discrete dividend jump at tau if needed
        if dividend_map:
            amount = dividend_map.get(round(tau, 12))
            if amount is not None:
                _apply_dividend_jump(V, grid, amount, space_grid=space_grid)
                if early_exercise:
                    V[:] = np.maximum(V, intrinsic)

    price = np.interp(spot, S, V)
    return price, S, V


def _european_vanilla_fd(
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
    method: PDEMethod,
    space_grid: PDESpaceGrid,
) -> tuple[float, np.ndarray, np.ndarray]:
    return _vanilla_fd_core(
        spot=spot,
        strike=strike,
        time_to_maturity=time_to_maturity,
        risk_free_rate=risk_free_rate,
        volatility=volatility,
        dividend_yield=dividend_yield,
        dividend_schedule=dividend_schedule,
        option_type=option_type,
        smax_mult=smax_mult,
        spot_steps=spot_steps,
        time_steps=time_steps,
        early_exercise=False,
        method=method,
        space_grid=space_grid,
        american_solver=PDEEarlyExercise.INTRINSIC,
    )


def _american_vanilla_fd(
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
    method: PDEMethod,
    space_grid: PDESpaceGrid,
    american_solver: PDEEarlyExercise,
    omega: float,
    tol: float,
    max_iter: int,
) -> tuple[float, np.ndarray, np.ndarray]:
    return _vanilla_fd_core(
        spot=spot,
        strike=strike,
        time_to_maturity=time_to_maturity,
        risk_free_rate=risk_free_rate,
        volatility=volatility,
        dividend_yield=dividend_yield,
        dividend_schedule=dividend_schedule,
        option_type=option_type,
        smax_mult=smax_mult,
        spot_steps=spot_steps,
        time_steps=time_steps,
        early_exercise=True,
        method=method,
        space_grid=space_grid,
        american_solver=american_solver,
        omega=omega,
        tol=tol,
        max_iter=max_iter,
    )


class _FDEuropeanValuation:
    """European option valuation using PDE finite differences."""

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

        return _european_vanilla_fd(
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
            method=params.method,
            space_grid=params.space_grid,
        )

    def present_value(self, params: PDEParams) -> float:
        pv, *_ = self._solve(params)
        return float(pv)


class _FDAmericanValuation:
    """American option valuation using PDE finite differences."""

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
        omega = float(params.omega)
        tol = float(params.tol)
        max_iter = int(params.max_iter)

        return _american_vanilla_fd(
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
            method=params.method,
            space_grid=params.space_grid,
            american_solver=params.american_solver,
            omega=omega,
            tol=tol,
            max_iter=max_iter,
        )

    def present_value(self, params: PDEParams) -> float:
        pv, *_ = self._solve(params)
        return float(pv)
