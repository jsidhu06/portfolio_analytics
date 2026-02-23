"""Finite difference (PDE) valuation implementations.

This module follows the same structure as other valuation modules:
- an internal implementation class that plugs into OptionValuation
- thin convenience wrappers for direct function-style pricing

Current scope
-------------
PDE via finite differences for vanilla European and American call/put:
- time stepping: implicit, explicit, or Crank–Nicolson
- optional Rannacher smoothing for Crank–Nicolson
- spatial grids: spot or log-spot
- American handling: intrinsic projection or Gauss-Seidel/PSOR
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Sequence

import logging
import math
import datetime as dt

import numpy as np

from ..enums import PDEEarlyExercise, PDEMethod, PDESpaceGrid, OptionType
from ..rates import DiscountCurve
from ..utils import calculate_year_fraction, log_timing
from ..exceptions import (
    ConfigurationError,
    StabilityError,
    UnsupportedFeatureError,
    ValidationError,
)
from .params import PDEParams

if TYPE_CHECKING:
    from .core import OptionValuation


logger = logging.getLogger(__name__)


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
        raise ValidationError("rhs length must match diag length")
    if lower.size != n - 1 or upper.size != n - 1:
        raise ValidationError("lower/upper must have length n-1")

    # Copy to avoid mutating inputs
    c: np.ndarray = upper.astype(float, copy=True)
    d: np.ndarray = diag.astype(float, copy=True)
    b: np.ndarray = lower.astype(float, copy=True)
    y: np.ndarray = rhs.astype(float, copy=True)

    # Forward elimination
    for i in range(1, n):
        w = b[i - 1] / d[i - 1]
        d[i] -= w * c[i - 1]
        y[i] -= w * y[i - 1]

    # Back substitution
    x: np.ndarray = np.empty(n, dtype=float)
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
    discrete_dividends: Sequence[tuple[dt.datetime, float]],
    pricing_date: dt.datetime,
    maturity: dt.datetime,
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
    df_tT: float,
    dq_tT: float,
    early_exercise: bool,
) -> tuple[float, float]:
    if option_type is OptionType.PUT:
        left = strike if early_exercise else strike * df_tT
        right = 0.0
    else:
        left = 0.0
        continuation = smax * dq_tT - strike * df_tT
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
    d_tau = time_to_maturity / time_steps
    dz_hull = volatility * np.sqrt(3.0 * d_tau)

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
    dividend_rate: float,
    volatility: float,
    hull_discounting: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Spatial operator coefficients on the spot grid.

    When *hull_discounting* is True (Hull's explicit scheme), the rV
    term is excluded from beta and instead applied as an implicit
    divisor ``1 / (1 + r * dt)`` in the time-step function.
    """
    diffusion = (volatility**2) * (spot_values**2) / (dS**2)
    drift = (risk_free_rate - dividend_rate) * spot_values / dS
    gamma = 0.5 * (diffusion - drift)
    beta = -diffusion if hull_discounting else -(diffusion + risk_free_rate)
    alpha = 0.5 * (diffusion + drift)
    return gamma, beta, alpha


def _log_operator_coeffs(
    *,
    dz: float,
    risk_free_rate: float,
    dividend_rate: float,
    volatility: float,
    hull_discounting: bool = False,
) -> tuple[float, float, float]:
    """Spatial operator coefficients on the log-spot grid.

    When *hull_discounting* is True (Hull's explicit scheme), r is
    excluded from beta.
    """
    mu = risk_free_rate - dividend_rate - 0.5 * volatility**2
    diffusion = (volatility**2) / (dz**2)
    drift = mu / dz
    gamma = 0.5 * (diffusion - drift)
    beta = -diffusion if hull_discounting else -(diffusion + risk_free_rate)
    alpha = 0.5 * (diffusion + drift)
    return gamma, beta, alpha


def _scaled_operator_coeffs(
    *,
    gamma: np.ndarray | float,
    beta: np.ndarray | float,
    alpha: np.ndarray | float,
    d_tau: float,
) -> tuple[np.ndarray | float, np.ndarray | float, np.ndarray | float]:
    a = -d_tau * gamma
    b = -d_tau * beta
    c = -d_tau * alpha
    return a, b, c


def _as_array(coeff: np.ndarray | float, size: int) -> np.ndarray:
    if isinstance(coeff, np.ndarray):
        return coeff
    return np.full(size, float(coeff))


# ---------------------------------------------------------------------------
# Time-step helpers (extracted from _vanilla_fd_core for readability)
# ---------------------------------------------------------------------------


def _explicit_step(
    V_old: np.ndarray,
    j: np.ndarray,
    a: np.ndarray | float,
    b: np.ndarray | float,
    c: np.ndarray | float,
    left: float,
    right: float,
    intrinsic: np.ndarray | None,
    *,
    r_dt: float = 0.0,
) -> np.ndarray:
    """Explicit (forward-Euler) time step.

    When *r_dt* > 0 (Hull's explicit scheme), the interior update is
    divided by ``(1 + r_dt)`` to apply implicit discounting of the rV
    term.
    """
    V_new = V_old.copy()
    interior = -a * V_old[j - 1] + (1.0 - b) * V_old[j] - c * V_old[j + 1]
    V_new[j] = interior / (1.0 + r_dt)
    V_new[0] = left
    V_new[-1] = right
    if intrinsic is not None:
        V_new[:] = np.maximum(V_new, intrinsic)
    return V_new


def _psor_solve(
    x: np.ndarray,
    exercise_j: np.ndarray,
    rhs: np.ndarray,
    A_lower: np.ndarray,
    A_diag: np.ndarray,
    A_upper: np.ndarray,
    V_left: float | np.floating,
    V_right: float | np.floating,
    omega: float,
    tol: float,
    max_iter: int,
) -> tuple[np.ndarray, int]:
    """Projected SOR (Gauss-Seidel with overrelaxation) for American exercise.

    Returns the updated interior values and the number of iterations used.
    """
    x = np.maximum(x, exercise_j)
    iter_used = max_iter
    for iter_idx in range(max_iter):
        x_prev = x.copy()
        for k in range(x.size):
            left_val = x[k - 1] if k > 0 else V_left
            right_val = x[k + 1] if k < x.size - 1 else V_right
            gs = (rhs[k] - A_lower[k] * left_val - A_upper[k] * right_val) / A_diag[k]
            sor = x[k] + omega * (gs - x[k])
            x[k] = max(sor, exercise_j[k])
        if np.max(np.abs(x - x_prev)) < tol:
            iter_used = iter_idx + 1
            break
    return x, iter_used


def _implicit_cn_step(
    V_old: np.ndarray,
    V: np.ndarray,
    j: np.ndarray,
    a: np.ndarray | float,
    b: np.ndarray | float,
    c: np.ndarray | float,
    left: float,
    right: float,
    method: "PDEMethod",
    spot_steps: int,
    intrinsic: np.ndarray | None,
    american_solver: "PDEEarlyExercise",
    omega: float | None,
    tol: float | None,
    max_iter: int | None,
) -> tuple[np.ndarray, int | None]:
    """One implicit or Crank-Nicolson time step (with optional early exercise).

    Returns the updated V array and the PSOR iteration count (None if PSOR was not used).
    """
    if method is PDEMethod.CRANK_NICOLSON:
        a = a * 0.5
        b = b * 0.5
        c = c * 0.5

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
    psor_iters: int | None = None

    if intrinsic is None:
        V[j] = x
    else:
        exercise_j = intrinsic[j]
        if american_solver is PDEEarlyExercise.INTRINSIC:
            V[j] = np.maximum(x, exercise_j)
        else:
            x, psor_iters = _psor_solve(
                x,
                exercise_j,
                rhs,
                A_lower,
                A_diag,
                A_upper,
                float(V[0]),
                float(V[-1]),
                float(omega),
                float(tol),
                int(max_iter),
            )
            V[j] = x

    return V, psor_iters


def _validate_fd_inputs(
    *,
    option_type: OptionType,
    time_to_maturity: float,
    spot_steps: int,
    time_steps: int,
    volatility: float,
    discount_curve: DiscountCurve,
    early_exercise: bool,
    method: PDEMethod,
    american_solver: PDEEarlyExercise,
    omega: float | None,
    tol: float | None,
    max_iter: int | None,
) -> None:
    """Validate FD PDE inputs before grid construction."""
    if option_type not in (OptionType.CALL, OptionType.PUT):
        raise UnsupportedFeatureError("FD PDE valuation supports only vanilla CALL/PUT.")
    if time_to_maturity <= 0:
        raise ValidationError("time_to_maturity must be positive")
    if spot_steps < 3:
        raise ValidationError("spot_steps must be >= 3")
    if time_steps < 1:
        raise ValidationError("time_steps must be >= 1")
    if volatility <= 0:
        raise ValidationError("volatility must be positive")
    if discount_curve is None:
        raise ValidationError("discount_curve is required for PDE valuation")
    if early_exercise and american_solver is PDEEarlyExercise.GAUSS_SEIDEL:
        if omega is None or tol is None or max_iter is None:
            raise ValidationError(
                "PSOR params (omega/tol/max_iter) are required for early exercise"
            )
    if method in (PDEMethod.EXPLICIT, PDEMethod.EXPLICIT_HULL) and (
        american_solver is PDEEarlyExercise.GAUSS_SEIDEL
    ):
        raise UnsupportedFeatureError("GAUSS_SEIDEL is not supported with explicit time stepping")


def _check_explicit_spot_stability(
    *,
    tau_grid: np.ndarray,
    volatility: float,
    smax: float,
    dS: float,
    time_to_maturity: float,
    discount_curve: DiscountCurve,
    dividend_curve: DiscountCurve | None,
) -> None:
    """Check CFL-type stability conditions for the explicit spot-grid scheme.

    Raises StabilityError if the drift or time-step size violates the
    stability bound for the forward-Euler discretisation on a uniform
    spot grid.
    """
    t_prev = time_to_maturity - tau_grid[:-1]
    t_next = time_to_maturity - tau_grid[1:]
    r_steps = np.array(
        [discount_curve.forward_rate(t1, t0) for t0, t1 in zip(t_prev, t_next)],
        dtype=float,
    )
    if dividend_curve is not None:
        q_steps = np.array(
            [dividend_curve.forward_rate(t1, t0) for t0, t1 in zip(t_prev, t_next)],
            dtype=float,
        )
    else:
        q_steps = np.zeros_like(r_steps)

    drift = float(np.max(np.abs(r_steps - q_steps)))
    if drift > volatility**2:
        raise StabilityError(
            "Explicit spot-grid scheme unstable: max |r-q| must be <= sigma^2. "
            "Use log-spot or implicit/CN."
        )

    denom = (volatility**2) * (smax**2) / (dS**2) + max(float(np.max(r_steps)), 0.0)
    if denom > 0.0:
        dt_max = 1.0 / denom
        max_dt = float(np.max(np.diff(tau_grid)))
        if max_dt > dt_max:
            min_steps = int(math.ceil(time_to_maturity / dt_max))
            raise StabilityError(
                "Explicit spot-grid scheme unstable: time step too large. "
                f"max_dt={max_dt:.4g} exceeds dt_max={dt_max:.4g}. "
                f"Increase time_steps to >= {min_steps} or use log-spot/implicit/CN."
            )


def _build_time_step_schedule(
    tau_grid: np.ndarray,
    method: PDEMethod,
    rannacher_steps: int,
) -> list[tuple[float, float, PDEMethod]]:
    """Build the time-step schedule from the tau grid.

    For Crank-Nicolson with Rannacher smoothing (Pooley-Vetzal-Forsyth 2003),
    the first *rannacher_steps* intervals are each replaced by two implicit
    (backward Euler) half-steps of size d_tau/2.  This damps payoff
    non-smoothness while preserving the overall time-grid structure.
    For all other methods the schedule is a straightforward pass-through.
    """
    steps: list[tuple[float, float, PDEMethod]] = []
    for n in range(1, tau_grid.size):
        tau_start = float(tau_grid[n - 1])
        tau_end = float(tau_grid[n])
        if method is PDEMethod.CRANK_NICOLSON and rannacher_steps > 0 and n <= rannacher_steps:
            tau_mid = 0.5 * (tau_start + tau_end)
            steps.append((tau_start, tau_mid, PDEMethod.IMPLICIT))
            steps.append((tau_mid, tau_end, PDEMethod.IMPLICIT))
        else:
            steps.append((tau_start, tau_end, method))
    return steps


def _vanilla_fd_core(
    *,
    spot: float,
    strike: float,
    time_to_maturity: float,
    volatility: float,
    discount_curve: DiscountCurve,
    dividend_curve: DiscountCurve | None,
    dividend_schedule: list[tuple[float, float]] | None,
    option_type: OptionType,
    smax_mult: float,
    spot_steps: int,
    time_steps: int,
    early_exercise: bool,
    method: PDEMethod,
    rannacher_steps: int,
    space_grid: PDESpaceGrid,
    american_solver: PDEEarlyExercise,
    omega: float | None = None,
    tol: float | None = None,
    max_iter: int | None = None,
) -> tuple[float, np.ndarray, np.ndarray]:
    _validate_fd_inputs(
        option_type=option_type,
        time_to_maturity=time_to_maturity,
        spot_steps=spot_steps,
        time_steps=time_steps,
        volatility=volatility,
        discount_curve=discount_curve,
        early_exercise=early_exercise,
        method=method,
        american_solver=american_solver,
        omega=omega,
        tol=tol,
        max_iter=max_iter,
    )

    smax = float(smax_mult * max(spot, strike))
    if space_grid is PDESpaceGrid.SPOT:
        grid = np.linspace(0.0, smax, spot_steps + 1)
        S = grid
        dS = smax / spot_steps
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

    if method in (PDEMethod.EXPLICIT, PDEMethod.EXPLICIT_HULL) and space_grid is PDESpaceGrid.SPOT:
        _check_explicit_spot_stability(
            tau_grid=tau_grid,
            volatility=volatility,
            smax=smax,
            dS=dS,
            time_to_maturity=time_to_maturity,
            discount_curve=discount_curve,
            dividend_curve=dividend_curve,
        )

    # March forward in tau: 0 -> T (equivalently backward in calendar time)
    df_0T = float(discount_curve.df(time_to_maturity))  # P(0,T)
    if dividend_curve is not None:
        dq_0T = float(dividend_curve.df(time_to_maturity))  # Dq(0,T)
    else:
        dq_0T = None

    psor_steps = 0
    psor_total_iters = 0
    psor_max_iters = 0
    psor_not_converged = 0

    steps = _build_time_step_schedule(tau_grid, method, rannacher_steps)

    for tau_prev, tau_curr, method_used in steps:
        d_tau = tau_curr - tau_prev
        t_prev = time_to_maturity - tau_prev
        t_curr = time_to_maturity - tau_curr

        r = float(discount_curve.forward_rate(t_curr, t_prev))
        if dividend_curve is not None:
            q = float(dividend_curve.forward_rate(t_curr, t_prev))
        else:
            q = 0.0

        hull_discounting = method_used is PDEMethod.EXPLICIT_HULL

        if space_grid is PDESpaceGrid.SPOT:
            gamma, beta, alpha = _spot_operator_coeffs(
                spot_values=S[1:-1],
                dS=dS,
                risk_free_rate=r,
                dividend_rate=q,
                volatility=volatility,
                hull_discounting=hull_discounting,
            )
        else:
            gamma, beta, alpha = _log_operator_coeffs(  # type: ignore[assignment]
                dz=dz,
                risk_free_rate=r,
                dividend_rate=q,
                volatility=volatility,
                hull_discounting=hull_discounting,
            )

        df_0t = float(discount_curve.df(t_curr))
        df_tT: float = df_0T / df_0t
        if dividend_curve is not None:
            dq_0t = float(dividend_curve.df(t_curr))
            dq_tT: float = dq_0T / dq_0t  # type: ignore[operator]
        else:
            dq_tT = float(np.exp(-q * tau_curr))

        left, right = _boundary_values(
            option_type=option_type,
            strike=strike,
            smax=float(S[-1]),
            df_tT=df_tT,
            dq_tT=dq_tT,
            early_exercise=early_exercise,
        )

        V_old = V.copy()

        a, b, c = _scaled_operator_coeffs(gamma=gamma, beta=beta, alpha=alpha, d_tau=d_tau)

        intrinsic_for_step = intrinsic if early_exercise else None

        if method_used in (PDEMethod.EXPLICIT, PDEMethod.EXPLICIT_HULL):
            V = _explicit_step(
                V_old,
                j,
                a,
                b,
                c,
                left,
                right,
                intrinsic_for_step,
                r_dt=r * d_tau if hull_discounting else 0.0,
            )
        else:
            V, psor_iters = _implicit_cn_step(
                V_old,
                V,
                j,
                a,
                b,
                c,
                left,
                right,
                method_used,
                spot_steps,
                intrinsic_for_step,
                american_solver,
                omega,
                tol,
                max_iter,
            )
            if psor_iters is not None:
                psor_steps += 1
                psor_total_iters += psor_iters
                psor_max_iters = max(psor_max_iters, psor_iters)
                if psor_iters == int(max_iter):
                    psor_not_converged += 1

        # Apply discrete dividend jump at tau if needed
        if dividend_map:
            amount = dividend_map.get(round(tau_curr, 12))
            if amount is not None:
                _apply_dividend_jump(V, grid, amount, space_grid=space_grid)
                if early_exercise:
                    V[:] = np.maximum(V, intrinsic)

    if psor_steps > 0:
        avg_iters = psor_total_iters / psor_steps
        logger.debug(
            "PDE PSOR steps=%d avg_iters=%.2f max_iters=%d not_converged=%d",
            psor_steps,
            avg_iters,
            psor_max_iters,
            psor_not_converged,
        )

    price = np.interp(spot, S, V)
    return price, S, V


def _european_vanilla_fd(
    *,
    spot: float,
    strike: float,
    time_to_maturity: float,
    volatility: float,
    discount_curve: DiscountCurve,
    dividend_curve: DiscountCurve | None,
    dividend_schedule: list[tuple[float, float]] | None,
    option_type: OptionType,
    smax_mult: float,
    spot_steps: int,
    time_steps: int,
    method: PDEMethod,
    rannacher_steps: int,
    space_grid: PDESpaceGrid,
) -> tuple[float, np.ndarray, np.ndarray]:
    return _vanilla_fd_core(
        spot=spot,
        strike=strike,
        time_to_maturity=time_to_maturity,
        volatility=volatility,
        discount_curve=discount_curve,
        dividend_curve=dividend_curve,
        dividend_schedule=dividend_schedule,
        option_type=option_type,
        smax_mult=smax_mult,
        spot_steps=spot_steps,
        time_steps=time_steps,
        early_exercise=False,
        method=method,
        rannacher_steps=rannacher_steps,
        space_grid=space_grid,
        american_solver=PDEEarlyExercise.INTRINSIC,
    )


def _american_vanilla_fd(
    *,
    spot: float,
    strike: float,
    time_to_maturity: float,
    volatility: float,
    discount_curve: DiscountCurve,
    dividend_curve: DiscountCurve | None,
    dividend_schedule: list[tuple[float, float]] | None,
    option_type: OptionType,
    smax_mult: float,
    spot_steps: int,
    time_steps: int,
    method: PDEMethod,
    rannacher_steps: int,
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
        volatility=volatility,
        discount_curve=discount_curve,
        dividend_curve=dividend_curve,
        dividend_schedule=dividend_schedule,
        option_type=option_type,
        smax_mult=smax_mult,
        spot_steps=spot_steps,
        time_steps=time_steps,
        early_exercise=True,
        method=method,
        rannacher_steps=rannacher_steps,
        space_grid=space_grid,
        american_solver=american_solver,
        omega=omega,
        tol=tol,
        max_iter=max_iter,
    )


class _FDEuropeanValuation:
    """European option valuation using PDE finite differences."""

    def __init__(self, parent: "OptionValuation") -> None:
        self.parent = parent

    def solve(self) -> tuple[float, np.ndarray, np.ndarray]:
        """Compute the full FD solution on the spot grid at pricing time."""
        pv, S, V = self._solve()
        return pv, S, V

    def _solve(self) -> tuple[float, np.ndarray, np.ndarray]:
        params = self.parent.params
        if not isinstance(params, PDEParams):
            raise ConfigurationError("PDE valuation requires PDEParams on OptionValuation")
        logger.debug(
            "PDE European method=%s grid=%s spot_steps=%d time_steps=%d",
            params.method.value,
            params.space_grid.value,
            params.spot_steps,
            params.time_steps,
        )
        spot = float(self.parent.underlying.initial_value)
        strike = self.parent.strike

        volatility = float(self.parent.underlying.volatility)
        discount_curve = self.parent.discount_curve
        dividend_curve = self.parent.underlying.dividend_curve
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
            volatility=volatility,
            discount_curve=discount_curve,
            dividend_curve=dividend_curve,
            dividend_schedule=dividend_schedule,
            option_type=self.parent.option_type,
            smax_mult=smax_mult,
            spot_steps=spot_steps,
            time_steps=time_steps,
            method=params.method,
            rannacher_steps=int(params.rannacher_steps),
            space_grid=params.space_grid,
        )

    def present_value(self) -> float:
        params = self.parent.params
        if not isinstance(params, PDEParams):
            raise ConfigurationError("PDE valuation requires PDEParams on OptionValuation")
        with log_timing(logger, "PDE European present_value", params.log_timings):
            pv, *_ = self._solve()
        return float(pv)


class _FDAmericanValuation:
    """American option valuation using PDE finite differences."""

    def __init__(self, parent: OptionValuation) -> None:
        self.parent = parent

    def solve(self) -> tuple[float, np.ndarray, np.ndarray]:
        """Compute the full FD solution on the spot grid at pricing time."""
        pv, S, V = self._solve()
        return pv, S, V

    def _solve(self) -> tuple[float, np.ndarray, np.ndarray]:
        params = self.parent.params
        if not isinstance(params, PDEParams):
            raise ConfigurationError("PDE valuation requires PDEParams on OptionValuation")
        logger.debug(
            "PDE American method=%s grid=%s solver=%s spot_steps=%d time_steps=%d",
            params.method.value,
            params.space_grid.value,
            params.american_solver.value,
            params.spot_steps,
            params.time_steps,
        )
        spot = float(self.parent.underlying.initial_value)
        strike = self.parent.strike

        volatility = float(self.parent.underlying.volatility)
        discount_curve = self.parent.discount_curve
        dividend_curve = self.parent.underlying.dividend_curve
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
            volatility=volatility,
            discount_curve=discount_curve,
            dividend_curve=dividend_curve,
            dividend_schedule=dividend_schedule,
            option_type=self.parent.option_type,
            smax_mult=smax_mult,
            spot_steps=spot_steps,
            time_steps=time_steps,
            method=params.method,
            rannacher_steps=int(params.rannacher_steps),
            space_grid=params.space_grid,
            american_solver=params.american_solver,
            omega=omega,
            tol=tol,
            max_iter=max_iter,
        )

    def present_value(self) -> float:
        params = self.parent.params
        if not isinstance(params, PDEParams):
            raise ConfigurationError("PDE valuation requires PDEParams on OptionValuation")
        with log_timing(logger, "PDE American present_value", params.log_timings):
            pv, *_ = self._solve()
        return float(pv)
