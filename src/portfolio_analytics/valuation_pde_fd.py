"""Finite difference (PDE) valuation implementations.

This module follows the same structure as other valuation modules:
- an internal implementation class that plugs into OptionValuation
- thin convenience wrappers for direct function-style pricing

Current scope
-------------
PDE via finite differences (Crank–Nicolson + PSOR) for vanilla AMERICAN call/put.
"""

from typing import TYPE_CHECKING

import numpy as np

from .enums import OptionType
from .utils import calculate_year_fraction

if TYPE_CHECKING:
    from .valuation import OptionValuation


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

    Solves in the transformed time variable $\tau$ (time remaining):
    - at maturity: $\tau = 0$
    - at pricing date: $\tau = T$
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

    price = float(np.interp(spot, S, V))
    return price, S, V


class _FDAmericanValuation:
    """American option valuation using PDE finite differences (CN + PSOR)."""

    def __init__(self, parent: "OptionValuation"):
        self.parent = parent

    def generate_payoff(self, **kwargs) -> tuple[np.ndarray, np.ndarray]:
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

    def present_value(self, full: bool = False, **kwargs) -> float | tuple[float, np.ndarray]:
        pv, _, V = self._solve(**kwargs)
        if full:
            return pv, V
        return pv
