"""Implied volatility solvers for European and American options."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Callable

import numpy as np
from scipy import optimize

from ..enums import (
    DayCountConvention,
    ExerciseType,
    ImpliedVolMethod,
    OptionType,
    PricingMethod,
    GreekCalculationMethod,
)
from ..utils import calculate_year_fraction, pv_discrete_dividends, log_timing
from .core import OptionValuation, UnderlyingPricingData
from .params import BinomialParams


logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ImpliedVolResult:
    """Result container for implied volatility calculation."""

    implied_vol: float
    iterations: int
    converged: bool


def _adjusted_spot_and_dividend_df(valuation: OptionValuation) -> tuple[float, float]:
    spot = float(valuation.underlying.initial_value)
    dividend_curve = valuation.underlying.dividend_curve
    discrete_dividends = valuation.underlying.discrete_dividends
    if not discrete_dividends:
        if dividend_curve is None:
            return spot, 1.0
        time_to_maturity = calculate_year_fraction(
            valuation.pricing_date,
            valuation.maturity,
            day_count_convention=DayCountConvention.ACT_365F,
        )
        return spot, float(dividend_curve.df(time_to_maturity))
    if valuation.discount_curve.flat_rate is None:
        raise NotImplementedError("Discrete dividend adjustments require a flat curve.")

    pv_divs = pv_discrete_dividends(
        discrete_dividends,
        valuation.pricing_date,
        valuation.maturity,
        float(valuation.discount_curve.flat_rate),
    )
    if dividend_curve is None:
        df_q = 1.0
    else:
        time_to_maturity = calculate_year_fraction(
            valuation.pricing_date,
            valuation.maturity,
            day_count_convention=DayCountConvention.ACT_365F,
        )
        df_q = float(dividend_curve.df(time_to_maturity))
    return max(spot - pv_divs, 0.0), df_q


def _price_bounds(valuation: OptionValuation) -> tuple[float, float]:
    time_to_maturity = calculate_year_fraction(
        valuation.pricing_date,
        valuation.maturity,
        day_count_convention=DayCountConvention.ACT_365F,
    )
    if time_to_maturity <= 0:
        raise ValueError("Option maturity must be after pricing date.")

    strike = float(valuation.strike)

    if valuation.exercise_type is ExerciseType.AMERICAN:
        spot = float(valuation.underlying.initial_value)
        if valuation.option_type is OptionType.CALL:
            lower = max(0.0, spot - strike)
            upper = spot
        else:  # PUT
            lower = max(0.0, strike - spot)
            upper = strike
        return lower, upper

    spot, df_q = _adjusted_spot_and_dividend_df(valuation)
    df_r = float(valuation.discount_curve.df(time_to_maturity))

    if valuation.option_type is OptionType.CALL:
        lower = max(0.0, spot * df_q - strike * df_r)
        upper = spot * df_q
    else:  # PUT
        lower = max(0.0, strike * df_r - spot * df_q)
        upper = strike * df_r

    return lower, upper


def _valuation_with_vol(valuation: OptionValuation, vol: float) -> OptionValuation:
    if not isinstance(valuation.underlying, UnderlyingPricingData):
        raise TypeError("Implied volatility requires UnderlyingPricingData (not PathSimulation).")

    bumped_underlying = valuation.underlying.replace(volatility=float(vol))
    return OptionValuation(
        name=valuation.name,
        underlying=bumped_underlying,
        spec=valuation.spec,
        pricing_method=valuation.pricing_method,
        params=valuation.params,
    )


def _non_bsm_initial_guess(
    valuation: OptionValuation,
    target_price: float,
    low: float,
    high: float,
) -> float:
    spot = float(valuation.underlying.initial_value)
    strike = float(valuation.strike)
    time_to_maturity = calculate_year_fraction(
        valuation.pricing_date,
        valuation.maturity,
        day_count_convention=DayCountConvention.ACT_365F,
    )
    time_sqrt = np.sqrt(max(time_to_maturity, 1.0e-8))

    if valuation.option_type is OptionType.CALL:
        intrinsic = max(spot - strike, 0.0)
    else:
        intrinsic = max(strike - spot, 0.0)

    extrinsic = max(target_price - intrinsic, 0.0)
    scale = max(spot, strike, 1.0)
    rough = 0.2 + 0.8 * (extrinsic / scale) / time_sqrt
    return float(np.clip(rough, low, high))


def _bracket_volatility(
    *,
    f: Callable[[float], float],
    low: float,
    high: float,
    max_expansions: int = 6,
) -> tuple[float, float, float, float]:
    f_low = f(low)
    f_high = f(high)

    if f_low > 0:
        for _ in range(max_expansions):
            low = max(low / 2.0, 1.0e-12)
            f_low = f(low)
            if f_low <= 0:
                break

    if f_high < 0:
        for _ in range(max_expansions):
            high *= 2.0
            f_high = f(high)
            if f_high >= 0:
                break

    return low, high, f_low, f_high


def _newton_raphson(
    *,
    f: Callable[[float], float],
    vega: Callable[[float], float],
    low: float,
    high: float,
    initial: float,
    tol: float,
    max_iter: int,
) -> ImpliedVolResult:
    vol = float(initial)
    iterations = 0

    for i in range(max_iter):
        iterations = i + 1
        diff = f(vol)
        if abs(diff) <= tol:
            return ImpliedVolResult(implied_vol=vol, iterations=iterations, converged=True)

        slope = vega(vol)
        if slope <= 0 or not np.isfinite(slope):
            break

        step = diff / slope
        candidate = vol - step

        if not np.isfinite(candidate) or candidate <= low or candidate >= high:
            candidate = 0.5 * (low + high)

        if diff > 0:
            high = min(high, vol)
        else:
            low = max(low, vol)

        if abs(high - low) <= tol:
            return ImpliedVolResult(implied_vol=candidate, iterations=iterations, converged=True)

        vol = candidate

    return ImpliedVolResult(implied_vol=vol, iterations=iterations, converged=False)


def _bisection(
    *,
    f: Callable[[float], float],
    low: float,
    high: float,
    tol: float,
    max_iter: int,
) -> ImpliedVolResult:
    f_low = f(low)
    f_high = f(high)

    if f_low > 0 or f_high < 0:
        raise ValueError("Price not bracketed by vol_bounds; adjust bounds.")

    vol = 0.5 * (low + high)
    for i in range(max_iter):
        f_mid = f(vol)
        if abs(f_mid) <= tol or abs(high - low) <= tol:
            return ImpliedVolResult(implied_vol=vol, iterations=i + 1, converged=True)
        if f_mid > 0:
            high = vol
        else:
            low = vol
        vol = 0.5 * (low + high)

    return ImpliedVolResult(implied_vol=vol, iterations=max_iter, converged=False)


def implied_volatility(
    target_price: float,
    valuation: OptionValuation,
    method: ImpliedVolMethod = ImpliedVolMethod.NEWTON_RAPHSON,
    *,
    initial_vol: float | None = None,
    vol_bounds: tuple[float, float] = (1.0e-6, 5.0),
    tol: float = 1.0e-8,
    max_iter: int = 100,
    log_timings: bool = False,
) -> ImpliedVolResult:
    """Solve for implied volatility using BSM, binomial, or PDE pricing.

    Parameters
    ==========
    target_price:
        Observed option price (per unit, not multiplied by contract size).
    valuation:
        OptionValuation configured for BSM (European) or BINOMIAL/PDE_FD (European/American).
    method:
        Root-finding method to use for the solver.
    initial_vol:
        Initial guess for volatility (annualized, decimal). If None, uses
        valuation.underlying.volatility or 0.2.
    vol_bounds:
        Lower/upper bounds for volatility search.
    tol:
        Absolute tolerance on price difference.
    max_iter:
        Maximum iterations for iterative solvers.

    Notes
    =====
    Vega is expected in per-1% volatility terms. The solver rescales vega to a
    per-1.0 volatility derivative when computing Newton updates.

    Returns
    =======
    ImpliedVolResult
        implied_vol: implied volatility (annualized, decimal)
        iterations: number of iterations performed
        converged: whether solver reached tolerance
    """

    if not isinstance(valuation, OptionValuation):
        raise TypeError("valuation must be an OptionValuation instance")
    if valuation.option_type not in (OptionType.CALL, OptionType.PUT):
        raise NotImplementedError("Implied volatility is only supported for vanilla CALL/PUT.")
    if valuation.pricing_method == PricingMethod.BSM:
        if valuation.exercise_type != ExerciseType.EUROPEAN:
            raise NotImplementedError("BSM implied volatility supports European options only.")
    elif valuation.pricing_method not in (PricingMethod.BINOMIAL, PricingMethod.PDE_FD):
        raise NotImplementedError(
            "Implied volatility supports BSM, BINOMIAL, or PDE_FD pricing methods only."
        )

    if not np.isfinite(target_price):
        raise ValueError("target_price must be finite")
    if target_price < 0:
        raise ValueError("target_price must be non-negative")

    low, high = vol_bounds
    if low <= 0 or high <= 0 or low >= high:
        raise ValueError("vol_bounds must be positive and satisfy low < high")

    if valuation.pricing_method == PricingMethod.BINOMIAL and isinstance(
        valuation.params, BinomialParams
    ):
        time_to_maturity = calculate_year_fraction(
            valuation.pricing_date,
            valuation.maturity,
            day_count_convention=DayCountConvention.ACT_365F,
        )
        dt = time_to_maturity / max(int(valuation.params.num_steps), 1)
        df_r = float(valuation.discount_curve.df(time_to_maturity))
        r = -np.log(df_r) / time_to_maturity
        dividend_curve = valuation.underlying.dividend_curve
        if dividend_curve is None:
            q = 0.0
        else:
            df_q = float(dividend_curve.df(time_to_maturity))
            q = -np.log(df_q) / time_to_maturity
        drift = r - q
        sigma_min = abs(drift) * np.sqrt(max(dt, 1.0e-12))
        low = max(low, sigma_min * 1.01)

    min_price, max_price = _price_bounds(valuation)
    if target_price < min_price - tol or target_price > max_price + tol:
        raise ValueError("target_price is outside no-arbitrage bounds for the provided inputs")

    def price_at(vol: float) -> float:
        return _valuation_with_vol(valuation, vol).present_value()

    def f(vol: float) -> float:
        return price_at(vol) - target_price

    def vega_at(vol: float) -> float:
        val = _valuation_with_vol(valuation, vol)
        greek_method = (
            GreekCalculationMethod.ANALYTICAL
            if valuation.pricing_method == PricingMethod.BSM
            else GreekCalculationMethod.NUMERICAL
        )
        vega = val.vega(greek_calc_method=greek_method)
        return float(vega) * 100.0  # convert from per-1% to per-1.0 volatility

    if initial_vol is not None:
        initial = float(initial_vol)
    elif valuation.pricing_method == PricingMethod.BSM:
        initial = float(valuation.underlying.volatility or 0.2)
    else:
        initial = _non_bsm_initial_guess(valuation, target_price, low, high)

    initial = max(low, min(high, initial))

    low, high, f_low, f_high = _bracket_volatility(f=f, low=low, high=high)
    if f_low > 0 or f_high < 0:
        raise ValueError("Price not bracketed by vol_bounds; adjust bounds.")

    with log_timing(logger, "Implied vol solver", log_timings):
        if method == ImpliedVolMethod.NEWTON_RAPHSON:
            result = _newton_raphson(
                f=f,
                vega=vega_at,
                low=low,
                high=high,
                initial=initial,
                tol=tol,
                max_iter=max_iter,
            )
            if not result.converged:
                result = _bisection(f=f, low=low, high=high, tol=tol, max_iter=max_iter)
        elif method == ImpliedVolMethod.BISECTION:
            result = _bisection(f=f, low=low, high=high, tol=tol, max_iter=max_iter)
        elif method == ImpliedVolMethod.BRENTQ:
            implied = optimize.brentq(f, low, high, xtol=tol, maxiter=max_iter)
            result = ImpliedVolResult(
                implied_vol=float(implied),
                iterations=max_iter,
                converged=True,
            )
        else:
            raise NotImplementedError(
                f"Implied vol method '{method.value}' is not implemented in this solver."
            )

    logger.debug(
        "Implied vol method=%s converged=%s iterations=%d implied_vol=%.6g",
        method.value,
        result.converged,
        result.iterations,
        result.implied_vol,
    )

    return result
