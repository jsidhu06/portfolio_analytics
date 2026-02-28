"""Monte Carlo Simulation option valuation implementations."""

from __future__ import annotations
from typing import TYPE_CHECKING
import logging
import numpy as np

from ..utils import calculate_year_fraction, log_timing
from ..stochastic_processes import PathSimulation
from ..enums import OptionType, AsianAveraging
from ..exceptions import ConfigurationError, NumericalError, ValidationError
from .params import MonteCarloParams


if TYPE_CHECKING:
    from .core import OptionValuation, AsianOptionSpec


logger = logging.getLogger(__name__)


def _warn_if_high_std_error(
    *,
    pv_pathwise: np.ndarray,
    pv_mean: float,
    params: MonteCarloParams,
    label: str,
) -> None:
    """Emit a warning log if MC standard error is high relative to the PV estimate."""
    if params.std_error_warn_ratio is None:
        return
    n_paths = pv_pathwise.size
    if n_paths < 2:
        return
    std_error = float(np.std(pv_pathwise, ddof=1) / np.sqrt(n_paths))
    scale = max(abs(pv_mean), 1.0e-12)
    ratio = std_error / scale
    logger.debug(
        "MC %s std_error=%.6g ratio=%.6g paths=%d",
        label,
        std_error,
        ratio,
        n_paths,
    )
    if ratio > params.std_error_warn_ratio:
        logger.warning(
            "MC %s standard error high: std_error=%.6g ratio=%.6g (>%.3g) paths=%d",
            label,
            std_error,
            ratio,
            params.std_error_warn_ratio,
            n_paths,
        )


def _resolve_time_index(time_grid: np.ndarray, target, label: str) -> int:
    """Return the index of the entry in *time_grid* closest to *target*.

    Uses ``np.searchsorted`` on year-fraction offsets so that minor
    datetime rounding (e.g. microseconds from ``pd.date_range``) does not
    cause a lookup failure.  Raises ``ValidationError`` if the nearest
    grid point is more than ~1 second away.
    """
    if len(time_grid) == 0:
        raise ValidationError(f"{label}: time_grid is empty.")

    # Fast path: exact object match (covers the common case)
    exact = np.where(time_grid == target)[0]
    if exact.size > 0:
        return int(exact[0])

    # Tolerance-based fallback: compare as year fractions from grid[0]
    ref = time_grid[0]
    target_yf = calculate_year_fraction(ref, target)
    grid_yf = np.array([calculate_year_fraction(ref, t) for t in time_grid], dtype=float)
    idx = int(np.argmin(np.abs(grid_yf - target_yf)))
    # ~1 second tolerance expressed as year fraction
    if abs(grid_yf[idx] - target_yf) > 1.0 / (365.25 * 86400):
        raise ValidationError(f"{label} not found in underlying time_grid.")
    return idx


def _vanilla_payoff(option_type: OptionType, strike: float, spot: np.ndarray) -> np.ndarray:
    """Vectorized vanilla payoff: max(S-K,0) for calls, max(K-S,0) for puts."""
    if option_type is OptionType.CALL:
        return np.maximum(spot - strike, 0.0)
    return np.maximum(strike - spot, 0.0)


# ---------------------------------------------------------------------------
# Laguerre basis + ridge regression for Longstaff-Schwartz
# ---------------------------------------------------------------------------


def _laguerre_basis(x: np.ndarray, deg: int) -> np.ndarray:
    """Build a Laguerre polynomial design matrix of shape ``(n, deg+1)``.

    Uses the standard (physicists') Laguerre recurrence::

        L_0(x) = 1
        L_1(x) = 1 - x
        L_{k+1}(x) = ((2k + 1 - x) L_k(x) - k L_{k-1}(x)) / (k + 1)

    These form an orthogonal basis on [0, inf) w.r.t. e^{-x}, which
    provides far better conditioning than raw power polynomials in spot.
    """
    x = np.asarray(x, dtype=float)
    cols: list[np.ndarray] = [np.ones_like(x)]
    if deg >= 1:
        cols.append(1.0 - x)
    for k in range(1, deg):
        cols.append(((2 * k + 1 - x) * cols[k] - k * cols[k - 1]) / (k + 1))
    return np.column_stack(cols)


def _ridge_lsm_continuation(
    S_t: np.ndarray,
    Y: np.ndarray,
    itm: np.ndarray,
    strike: float | None,
    deg: int,
    ridge_lambda: float,
    min_itm: int,
) -> np.ndarray:
    """Robust continuation-value estimate for Longstaff-Schwartz.

    Regresses discounted next-step values onto a Laguerre polynomial basis
    in moneyness ``S/K`` using ridge (Tikhonov) regularisation.

    When too few ITM paths are available for a stable regression the
    continuation value falls back to the discounted next-step value
    (path-wise), which is the conservative "do no harm" default.

    Parameters
    ----------
    S_t : np.ndarray
        Spot prices at this time step for all paths.
    Y : np.ndarray
        Discounted next-step values for all paths.
    itm : np.ndarray
        Boolean mask indicating in-the-money paths.
    strike : float | None
        Option strike price (used to compute moneyness ``S/K``).
        When ``None`` (custom payoff), the mean ITM spot is used as the
        normaliser so the Laguerre basis inputs remain well-scaled.
    deg : int
        Laguerre polynomial degree.
    ridge_lambda : float
        Ridge regularisation parameter (>= 0).
    min_itm : int
        Minimum ITM paths required for regression; fewer triggers fallback.

    Returns
    -------
    np.ndarray
        Estimated continuation value for every path (zero for OTM paths).
    """
    cont = np.zeros_like(S_t, dtype=float)
    if not np.any(itm):
        return cont

    S_itm = S_t[itm]
    Y_itm = Y[itm]
    n = S_itm.size

    # Too few ITM points for a stable fit — use cross-sectional mean
    # (degree-0 regression) to avoid path-wise foresight bias.
    if n < max(min_itm, deg + 1):
        cont[itm] = np.mean(Y_itm)
        return cont

    normaliser = strike if strike is not None else max(float(np.mean(S_itm)), 1e-12)
    x = S_itm / normaliser  # moneyness, always positive
    X = _laguerre_basis(x, deg=deg)
    p = X.shape[1]

    # Ridge solve:  beta = (X^T X + lambda I)^{-1} X^T y
    XtX = X.T @ X
    Xty = X.T @ Y_itm
    beta = np.linalg.solve(XtX + ridge_lambda * np.eye(p), Xty)
    cont[itm] = X @ beta
    return cont


def _year_fractions(pricing_date, dates: np.ndarray) -> np.ndarray:
    return np.array(
        [calculate_year_fraction(pricing_date, d) for d in dates],
        dtype=float,
    )


class _MCEuropeanValuation:
    """Implementation of European option valuation using Monte Carlo."""

    def __init__(self, parent: OptionValuation) -> None:
        self.parent = parent
        if not isinstance(parent.params, MonteCarloParams):
            raise ConfigurationError(
                "Monte Carlo valuation requires MonteCarloParams on OptionValuation"
            )
        self.mc_params: MonteCarloParams = parent.params
        if not isinstance(parent.underlying, PathSimulation):
            raise ConfigurationError(
                "Monte Carlo valuation requires a PathSimulation underlying on OptionValuation"
            )
        self.underlying = parent.underlying

    def solve(self) -> np.ndarray:
        """Generate undiscounted payoff vector at maturity (one value per path)."""
        paths = self.underlying.simulate(random_seed=self.mc_params.random_seed)
        time_grid = self.underlying.time_grid

        time_index_end = _resolve_time_index(time_grid, self.parent.maturity, "maturity")
        maturity_value = paths[time_index_end]

        K = self.parent.strike
        if self.parent.option_type in (OptionType.CALL, OptionType.PUT):
            return _vanilla_payoff(self.parent.option_type, K, maturity_value)

        payoff_fn = getattr(self.parent.spec, "payoff", None)
        if payoff_fn is None:
            raise ValidationError("Unsupported option type for Monte Carlo valuation.")
        return payoff_fn(maturity_value)

    def present_value(self) -> float:
        """Return the scalar present value."""
        with log_timing(logger, "MC European present_value", self.mc_params.log_timings):
            pv_pathwise = self.present_value_pathwise()
            pv = float(np.mean(pv_pathwise))
        logger.debug(
            "MC European paths=%d time_steps=%d",
            pv_pathwise.size,
            len(self.underlying.time_grid) - 1,
        )
        _warn_if_high_std_error(
            pv_pathwise=pv_pathwise,
            pv_mean=pv,
            params=self.mc_params,
            label="European",
        )
        return pv

    def present_value_pathwise(self) -> np.ndarray:
        """Return discounted present values for each path."""
        payoff_vector = self.solve()
        ttm = float(calculate_year_fraction(self.parent.pricing_date, self.parent.maturity))
        discount_factor = float(self.parent.discount_curve.df(ttm))
        return discount_factor * payoff_vector

    # ------------------------------------------------------------------
    # MC Greeks — pathwise and likelihood-ratio estimators
    # ------------------------------------------------------------------

    def _simulate_terminal(self) -> tuple[np.ndarray, int, float, float]:
        """Simulate paths and return terminal prices with key scalars.

        Returns
        -------
        (ST, idx, ttm, df) — terminal spot array, time-grid index of
        maturity, time to maturity in years, and risk-free discount factor.
        """
        paths = self.underlying.simulate(random_seed=self.mc_params.random_seed)
        time_grid = self.underlying.time_grid
        idx = _resolve_time_index(time_grid, self.parent.maturity, "maturity")
        ST: np.ndarray = paths[idx]
        ttm = float(calculate_year_fraction(self.parent.pricing_date, self.parent.maturity))
        df = float(self.parent.discount_curve.df(ttm))
        return ST, idx, ttm, df

    def _effective_terminal_z(self, idx: int, ttm: float) -> np.ndarray:
        r"""Compute the effective terminal standard normal from cached per-step normals.

        .. math::
            Z_{\text{eff}}
            = \frac{\sum_i \sqrt{\Delta t_i}\,Z_i}{\sqrt{T}}

        Must be called **after** :meth:`_simulate_terminal` (which populates
        :pyattr:`PathSimulation.last_normals`).
        """
        z_all = self.underlying.last_normals  # (num_steps, num_paths)
        if z_all is None:
            raise ValidationError(
                "You must run PathSimulation.simulate() before calling likelihood-ratio Greeks."
            )
        time_deltas = self.underlying._time_deltas()  # (num_steps,)
        sqrt_dt = np.sqrt(time_deltas[:idx])

        if not np.isclose(time_deltas[:idx].sum(), ttm):
            raise NumericalError(
                f"Time deltas sum {time_deltas[:idx].sum()} does not match time to maturity {ttm}"
            )
        return sqrt_dt @ z_all[:idx] / np.sqrt(ttm)

    # --- pathwise Greeks --------------------------------------------------

    def delta_pathwise(self) -> float:
        r"""Pathwise (IPA) delta estimator.

        Call: :math:`\\Delta = e^{-rT}\\,\\mathbb{E}[\\mathbf{1}_{S_T>K}\\,S_T/S_0]`

        Put:  :math:`\\Delta = -e^{-rT}\\,\\mathbb{E}[\\mathbf{1}_{S_T<K}\\,S_T/S_0]`
        """
        ST, _idx, _ttm, df = self._simulate_terminal()
        S0 = float(self.underlying.initial_value)
        K = self.parent.strike
        if self.parent.option_type is OptionType.CALL:
            return float(np.mean(df * (ST > K) * (ST / S0)))
        return float(np.mean(-df * (ST < K) * (ST / S0)))

    def vega_pathwise(self) -> float:
        r"""Pathwise (IPA) vega estimator (per 1 pp change in vol).

        Uses :math:`\\partial S_T/\\partial\\sigma = S_T(-\\sigma T + \\sqrt{T}\\,Z)`.

        Call: :math:`\\nu = e^{-rT}\\,\\mathbb{E}[\\mathbf{1}_{S_T>K}\\,S_T(-\\sigma T+\\sqrt{T}Z)]`

        Put:  :math:`\\nu = -e^{-rT}\\,\\mathbb{E}[\\mathbf{1}_{S_T<K}\\,S_T(-\\sigma T+\\sqrt{T}Z)]`
        """
        ST, idx, ttm, df = self._simulate_terminal()
        Z = self._effective_terminal_z(idx, ttm)
        sigma = float(self.underlying.volatility)
        K = self.parent.strike
        dST_dsigma = ST * (-sigma * ttm + np.sqrt(ttm) * Z)
        if self.parent.option_type is OptionType.CALL:
            return float(np.mean(df * (ST > K) * dST_dsigma)) / 100
        return float(np.mean(-df * (ST < K) * dST_dsigma)) / 100

    # --- likelihood-ratio Greeks ------------------------------------------

    def delta_lr(self) -> float:
        r"""Likelihood-ratio (score-function) delta estimator.

        :math:`\\Delta = e^{-rT}\\,\\mathbb{E}\\!\\left[\\Phi(S_T)\\,
        \\frac{Z}{\\sigma\\sqrt{T}\\,S_0}\\right]`
        """
        ST, idx, ttm, df = self._simulate_terminal()
        Z = self._effective_terminal_z(idx, ttm)
        S0 = float(self.underlying.initial_value)
        sigma = float(self.underlying.volatility)
        K = self.parent.strike
        payoff = _vanilla_payoff(self.parent.option_type, K, ST)
        return float(np.mean(df * payoff * Z / (sigma * np.sqrt(ttm) * S0)))

    def vega_lr(self) -> float:
        r"""Likelihood-ratio (score-function) vega estimator (per 1 pp).

        The score of the lognormal density w.r.t. :math:`\\sigma` is

        .. math::
            \\frac{\\partial\\ln p}{\\partial\\sigma}
            = \\frac{Z^2 - 1}{\\sigma} \\;-\\; Z\\sqrt{T}

        The second term arises because the risk-neutral drift
        :math:`\\mu = r - q - \\tfrac12\\sigma^2` depends on :math:`\\sigma`
        via the Itô correction.
        """
        ST, idx, ttm, df = self._simulate_terminal()
        Z = self._effective_terminal_z(idx, ttm)
        sigma = float(self.underlying.volatility)
        K = self.parent.strike
        payoff = _vanilla_payoff(self.parent.option_type, K, ST)
        score = (Z**2 - 1) / sigma - Z * np.sqrt(ttm)
        return float(np.mean(df * payoff * score)) / 100

    # --- finite-difference of pathwise delta ------------------------------

    def gamma_pathwise_fd(self, epsilon: float | None = None) -> float:
        r"""Gamma via central difference of pathwise deltas on one simulation.

        Under GBM, :math:`S_T` scales linearly with :math:`S_0`, so bumped
        terminal values are obtained analytically from a single set of paths:

        .. math::
            S_T^{\pm} = S_T \cdot (S_0 \pm h) / S_0

        The pathwise delta at each bumped spot is computed on these
        rescaled terminals (same random draws), and gamma follows from
        the standard central-difference formula:

        .. math::
            \Gamma \approx
            \frac{\Delta_{\text{pw}}(S_0+h) - \Delta_{\text{pw}}(S_0-h)}{2h}
        """
        ST, _idx, _ttm, df = self._simulate_terminal()
        S0 = float(self.underlying.initial_value)
        K = self.parent.strike
        is_call = self.parent.option_type is OptionType.CALL

        if epsilon is None:
            epsilon = S0 / 100  # 1 % of spot

        S0_up = S0 + epsilon
        S0_dn = S0 - epsilon
        ST_up = ST * (S0_up / S0)
        ST_dn = ST * (S0_dn / S0)

        if is_call:
            delta_up = float(np.mean(df * (ST_up > K) * (ST_up / S0_up)))
            delta_dn = float(np.mean(df * (ST_dn > K) * (ST_dn / S0_dn)))
        else:
            delta_up = float(np.mean(-df * (ST_up < K) * (ST_up / S0_up)))
            delta_dn = float(np.mean(-df * (ST_dn < K) * (ST_dn / S0_dn)))

        return (delta_up - delta_dn) / (2 * epsilon)


class _MCAmericanValuation:
    """Implementation of American option valuation using Longstaff-Schwartz Monte Carlo."""

    def __init__(self, parent: OptionValuation) -> None:
        self.parent = parent
        if not isinstance(parent.params, MonteCarloParams):
            raise ConfigurationError(
                "Monte Carlo valuation requires MonteCarloParams on OptionValuation"
            )
        self.mc_params: MonteCarloParams = parent.params
        if not isinstance(parent.underlying, PathSimulation):
            raise ConfigurationError(
                "Monte Carlo valuation requires a PathSimulation underlying on OptionValuation"
            )
        self.underlying = parent.underlying

    def solve(self) -> tuple[np.ndarray, np.ndarray, int, int]:
        """Generate underlying paths and intrinsic payoff matrix over time.

        Returns
        =======
        tuple of (spot_paths, payoff, time_index_start, time_index_end)
        """
        paths = self.underlying.simulate(random_seed=self.mc_params.random_seed)
        time_grid = self.underlying.time_grid
        time_index_start = _resolve_time_index(time_grid, self.parent.pricing_date, "Pricing date")
        time_index_end = _resolve_time_index(time_grid, self.parent.maturity, "maturity")

        spot_paths = paths[time_index_start : time_index_end + 1]

        K = self.parent.strike
        if self.parent.option_type in (OptionType.CALL, OptionType.PUT):
            payoff = _vanilla_payoff(self.parent.option_type, K, spot_paths)
        else:
            payoff_fn = self.parent.spec.payoff  # type: ignore[union-attr]
            payoff = payoff_fn(spot_paths)

        return spot_paths, payoff, time_index_start, time_index_end

    def present_value(self) -> float:
        """Calculate PV using Longstaff-Schwartz regression method."""
        with log_timing(logger, "MC American present_value", self.mc_params.log_timings):
            pv_pathwise = self.present_value_pathwise()
            pv = float(np.mean(pv_pathwise))
        logger.debug(
            "MC American paths=%d time_steps=%d deg=%d",
            pv_pathwise.size,
            len(self.underlying.time_grid) - 1,
            self.mc_params.deg,
        )
        _warn_if_high_std_error(
            pv_pathwise=pv_pathwise,
            pv_mean=pv,
            params=self.mc_params,
            label="American",
        )
        return pv

    def present_value_pathwise(self) -> np.ndarray:
        """Return discounted present values for each path (LSM output at pricing date)."""
        spot_paths, intrinsic_values, time_index_start, time_index_end = self.solve()
        time_list = self.underlying.time_grid[time_index_start : time_index_end + 1]
        t_grid = _year_fractions(self.parent.pricing_date, time_list)
        discount_factors = self.parent.discount_curve.df(t_grid)
        values = np.zeros_like(intrinsic_values)
        values[-1] = intrinsic_values[-1]

        for t in range(len(time_list) - 2, 0, -1):
            df_step = discount_factors[t + 1] / discount_factors[t]
            itm = intrinsic_values[t] > 0

            continuation = _ridge_lsm_continuation(
                S_t=spot_paths[t],
                Y=df_step * values[t + 1],
                itm=itm,
                strike=self.parent.strike,
                deg=self.mc_params.deg,
                ridge_lambda=self.mc_params.ridge_lambda,
                min_itm=self.mc_params.min_itm,
            )

            values[t] = np.where(
                intrinsic_values[t] > continuation,
                intrinsic_values[t],
                df_step * values[t + 1],
            )

        df0 = discount_factors[1] / discount_factors[0]
        return df0 * values[1]


class _MCAsianValuation:
    """Implementation of Asian option valuation using Monte Carlo.

    Asian options are path-dependent options where the payoff depends on the average
    price of the underlying over the averaging period.
    """

    def __init__(self, parent: OptionValuation) -> None:
        self.parent = parent
        if not isinstance(parent.params, MonteCarloParams):
            raise ConfigurationError(
                "Monte Carlo valuation requires MonteCarloParams on OptionValuation"
            )
        self.mc_params: MonteCarloParams = parent.params
        if not isinstance(parent.underlying, PathSimulation):
            raise ConfigurationError(
                "Monte Carlo valuation requires a PathSimulation underlying on OptionValuation"
            )
        self.underlying = parent.underlying
        self.spec: AsianOptionSpec = parent.spec  # type: ignore[assignment]

    def solve(self) -> np.ndarray:
        """Generate undiscounted payoff vector based on path averages.

        Returns
        -------
        np.ndarray
            Payoff for each path based on the average spot price
        """
        paths = self.underlying.simulate(random_seed=self.mc_params.random_seed)
        time_grid = self.underlying.time_grid

        # Determine averaging period
        spec = self.spec
        averaging_start = spec.averaging_start if spec.averaging_start else self.parent.pricing_date

        # Find indices for averaging period
        time_index_start = _resolve_time_index(time_grid, averaging_start, "averaging_start")
        time_index_end = _resolve_time_index(time_grid, self.parent.maturity, "maturity")

        # Extract paths over averaging period (inclusive)
        averaging_paths = paths[time_index_start : time_index_end + 1, :]

        # Calculate average for each path
        if spec.averaging is AsianAveraging.ARITHMETIC:
            # Arithmetic average: (1/N) * Σ S_i
            avg_prices = np.mean(averaging_paths, axis=0)
        elif spec.averaging is AsianAveraging.GEOMETRIC:
            # Geometric average: (Π S_i)^(1/N)
            # Use log space for numerical stability: exp(mean(log(S_i)))
            if np.any(averaging_paths <= 0.0):
                raise NumericalError("Geometric averaging requires strictly positive path prices.")
            log_prices = np.log(averaging_paths)
            avg_prices = np.exp(np.mean(log_prices, axis=0))
        else:
            raise ValidationError(
                f"Unsupported averaging method for Asian valuation: {spec.averaging}"
            )

        # Calculate payoff based on average
        K = self.parent.strike

        # Asian call: max(S_avg - K, 0)
        # Asian put: max(K - S_avg, 0)
        if self.spec.call_put is OptionType.CALL:
            return np.maximum(avg_prices - K, 0.0)
        return np.maximum(K - avg_prices, 0.0)

    def present_value(self) -> float:
        """Return the scalar present value."""
        with log_timing(logger, "MC Asian present_value", self.mc_params.log_timings):
            pv_pathwise = self.present_value_pathwise()
            pv = float(np.mean(pv_pathwise))
        logger.debug(
            "MC Asian paths=%d time_steps=%d",
            pv_pathwise.size,
            len(self.underlying.time_grid) - 1,
        )
        _warn_if_high_std_error(
            pv_pathwise=pv_pathwise,
            pv_mean=pv,
            params=self.mc_params,
            label="Asian",
        )
        return pv

    def present_value_pathwise(self) -> np.ndarray:
        """Return discounted present values for each path."""
        payoff_vector = self.solve()
        ttm = float(calculate_year_fraction(self.parent.pricing_date, self.parent.maturity))
        discount_factor = float(self.parent.discount_curve.df(ttm))
        return discount_factor * payoff_vector


def _running_averages(paths: np.ndarray, averaging: AsianAveraging) -> np.ndarray:
    """Compute the running average at each time step along axis 0.

    Parameters
    ----------
    paths : np.ndarray, shape (T, n_paths)
        Spot prices at each observation date for every simulated path.
    averaging : AsianAveraging
        ARITHMETIC or GEOMETRIC.

    Returns
    -------
    np.ndarray, shape (T, n_paths)
        Running average at each observation date.
    """
    n_steps = paths.shape[0]
    counts = np.arange(1, n_steps + 1, dtype=float).reshape(-1, 1)

    if averaging is AsianAveraging.ARITHMETIC:
        return np.cumsum(paths, axis=0) / counts

    if averaging is AsianAveraging.GEOMETRIC:
        if np.any(paths <= 0.0):
            raise NumericalError("Geometric averaging requires strictly positive path prices.")
        return np.exp(np.cumsum(np.log(paths), axis=0) / counts)

    raise ValidationError(f"Unsupported averaging method: {averaging}")


def _build_2d_design_matrix(s: np.ndarray, a: np.ndarray, deg: int) -> np.ndarray:
    """Build a 2-D polynomial design matrix with cross-terms.

    Includes all monomials :math:`s^i a^j` with :math:`i + j \\le deg`.
    Both inputs are standardised (zero-mean, unit-variance) before
    constructing the monomials for numerical stability.

    Parameters
    ----------
    s, a : np.ndarray, shape (n,)
        Spot prices and running averages for the ITM subset.
    deg : int
        Maximum total polynomial degree.

    Returns
    -------
    np.ndarray, shape (n, k) where k = (deg+1)(deg+2)/2
    """
    s_n = (s - s.mean()) / max(float(s.std()), 1e-12)
    a_n = (a - a.mean()) / max(float(a.std()), 1e-12)

    cols: list[np.ndarray] = []
    for total in range(deg + 1):
        for j in range(total + 1):
            i = total - j
            cols.append(s_n**i * a_n**j)
    return np.column_stack(cols)


class _MCAsianAmericanValuation:
    """American Asian option via Longstaff-Schwartz on the joint (S_t, A_t) state.

    At each exercise opportunity *t* the holder can receive max(A_t − K, 0)
    (call) or max(K − A_t, 0) (put), where A_t is the running average of
    the spot observations from the averaging start up to *t*.

    The continuation value is estimated by OLS regression on a 2-D polynomial
    basis in (S_t, A_t) — the joint Markov state that fully summarises the
    path history relevant to the Asian payoff.

    Both arithmetic and geometric averaging are supported.
    """

    def __init__(self, parent: OptionValuation) -> None:
        self.parent = parent
        if not isinstance(parent.params, MonteCarloParams):
            raise ConfigurationError(
                "Monte Carlo valuation requires MonteCarloParams on OptionValuation"
            )
        self.mc_params: MonteCarloParams = parent.params
        self.underlying: PathSimulation = parent.underlying  # type: ignore[assignment]
        self.spec: AsianOptionSpec = parent.spec  # type: ignore[assignment]

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _asian_payoff(self, avg: np.ndarray) -> np.ndarray:
        """Asian payoff given average prices."""
        K = self.parent.strike
        if self.spec.call_put is OptionType.CALL:
            return np.maximum(avg - K, 0.0)
        return np.maximum(K - avg, 0.0)

    def _get_averaging_data(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Simulate paths and compute running averages & intrinsic values.

        Returns
        -------
        (averaging_paths, running_avg, intrinsic, time_list)
            averaging_paths : (N_obs, n_paths) spot prices at observation dates
            running_avg     : (N_obs, n_paths) running average at each date
            intrinsic       : (N_obs, n_paths) intrinsic payoff at each date
            time_list       : (N_obs,) datetime time grid for the averaging window
        """
        paths = self.underlying.simulate(random_seed=self.mc_params.random_seed)
        time_grid = self.underlying.time_grid
        spec = self.spec
        averaging_start = spec.averaging_start if spec.averaging_start else self.parent.pricing_date

        idx_start = _resolve_time_index(time_grid, averaging_start, "averaging_start")
        idx_end = _resolve_time_index(time_grid, self.parent.maturity, "maturity")

        averaging_paths = paths[idx_start : idx_end + 1]
        running_avg = _running_averages(averaging_paths, spec.averaging)
        intrinsic = self._asian_payoff(running_avg)
        time_list = time_grid[idx_start : idx_end + 1]

        return averaging_paths, running_avg, intrinsic, time_list

    # ------------------------------------------------------------------
    # public interface
    # ------------------------------------------------------------------

    def solve(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate path-level Asian option data.

        Returns
        -------
        (averaging_paths, running_avg, intrinsic_values)
            Each has shape (N_obs, n_paths).
        """
        averaging_paths, running_avg, intrinsic, _ = self._get_averaging_data()
        return averaging_paths, running_avg, intrinsic

    def present_value(self) -> float:
        """Calculate PV using Longstaff-Schwartz with (S, A) regression."""
        with log_timing(logger, "MC Asian American present_value", self.mc_params.log_timings):
            pv_pathwise = self.present_value_pathwise()
            pv = float(np.mean(pv_pathwise))
        logger.debug(
            "MC Asian American paths=%d time_steps=%d deg=%d",
            pv_pathwise.size,
            len(self.underlying.time_grid) - 1,
            self.mc_params.deg,
        )
        _warn_if_high_std_error(
            pv_pathwise=pv_pathwise,
            pv_mean=pv,
            params=self.mc_params,
            label="Asian American",
        )
        return pv

    def present_value_pathwise(self) -> np.ndarray:
        """Discounted PV for each path via LSM on (S_t, A_t)."""
        averaging_paths, running_avg, intrinsic, time_list = self._get_averaging_data()

        t_grid = _year_fractions(self.parent.pricing_date, time_list)
        discount_factors = self.parent.discount_curve.df(t_grid)

        n_obs = averaging_paths.shape[0]
        n_paths = averaging_paths.shape[1]
        deg = self.mc_params.deg

        # Terminal payoff
        values = np.zeros((n_obs, n_paths))
        values[-1] = intrinsic[-1]

        # Backward induction
        for t in range(n_obs - 2, 0, -1):
            df_step = discount_factors[t + 1] / discount_factors[t]
            itm = intrinsic[t] > 0

            continuation = np.zeros(n_paths)
            if np.any(itm):
                S_itm = averaging_paths[t][itm]
                A_itm = running_avg[t][itm]
                V_itm = df_step * values[t + 1][itm]
                n_itm = S_itm.size

                # Fallback: too few ITM points — degree-0 (mean) to
                # avoid path-wise foresight bias.
                if n_itm < max(self.mc_params.min_itm, deg + 1):
                    continuation[itm] = np.mean(V_itm)
                else:
                    # 2-D polynomial regression on (S, A) with ridge
                    X = _build_2d_design_matrix(S_itm, A_itm, deg)
                    p = X.shape[1]
                    XtX = X.T @ X
                    Xty = X.T @ V_itm
                    beta = np.linalg.solve(XtX + self.mc_params.ridge_lambda * np.eye(p), Xty)
                    continuation[itm] = X @ beta

            values[t] = np.where(
                intrinsic[t] > continuation,
                intrinsic[t],
                df_step * values[t + 1],
            )

        df0 = discount_factors[1] / discount_factors[0]
        return df0 * values[1]
