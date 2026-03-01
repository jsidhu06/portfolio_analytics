"""Path simulation classes for stochastic processes used in pricing."""

from __future__ import annotations

import warnings
from typing import Sequence
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace as dc_replace
import copy
import datetime as dt
import numpy as np
import pandas as pd
from .market_environment import MarketData, CorrelationContext
from .enums import DayCountConvention
from .utils import calculate_year_fraction
from .rates import DiscountCurve
from .exceptions import ConfigurationError, ValidationError


@dataclass(frozen=True, slots=True)
class SimulationConfig:
    """Configuration for a path simulation run.

    Notes
    -----
        - The simulation **start date** is taken from ``MarketData.pricing_date`` (the process pricing date).
            Therefore this config only specifies the horizon **end** (``end_date``) and the discretization.

        Grid specification
        ------------------
        Exactly one of the following modes must be used:

        1) Explicit grid: provide ``time_grid`` (and do not provide ``end_date``, ``frequency``, or ``num_steps``).
        2) Calendar grid: provide ``end_date`` + ``frequency``.
        3) Uniform-step grid: provide ``end_date`` + ``num_steps``.
    """

    paths: int
    frequency: str | None = None
    end_date: dt.datetime | None = None
    num_steps: int | None = None
    day_count_convention: DayCountConvention = DayCountConvention.ACT_365F
    time_grid: np.ndarray | None = None  # optional portfolio override
    observation_dates: set[dt.datetime] = field(default_factory=set)
    grid_start: dt.datetime | None = None
    antithetic: bool = True
    moment_matching: bool = True

    def __post_init__(self) -> None:
        if self.paths is None or int(self.paths) <= 0:
            raise ValidationError("SimulationConfig.paths must be a positive integer")

        if not isinstance(self.day_count_convention, DayCountConvention):
            raise ConfigurationError(
                f"day_count_convention must be a DayCountConvention enum, "
                f"got {type(self.day_count_convention).__name__}"
            )

        has_end_date = self.end_date is not None
        has_time_grid = self.time_grid is not None
        has_frequency = self.frequency is not None
        has_num_steps = self.num_steps is not None

        # Explicit grid mode
        if has_time_grid:
            if has_end_date or has_frequency or has_num_steps:
                raise ValidationError(
                    "When time_grid is provided, end_date, frequency, and num_steps must be omitted."
                )
            if len(self.time_grid) == 0:
                raise ValidationError("SimulationConfig.time_grid must be non-empty when provided")
            return

        # Otherwise we require an end_date and one discretization knob.
        if not has_end_date:
            raise ValidationError(
                "SimulationConfig.end_date must be provided when time_grid is not set"
            )

        if has_frequency == has_num_steps:
            raise ValidationError(
                "SimulationConfig requires exactly one of frequency or num_steps when end_date is provided."
            )

        if has_frequency:
            if not isinstance(self.frequency, str) or not self.frequency.strip():
                raise ValidationError("SimulationConfig.frequency must be a non-empty string")

        if has_num_steps:
            try:
                steps = int(self.num_steps)
            except (TypeError, ValueError) as exc:
                raise ConfigurationError("SimulationConfig.num_steps must be an integer") from exc
            if steps <= 0:
                raise ValidationError("SimulationConfig.num_steps must be a positive integer")

    def _can_rebuild_grid(self) -> bool:
        """True when the config has end_date + a discretisation method."""
        return self.end_date is not None and (
            self.num_steps is not None or self.frequency is not None
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class GBMParams:
    initial_value: float
    volatility: float
    discrete_dividends: Sequence[tuple[dt.datetime, float]] | None = None
    dividend_curve: DiscountCurve | None = None

    def __post_init__(self) -> None:
        if self.initial_value is None:
            raise ValidationError("GBMParams requires initial_value to be not None")
        if self.volatility is None:
            raise ValidationError("GBMParams requires volatility to be not None")
        if not np.isfinite(float(self.initial_value)):
            raise ValidationError("GBMParams requires initial_value to be finite")
        if not np.isfinite(float(self.volatility)):
            raise ValidationError("GBMParams requires volatility to be finite")
        if float(self.volatility) < 0.0:
            raise ValidationError("GBMParams requires volatility to be >= 0")
        object.__setattr__(
            self,
            "discrete_dividends",
            tuple(self.discrete_dividends) if self.discrete_dividends is not None else tuple(),
        )
        if self.dividend_curve is not None and self.discrete_dividends:
            warnings.warn(
                "GBMParams: both dividend_curve and discrete_dividends provided. "
                "The continuous yield will enter the drift and discrete dividends "
                "will be subtracted at each ex-date.",
                stacklevel=2,
            )


@dataclass(frozen=True, slots=True, kw_only=True)
class JDParams:
    initial_value: float
    volatility: float
    lambd: float  # lambda (per year)
    mu: float  # mu_J (mean of log jump size)
    delta: float  # delta_J (std of log jump size)
    discrete_dividends: Sequence[tuple[dt.datetime, float]] | None = None
    dividend_curve: DiscountCurve | None = None

    def __post_init__(self) -> None:
        if self.initial_value is None:
            raise ValidationError("JDParams requires initial_value to be not None")
        if self.volatility is None:
            raise ValidationError("JDParams requires volatility to be not None")
        if not np.isfinite(float(self.initial_value)):
            raise ValidationError("JDParams requires initial_value to be finite")
        if not np.isfinite(float(self.volatility)):
            raise ValidationError("JDParams requires volatility to be finite")
        if float(self.volatility) < 0.0:
            raise ValidationError("JDParams requires volatility to be >= 0")

        if self.lambd is None or self.mu is None or self.delta is None:
            raise ValidationError("JDParams requires lambd, mu, and delta to be not None")

        if not np.isfinite(float(self.lambd)):
            raise ValidationError("JDParams requires lambd to be finite")
        if not np.isfinite(float(self.mu)):
            raise ValidationError("JDParams requires mu to be finite")
        if not np.isfinite(float(self.delta)):
            raise ValidationError("JDParams requires delta to be finite")
        if float(self.lambd) < 0.0:
            raise ValidationError("JDParams requires lambd to be >= 0")
        if float(self.delta) < 0.0:
            raise ValidationError("JDParams requires delta to be >= 0")
        object.__setattr__(
            self,
            "discrete_dividends",
            tuple(self.discrete_dividends) if self.discrete_dividends is not None else tuple(),
        )
        if self.dividend_curve is not None and self.discrete_dividends:
            warnings.warn(
                "JDParams: both dividend_curve and discrete_dividends provided. "
                "The continuous yield will enter the drift and discrete dividends "
                "will be subtracted at each ex-date.",
                stacklevel=2,
            )


@dataclass(frozen=True, slots=True, kw_only=True)
class SRDParams:
    initial_value: float
    volatility: float
    kappa: float  # mean reversion speed
    theta: float  # long-run mean

    def __post_init__(self) -> None:
        if self.initial_value is None:
            raise ValidationError("SRDParams requires initial_value to be not None")
        if self.volatility is None:
            raise ValidationError("SRDParams requires volatility to be not None")
        if not np.isfinite(float(self.initial_value)):
            raise ValidationError("SRDParams requires initial_value to be finite")
        if not np.isfinite(float(self.volatility)):
            raise ValidationError("SRDParams requires volatility to be finite")
        if float(self.volatility) < 0.0:
            raise ValidationError("SRDParams requires volatility to be >= 0")

        if self.kappa is None or self.theta is None:
            raise ValidationError("SRDParams requires kappa and theta to be not None")

        if not np.isfinite(float(self.kappa)):
            raise ValidationError("SRDParams requires kappa to be finite")
        if not np.isfinite(float(self.theta)):
            raise ValidationError("SRDParams requires theta to be finite")
        if float(self.kappa) < 0.0:
            raise ValidationError("SRDParams requires kappa to be >= 0")
        if float(self.theta) < 0.0:
            raise ValidationError("SRDParams requires theta to be >= 0")

        # Feller condition: 2*kappa*theta >= sigma^2 ensures the process
        # stays strictly positive almost surely.  When violated the full-
        # truncation Euler scheme still converges, but paths will touch
        # zero more frequently.
        if 2.0 * float(self.kappa) * float(self.theta) < float(self.volatility) ** 2:
            warnings.warn(
                f"Feller condition violated: 2·κ·θ = {2 * self.kappa * self.theta:.6g} "
                f"< σ² = {self.volatility**2:.6g}. "
                "The CIR process may frequently touch zero.",
                stacklevel=2,
            )


class PathSimulation(ABC):
    """Base class for one-factor stochastic process path simulations.

    Parameters
    ----------
    market_data
        Market data used by the process (pricing date, discount curve, currency).
    process_params
        Model-specific process parameters (for example GBM, JD, or SRD parameters).
    sim
        Simulation configuration controlling path count and time grid rules.
    corr
        Optional multi-asset correlation context.
    name
        Optional process name; required when ``corr`` is provided.
    """

    def __init__(
        self,
        market_data: MarketData,
        process_params: GBMParams | JDParams | SRDParams,
        sim: SimulationConfig,
        corr: CorrelationContext | None = None,
        name: str | None = None,
    ) -> None:
        self._name = name
        self._market_data = market_data
        self._process_params = process_params
        self._sim = sim
        self._correlation_context = corr

        # Mutable working state (not from config)
        if isinstance(process_params, (GBMParams, JDParams)):
            self.discrete_dividends = process_params.discrete_dividends
        else:
            self.discrete_dividends = tuple()

        self._last_normals: np.ndarray | None = None
        self.time_grid = sim.time_grid
        self.observation_dates = set(sim.observation_dates)
        if self.discrete_dividends:
            for ex_date, _ in self.discrete_dividends:
                self.observation_dates.add(ex_date)

        if corr is not None:
            if name is None:
                raise ValidationError(
                    "name is required when using CorrelationContext "
                    "(used to look up the asset index)"
                )
            # Validate early: fail now rather than at simulation time.
            self._rn_index: int = corr.asset_index(name)

    @property
    def name(self) -> str | None:
        """Optional process identifier."""
        return self._name

    @property
    def market_data(self) -> MarketData:
        """Market data bound to the process."""
        return self._market_data

    @property
    def initial_value(self) -> float:
        """Initial state value at pricing date."""
        return self._process_params.initial_value

    @property
    def volatility(self) -> float:
        """Instantaneous volatility parameter."""
        return self._process_params.volatility

    @property
    def dividend_curve(self) -> DiscountCurve | None:
        """Continuous dividend discount curve, if provided."""
        return getattr(self._process_params, "dividend_curve", None)

    @property
    def paths(self) -> int:
        """Number of Monte Carlo paths."""
        return self._sim.paths

    @property
    def frequency(self) -> str | None:
        """Pandas frequency string for generated calendar grids."""
        return self._sim.frequency

    @property
    def num_steps(self) -> int | None:
        """Number of time steps for uniform-step grids."""
        return self._sim.num_steps

    @property
    def grid_start(self) -> dt.datetime | None:
        """Start of the dense time grid region.

        When set, ``num_steps`` equally-spaced intervals are placed in
        ``[grid_start, end_date]`` rather than ``[pricing_date, end_date]``.
        Defaults to ``None`` (dense grid starts at ``pricing_date``).
        """
        return self._sim.grid_start

    @property
    def day_count_convention(self) -> DayCountConvention:
        """Day-count basis used to convert dates to year fractions."""
        return self._sim.day_count_convention

    @property
    def end_date(self) -> dt.datetime | None:
        """Simulation horizon end date."""
        # Computed based on time_grid if present, else from sim config
        if self.time_grid is not None:
            return max(self.time_grid)
        return self._sim.end_date

    @property
    def correlation_context(self) -> CorrelationContext | None:
        """Correlation context for multi-asset path generation, if used."""
        return self._correlation_context

    @property
    def last_normals(self) -> np.ndarray | None:
        """Standard normals from the most recent :meth:`simulate` call.

        Shape ``(num_steps, num_paths)``.  Returns ``None`` before the
        first simulation.
        """
        return self._last_normals

    def _build_time_grid(self) -> np.ndarray:
        """Build a sorted datetime grid from simulation settings.

        When ``grid_start`` is set (e.g. for forward-starting Asians),
        the dense equally-spaced grid covers ``[grid_start, end_date]``
        with ``num_steps + 1`` points (or at the requested ``frequency``).
        ``pricing_date`` is always included as a boundary so the diffusion
        spans the full ``[pricing_date, end_date]`` interval.

        ``num_steps`` guarantees *at least* N+1 equally-spaced points in
        the dense region; ``observation_dates`` (e.g. ex-dividend dates)
        may add more.
        """
        dense_start = self.grid_start or self.pricing_date
        end = self.end_date

        if self.num_steps is not None:
            dates = pd.date_range(
                start=dense_start, end=end, periods=int(self.num_steps) + 1
            ).to_pydatetime()
        else:
            dates = pd.date_range(start=dense_start, end=end, freq=self.frequency).to_pydatetime()

        # Merge generated dates with required boundary + special dates
        all_dates = set(dates) | {self.pricing_date, dense_start, end} | self.observation_dates
        return np.array(sorted(all_dates))

    def _ensure_time_grid(self) -> None:
        """Populate ``time_grid`` lazily when not explicitly provided."""
        if self.time_grid is None:
            self.time_grid = self._build_time_grid()

    def _time_deltas(self) -> np.ndarray:
        """Return year-fraction deltas between consecutive grid dates."""
        self._ensure_time_grid()
        deltas = []
        for t in range(1, len(self.time_grid)):
            deltas.append(
                calculate_year_fraction(
                    self.time_grid[t - 1],
                    self.time_grid[t],
                    day_count_convention=self.day_count_convention,
                )
            )
        return np.array(deltas, dtype=float)

    def _standard_normals(self, random_seed: int | None, steps: int, paths: int) -> np.ndarray:
        """Generate or extract standard-normal shocks for path simulation."""
        if self.correlation_context is None:
            rng = np.random.default_rng(random_seed)

            if self._sim.antithetic and paths % 2 == 0:
                # Antithetic variates: generate half, mirror with negation
                half_paths = paths // 2
                ran = rng.standard_normal((steps, half_paths))
                ran = np.concatenate((ran, -ran), axis=1)
            else:
                if self._sim.antithetic and paths % 2 != 0:
                    warnings.warn(
                        f"antithetic=True but paths={paths} is odd; "
                        "antithetic variates require an even number of paths. "
                        "Falling back to plain random sampling.",
                        stacklevel=2,
                    )
                ran = rng.standard_normal((steps, paths))

            if self._sim.moment_matching:
                # Moment matching: centre and scale to N(0,1)
                ran = (ran - np.mean(ran)) / np.std(ran)

            self._last_normals = ran
            return ran

        corr = self.correlation_context
        base = corr.random_numbers[:, :steps, :]
        correlated = np.einsum("ij,jtp->itp", corr.cholesky_matrix, base)
        result = correlated[self._rn_index]
        self._last_normals = result
        return result

    @property
    def pricing_date(self) -> dt.datetime:
        """Pricing date from market data."""
        return self._market_data.pricing_date

    @property
    def currency(self) -> str:
        """Market-data currency."""
        return self._market_data.currency

    @property
    def discount_curve(self) -> DiscountCurve:
        """Discount curve from market data."""
        return self._market_data.discount_curve

    @staticmethod
    def _replace_allowed_fields() -> set[str]:
        return {
            "name",
            "market_data",
            "initial_value",
            "volatility",
            "dividend_curve",
            "discrete_dividends",
            "paths",
            "frequency",
            "num_steps",
            "day_count_convention",
            "time_grid",
            "observation_dates",
            "grid_start",
            "end_date",
            "pricing_date",
            "discount_curve",
            "currency",
        }

    def _validate_replace_kwargs(self, kwargs: dict) -> None:
        allowed_fields = self._replace_allowed_fields()
        unknown = set(kwargs).difference(allowed_fields)
        if unknown:
            raise ValidationError(
                f"replace() only supports overriding {sorted(allowed_fields)}; "
                f"unknown: {sorted(unknown)}"
            )

    def _clone_for_replace(self) -> PathSimulation:
        cloned = copy.copy(self)
        cloned.observation_dates = set(self.observation_dates)
        cloned.time_grid = None if self.time_grid is None else np.array(self.time_grid, copy=True)
        cloned._last_normals = None
        return cloned

    @staticmethod
    def _apply_name_override(cloned: PathSimulation, kwargs: dict) -> None:
        if "name" in kwargs:
            cloned._name = kwargs["name"]

    @staticmethod
    def _apply_process_param_overrides(cloned: PathSimulation, kwargs: dict) -> None:
        process_param_keys = {"initial_value", "volatility", "dividend_curve", "discrete_dividends"}
        if not process_param_keys.intersection(kwargs):
            return

        param_updates = {}
        for key in ("initial_value", "volatility", "dividend_curve", "discrete_dividends"):
            if key in kwargs:
                if not hasattr(cloned._process_params, key):
                    raise ValidationError(
                        f"{key} is not supported for {type(cloned._process_params).__name__}"
                    )
                param_updates[key] = kwargs[key]
        if param_updates:
            cloned._process_params = dc_replace(cloned._process_params, **param_updates)

    @staticmethod
    def _apply_sim_overrides(cloned: PathSimulation, kwargs: dict) -> None:
        sim_keys = {
            "paths",
            "frequency",
            "num_steps",
            "day_count_convention",
            "end_date",
            "grid_start",
        }
        if not sim_keys.intersection(kwargs):
            return

        sim_updates = {}
        if "paths" in kwargs:
            sim_updates["paths"] = kwargs["paths"]
        if "frequency" in kwargs:
            sim_updates["frequency"] = kwargs["frequency"]
        if "num_steps" in kwargs:
            sim_updates["num_steps"] = kwargs["num_steps"]
        if "day_count_convention" in kwargs:
            sim_updates["day_count_convention"] = kwargs["day_count_convention"]
        if "end_date" in kwargs:
            sim_updates["end_date"] = kwargs["end_date"]
        if "grid_start" in kwargs:
            sim_updates["grid_start"] = kwargs["grid_start"]
        cloned._sim = dc_replace(cloned._sim, **sim_updates)

    @staticmethod
    def _apply_observation_dates_override(cloned: PathSimulation, kwargs: dict) -> None:
        if "observation_dates" in kwargs:
            cloned.observation_dates = set(kwargs["observation_dates"] or [])

    @staticmethod
    def _apply_discrete_dividends_override(cloned: PathSimulation, kwargs: dict) -> None:
        if "discrete_dividends" in kwargs:
            cloned.discrete_dividends = tuple(kwargs["discrete_dividends"] or [])

    @staticmethod
    def _apply_time_grid_override(cloned: PathSimulation, kwargs: dict) -> bool:
        if "time_grid" not in kwargs:
            return False
        cloned.time_grid = (
            None if kwargs["time_grid"] is None else np.array(kwargs["time_grid"], copy=True)
        )
        return True

    @staticmethod
    def _needs_grid_rebuild(kwargs: dict) -> bool:
        grid_rebuild_keys = {"pricing_date", "end_date", "frequency", "num_steps", "grid_start"}
        return bool(grid_rebuild_keys.intersection(kwargs))

    @staticmethod
    def _handle_grid_state_after_overrides(cloned: PathSimulation, kwargs: dict) -> None:
        if PathSimulation._needs_grid_rebuild(kwargs):
            cloned.time_grid = None
            if cloned._sim.grid_start is not None and not cloned._sim._can_rebuild_grid():
                raise ConfigurationError(
                    "grid_start requires end_date + num_steps or frequency "
                    "for grid generation. Cannot set grid_start on a "
                    "SimulationConfig that lacks a discretisation method."
                )
            return

        if "observation_dates" in kwargs and cloned.time_grid is not None:
            augmented = sorted(set(cloned.time_grid) | cloned.observation_dates)
            cloned.time_grid = np.array(augmented, dtype=cloned.time_grid.dtype)
            return

        if "observation_dates" in kwargs:
            cloned.time_grid = None

    @staticmethod
    def _apply_market_data_overrides(cloned: PathSimulation, kwargs: dict) -> None:
        if "market_data" in kwargs:
            cloned._market_data = kwargs["market_data"]
            return

        market_data_keys = {"pricing_date", "discount_curve", "currency"}
        if not market_data_keys.intersection(kwargs):
            return

        updates = {}
        if "pricing_date" in kwargs:
            updates["pricing_date"] = kwargs["pricing_date"]
        if "discount_curve" in kwargs:
            updates["discount_curve"] = kwargs["discount_curve"]
        if "currency" in kwargs:
            updates["currency"] = kwargs["currency"]
        cloned._market_data = dc_replace(cloned._market_data, **updates)

    @staticmethod
    def _sync_dividend_dates_into_observation_dates(cloned: PathSimulation) -> None:
        if cloned.discrete_dividends:
            for ex_date, _ in cloned.discrete_dividends:
                cloned.observation_dates.add(ex_date)

    def replace(self, **kwargs) -> PathSimulation:
        """Return a cloned instance with selected attributes replaced.

        Parameters
        ----------
        **kwargs
            Supported overrides for process params, simulation config, and
            market data fields.

        Returns
        -------
        PathSimulation
            Clone with requested updates and cleared simulation caches.
        """
        self._validate_replace_kwargs(kwargs)

        cloned = self._clone_for_replace()
        self._apply_name_override(cloned, kwargs)
        self._apply_process_param_overrides(cloned, kwargs)
        self._apply_sim_overrides(cloned, kwargs)
        self._apply_observation_dates_override(cloned, kwargs)
        self._apply_discrete_dividends_override(cloned, kwargs)

        explicit_grid_override = self._apply_time_grid_override(cloned, kwargs)
        if not explicit_grid_override:
            self._handle_grid_state_after_overrides(cloned, kwargs)

        self._apply_market_data_overrides(cloned, kwargs)
        self._sync_dividend_dates_into_observation_dates(cloned)
        return cloned

    def simulate(self, random_seed: int | None = None) -> np.ndarray:
        """Generate and return simulated instrument value paths.

        Parameters
        ----------
        random_seed
            Random seed for reproducible path generation.

        Returns
        -------
        np.ndarray
            Simulated paths with shape ``(num_times, num_paths)``.
        """
        return self._generate_paths(random_seed=random_seed)

    @abstractmethod
    def _generate_paths(
        self,
        random_seed: int | None = None,
    ) -> np.ndarray:
        """Generate paths for the stochastic process.

        Subclasses must implement this method to define their specific
        path generation logic.

        Parameters
        ----------
        random_seed
            Random seed for reproducibility.
        """
        raise NotImplementedError


class GBMProcess(PathSimulation):
    """Class to generate simulated paths based on
    the Black-Scholes-Merton geometric Brownian motion model.
    """

    def _generate_paths(self, random_seed: int | None = None) -> np.ndarray:
        """Generate geometric Brownian motion paths.

        Implements the classic Black-Scholes-Merton model:
        dS_t = mu * S_t * dt + sigma * S_t * dW_t

        Parameters
        ----------
        random_seed
            Random seed for reproducibility.

        Notes
        -----
        The drift term (mu) is derived from the risk-free rate
        to ensure the model is calibrated to the discount curve.
        """
        self._ensure_time_grid()

        num_steps = len(self.time_grid) - 1
        num_paths = self.paths

        # Pre-calculate time deltas and random numbers
        time_deltas = self._time_deltas()
        z = self._standard_normals(random_seed, num_steps, num_paths)
        t_grid = np.array(
            [
                calculate_year_fraction(
                    self.pricing_date,
                    t,
                    day_count_convention=self.day_count_convention,
                )
                for t in self.time_grid
            ],
            dtype=float,
        )
        r_steps = np.array(
            [self.discount_curve.forward_rate(t0, t1) for t0, t1 in zip(t_grid[:-1], t_grid[1:])],
            dtype=float,
        )
        if self.dividend_curve is not None:
            q_steps = np.array(
                [
                    self.dividend_curve.forward_rate(t0, t1)
                    for t0, t1 in zip(t_grid[:-1], t_grid[1:])
                ],
                dtype=float,
            )
        else:
            q_steps = np.zeros_like(r_steps)

        # If no discrete dividends, use fully vectorized implementation (faster)
        if not self.discrete_dividends:
            dt_matrix = time_deltas.reshape(-1, 1)  # (num_steps, 1)

            # log S_t = log S_{t-1} + (r - q - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z
            drift = ((r_steps - q_steps)[:, None] - 0.5 * self.volatility**2) * dt_matrix
            diffusion = self.volatility * np.sqrt(dt_matrix) * z

            log_increments = drift + diffusion

            # Cumulative sum of log-increments
            # Prepend 0 for the start (log S_0 shift)
            log_path_increments = np.vstack([np.zeros(num_paths), log_increments])
            log_paths = np.cumsum(log_path_increments, axis=0)

            return self.initial_value * np.exp(log_paths)

        # Fallback to loop for discrete dividends case
        paths: np.ndarray = np.zeros((num_steps + 1, num_paths), dtype=float)
        paths[0] = self.initial_value

        dividend_by_date: dict[dt.datetime, float] = {}
        time_set = set(self.time_grid)
        for ex_date, amount in self.discrete_dividends:
            if ex_date in time_set:
                dividend_by_date[ex_date] = dividend_by_date.get(ex_date, 0.0) + float(amount)

        # Apply pricing-date dividend: input spot is cum-dividend, so the
        # stock goes ex immediately at t=0.
        div_t0 = dividend_by_date.get(self.time_grid[0])  # type: ignore[call-overload]
        if div_t0 is not None:
            paths[0] = np.maximum(paths[0] - div_t0, 0.0)

        for t in range(1, num_steps + 1):
            dt_step = time_deltas[t - 1]
            increment = ((r_steps[t - 1] - q_steps[t - 1]) - 0.5 * self.volatility**2) * dt_step
            increment += self.volatility * np.sqrt(dt_step) * z[t - 1]
            paths[t] = paths[t - 1] * np.exp(increment)

            div_amt = dividend_by_date.get(self.time_grid[t])  # type: ignore[call-overload]
            if div_amt is not None:
                paths[t] = np.maximum(paths[t] - div_amt, 0.0)

        return paths


class JDProcess(PathSimulation):
    """Merton (1976) jump diffusion (lognormal jumps).

    Risk-neutral discretisation (per step Δt):
        S_{t+Δt} = S_t * exp((r - q - λk - 0.5σ^2)Δt + σ√Δt Z)
                        * exp( Σ_{i=1}^{N} Y_i )

    where:
        N ~ Poisson(λΔt)
        Y_i ~ Normal(μ, δ^2)
        k = E[e^Y - 1] = exp(μ + 0.5δ^2) - 1
    """

    _process_params: JDParams

    def __init__(
        self,
        market_data: MarketData,
        process_params: JDParams,
        sim: SimulationConfig,
        corr: CorrelationContext | None = None,
        name: str | None = None,
    ):
        super().__init__(market_data, process_params, sim, corr=corr, name=name)

    @property
    def lambd(self) -> float:
        """Jump intensity parameter λ (per year)."""
        return self._process_params.lambd

    @property
    def mu(self) -> float:
        """Mean of log jump size distribution."""
        return self._process_params.mu

    @property
    def delta(self) -> float:
        """Standard deviation of log jump size distribution."""
        return self._process_params.delta

    def _generate_paths(self, random_seed: int | None = None) -> np.ndarray:
        """Generate jump-diffusion paths.

        Parameters
        ----------
        random_seed
            Random seed for reproducibility.

        Returns
        -------
        np.ndarray
            Simulated paths with shape ``(num_times, num_paths)``.
        """
        self._ensure_time_grid()

        num_steps = len(self.time_grid) - 1
        num_paths = self.paths

        t_grid = np.array(
            [
                calculate_year_fraction(
                    self.pricing_date,
                    t,
                    day_count_convention=self.day_count_convention,
                )
                for t in self.time_grid
            ],
            dtype=float,
        )
        r_steps = np.array(
            [self.discount_curve.forward_rate(t0, t1) for t0, t1 in zip(t_grid[:-1], t_grid[1:])],
            dtype=float,
        )
        if self.dividend_curve is not None:
            q_steps = np.array(
                [
                    self.dividend_curve.forward_rate(t0, t1)
                    for t0, t1 in zip(t_grid[:-1], t_grid[1:])
                ],
                dtype=float,
            )
        else:
            q_steps = np.zeros_like(r_steps)

        lam = float(self.lambd)
        mu_j = float(self.mu)
        sig_j = float(self.delta)
        vol = float(self.volatility)
        k = np.exp(mu_j + 0.5 * sig_j**2) - 1.0

        # Derive independent RNG streams for diffusion and jump components.
        # Using the same seed for both would create correlated bit streams;
        # SeedSequence.spawn() guarantees statistical independence.
        if random_seed is not None:
            ss = np.random.SeedSequence(random_seed)
            diffusion_ss, jump_ss = ss.spawn(2)
            diffusion_seed = int(diffusion_ss.generate_state(1)[0])
            jump_rng = np.random.default_rng(jump_ss)
        else:
            diffusion_seed = None
            jump_rng = np.random.default_rng()

        z = self._standard_normals(diffusion_seed, num_steps, num_paths)
        time_deltas = self._time_deltas()

        if not self.discrete_dividends:
            dt_matrix = time_deltas.reshape(-1, 1)  # (num_steps, 1)

            # 1. Diffusion component
            drift_diffusion = ((r_steps - q_steps)[:, None] - lam * k - 0.5 * vol**2) * dt_matrix
            diffusion_term = vol * np.sqrt(dt_matrix) * z

            # 2. Jump component (uses independent RNG stream)
            poi_counts = jump_rng.poisson(lam * dt_matrix, size=(len(dt_matrix), num_paths))

            # Sum of N_Δ i.i.d. N(μ_J, δ_J²) jumps is N(N_Δ·μ_J, N_Δ·δ_J²).
            # Sample the sum directly: N_Δ·μ_J + √N_Δ·δ_J·Z_J.
            jump_normals = jump_rng.standard_normal(size=(len(dt_matrix), num_paths))
            jump_magnitude = np.where(
                poi_counts > 0, poi_counts * mu_j + np.sqrt(poi_counts) * sig_j * jump_normals, 0.0
            )

            log_increments = drift_diffusion + diffusion_term + jump_magnitude

            log_path_increments = np.vstack([np.zeros(num_paths), log_increments])
            log_paths = np.cumsum(log_path_increments, axis=0)

            return self.initial_value * np.exp(log_paths)

        # Fallback loop (discrete dividends)
        paths: np.ndarray = np.zeros((num_steps + 1, num_paths), dtype=float)
        paths[0] = self.initial_value

        dividend_by_date: dict[dt.datetime, float] = {}
        time_set = set(self.time_grid)
        for ex_date, amount in self.discrete_dividends:
            if ex_date in time_set:
                dividend_by_date[ex_date] = dividend_by_date.get(ex_date, 0.0) + float(amount)

        for t in range(1, num_steps + 1):
            dt_step = time_deltas[t - 1]
            drift = ((r_steps[t - 1] - q_steps[t - 1]) - lam * k - 0.5 * vol**2) * dt_step
            diffusion = vol * np.sqrt(dt_step) * z[t - 1]
            diffusion_multiplier = np.exp(drift + diffusion)

            jump_counts = jump_rng.poisson(lam * dt_step, size=num_paths)
            jump_sum = np.where(
                jump_counts > 0,
                jump_counts * mu_j
                + np.sqrt(jump_counts) * sig_j * jump_rng.standard_normal(num_paths),
                0.0,
            )
            jump_multiplier = np.exp(jump_sum)

            paths[t] = paths[t - 1] * diffusion_multiplier * jump_multiplier

            div_amt = dividend_by_date.get(self.time_grid[t])  # type: ignore[call-overload]
            if div_amt is not None:
                paths[t] = np.maximum(paths[t] - div_amt, 0.0)

        return paths


class SRDProcess(PathSimulation):
    """Cox–Ingersoll–Ross (1985) square-root diffusion (CIR/SRD).

    Model (under chosen measure):
        dX_t = kappa*(theta - X_t) dt + sigma*sqrt(X_t) dW_t

    Uses full truncation Euler to preserve non-negativity.
    """

    _process_params: SRDParams

    def __init__(
        self,
        market_data: MarketData,
        process_params: SRDParams,
        sim: SimulationConfig,
        corr: CorrelationContext | None = None,
        name: str | None = None,
    ):
        super().__init__(market_data, process_params, sim, corr=corr, name=name)

    @property
    def kappa(self) -> float:
        """Mean reversion speed parameter."""
        return self._process_params.kappa

    @property
    def theta(self) -> float:
        """Long-run mean level parameter."""
        return self._process_params.theta

    def _generate_paths(self, random_seed: int | None = None) -> np.ndarray:
        """Generate Cox-Ingersoll-Ross (square-root diffusion) paths.

        Uses the full-truncation Euler scheme of Lord, Koekkoek & van Dijk
        (2010) to guarantee non-negative paths: the drift and diffusion
        coefficients are evaluated at max(x, 0), and the raw Euler step
        is floored at zero after each increment.

        Parameters
        ----------
        random_seed
            Random seed for reproducibility.

        Returns
        -------
        np.ndarray
            Simulated non-negative SRD paths with shape ``(num_times, num_paths)``.
        """
        self._ensure_time_grid()

        num_steps = len(self.time_grid) - 1
        num_paths = self.paths

        # Truncated paths (non-negative) and raw Euler buffer
        paths: np.ndarray = np.zeros((num_steps + 1, num_paths), dtype=float)
        euler_raw = np.zeros_like(paths)

        paths[0] = self.initial_value
        euler_raw[0] = self.initial_value

        z = self._standard_normals(random_seed, num_steps, num_paths)
        time_deltas = self._time_deltas()

        for t in range(1, num_steps + 1):
            dt_step = time_deltas[t - 1]
            x_prev = paths[t - 1, :]  # truncated (non-negative) values
            mean_reversion = self.kappa * (self.theta - x_prev) * dt_step
            diffusion = self.volatility * np.sqrt(x_prev * dt_step) * z[t - 1]
            euler_raw[t] = euler_raw[t - 1] + mean_reversion + diffusion
            paths[t] = np.maximum(euler_raw[t], 0.0)

        return paths
