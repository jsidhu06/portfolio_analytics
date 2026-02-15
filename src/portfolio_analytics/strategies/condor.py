from __future__ import annotations

from dataclasses import dataclass
import datetime as dt
from typing import TYPE_CHECKING

import numpy as np

from ..valuation import OptionSpec, OptionValuation
from ..enums import ExerciseType, OptionType, PositionSide
from ..valuation import ValuationParams

if TYPE_CHECKING:
    from ..stochastic_processes import PathSimulation
    from ..valuation import UnderlyingPricingData
    from ..enums import PricingMethod


@dataclass(frozen=True, slots=True)
class CondorSpec:
    """A simple 4-leg condor strategy specification.

    This is provided primarily as an educational/example structure.

    Notes
    -----
    A *long condor* payoff is:

    - put spread:   +put(K2) - put(K1)
    - call spread:  +call(K3) - call(K4)

    where strikes are ordered $K_1 < K_2 < K_3 < K_4$.

    The class provides:
    - `terminal_payoff(spot)` for the combined terminal payoff
    - `present_value(...)` to value as sum of 4 vanilla legs (EU or AM)
    - `leg_definitions()` to build a basket of vanilla options for valuation
    """

    exercise_type: ExerciseType
    strikes: tuple[float, float, float, float]
    maturity: dt.datetime
    currency: str
    side: PositionSide = PositionSide.LONG
    contract_size: int | float = 100

    def __post_init__(self) -> None:
        if not isinstance(self.exercise_type, ExerciseType):
            raise TypeError(
                f"exercise_type must be ExerciseType enum, got {type(self.exercise_type).__name__}"
            )
        if not isinstance(self.side, PositionSide):
            raise TypeError(f"side must be PositionSide enum, got {type(self.side).__name__}")

        k1, k2, k3, k4 = self.strikes
        if not (k1 < k2):
            raise ValueError("Condor strikes must satisfy K1 < K2")
        if not (k3 < k4):
            raise ValueError("Condor strikes must satisfy K3 < K4")
        if not (k2 < k3):
            raise ValueError("Condor strikes must satisfy K2 < K3")

    def terminal_payoff(self, spot: np.ndarray | float) -> np.ndarray:
        """Vectorized terminal payoff at maturity as a function of spot.

        Notes
        -----
        This is a *terminal payoff* function (European-style payoff). For AMERICAN
        exercise products, pricing requires an early-exercise model. Use
        `present_value(...)` to value the strategy as a sum of independently
        exercisable vanilla legs.
        """
        if self.exercise_type != ExerciseType.EUROPEAN:
            raise ValueError("terminal_payoff(...) is only valid for EUROPEAN exercise")

        s = np.asarray(spot, dtype=float)
        k1, k2, k3, k4 = self.strikes
        payoff = (
            np.maximum(k2 - s, 0.0)
            - np.maximum(k1 - s, 0.0)
            + np.maximum(s - k3, 0.0)
            - np.maximum(s - k4, 0.0)
        )
        if self.side == PositionSide.SHORT:
            payoff = -payoff
        return payoff

    def present_value(
        self,
        *,
        name: str,
        underlying: "PathSimulation | UnderlyingPricingData",
        pricing_method: "PricingMethod",
        params: ValuationParams | None = None,
    ) -> float:
        """Value the condor as a 4-leg strategy: sum of vanilla leg present values.

        This matches the standard interpretation of an options strategy where each
        leg is its own contract. For AMERICAN exercise, each leg is assumed to be
        independently exercisable (so the strategy PV is the sum of leg PVs).

        Parameters
        ==========
        name:
            Base name used when constructing per-leg valuation objects.
        underlying:
            Underlying model/data passed through to per-leg `OptionValuation`.
        pricing_method:
            PricingMethod used for each leg.
        params:
            Params applied to each per-leg valuation.
        """
        total = 0.0
        for opt_type, strike, weight in self.leg_definitions():
            leg_spec = OptionSpec(
                option_type=opt_type,
                exercise_type=self.exercise_type,
                strike=strike,
                maturity=self.maturity,
                currency=self.currency,
                contract_size=self.contract_size,
            )
            leg_val = OptionValuation(
                name=f"{name}_leg_{opt_type.value}_{strike}",
                underlying=underlying,
                spec=leg_spec,
                pricing_method=pricing_method,
                params=params,
            )
            total += weight * leg_val.present_value()

        return float(total)

    def leg_definitions(self) -> list[tuple[OptionType, float, float]]:
        """Return `(option_type, strike, weight)` definitions for the 4 legs.

        Weights are +1/-1 for a LONG position. For SHORT, weights are negated.
        """
        k1, k2, k3, k4 = self.strikes
        legs: list[tuple[OptionType, float, float]] = [
            (OptionType.PUT, k1, -1.0),
            (OptionType.PUT, k2, +1.0),
            (OptionType.CALL, k3, +1.0),
            (OptionType.CALL, k4, -1.0),
        ]
        if self.side == PositionSide.SHORT:
            legs = [(opt_type, strike, -weight) for (opt_type, strike, weight) in legs]
        return legs
