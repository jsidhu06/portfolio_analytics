"""Delta hedging simulator.

Simulates delta hedging strategies and analyzes PnL attribution.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING
import numpy as np
import pandas as pd
from ..utils import calculate_year_fraction

if TYPE_CHECKING:
    from ..valuation.core import OptionValuation
    from ..stochastic_processes import PathSimulation


@dataclass(frozen=True)
class HedgingParams:
    """Parameters for delta hedging simulation.

    Parameters
    ----------
    rebalance_frequency : str, optional
        Rebalancing frequency: 'D' (daily), 'W' (weekly), 'M' (monthly) (default: 'D')
    transaction_cost_bps : float, optional
        Transaction cost in basis points (default: 0.0)
    include_gamma_hedge : bool, optional
        Whether to include gamma hedging (default: False)
    """

    rebalance_frequency: str = "D"
    transaction_cost_bps: float = 0.0
    include_gamma_hedge: bool = False

    def __post_init__(self):
        """Validate parameters."""
        if self.rebalance_frequency not in ("D", "W", "M"):
            raise ValueError("rebalance_frequency must be 'D', 'W', or 'M'")
        if self.transaction_cost_bps < 0:
            raise ValueError("transaction_cost_bps must be non-negative")


@dataclass
class HedgingResult:
    """Result of delta hedging simulation.

    Parameters
    ----------
    pnl_total : np.ndarray
        Total PnL for each path
    pnl_attribution : pd.DataFrame
        PnL attribution by Greek (delta, gamma, theta, vega, etc.)
    hedge_ratios : np.ndarray
        Hedge ratios over time (shape: [num_timesteps, num_paths])
    final_pnl_mean : float
        Mean final PnL
    final_pnl_std : float
        Standard deviation of final PnL
    """

    pnl_total: np.ndarray
    pnl_attribution: pd.DataFrame
    hedge_ratios: np.ndarray
    final_pnl_mean: float
    final_pnl_std: float


class DeltaHedgingSimulator:
    """Simulator for delta hedging strategies.

    Simulates delta hedging of an option position and tracks PnL.
    """

    def __init__(
        self,
        option: "OptionValuation",
        underlying_process: "PathSimulation",
        params: HedgingParams,
    ):
        """Initialize delta hedging simulator.

        Parameters
        ----------
        option : OptionValuation
            Option to hedge
        underlying_process : PathSimulation
            Stochastic process for underlying asset
        params : HedgingParams
            Hedging parameters
        """
        self.option = option
        self.underlying_process = underlying_process
        self.params = params

    def run(self, random_seed: int | None = None) -> HedgingResult:
        """Run delta hedging simulation.

        Parameters
        ----------
        random_seed : int, optional
            Random seed for path generation

        Returns
        -------
        HedgingResult
            Hedging simulation results
        """
        # Generate underlying paths
        paths = self.underlying_process.get_instrument_values(random_seed=random_seed)
        if self.underlying_process.time_grid is None:
            self.underlying_process.generate_time_grid()
        time_grid = self.underlying_process.time_grid

        num_paths = paths.shape[1]
        num_steps = paths.shape[0]

        # Initialize arrays
        hedge_ratios = np.zeros((num_steps, num_paths))
        pnl_total = np.zeros(num_paths)
        pnl_delta = np.zeros(num_paths)
        pnl_gamma = np.zeros(num_paths)
        pnl_theta = np.zeros(num_paths)
        pnl_vega = np.zeros(num_paths)

        # Get initial option price
        initial_option_price = self.option.present_value()

        # Transaction cost rate
        cost_rate = self.params.transaction_cost_bps / 10000.0

        # Simulate hedging for each path
        for path_idx in range(num_paths):
            path = paths[:, path_idx]
            cash = -initial_option_price  # Short option position
            shares = 0.0

            for step in range(num_steps - 1):
                current_spot = path[step]
                next_spot = path[step + 1]

                # Calculate current delta
                bumped_underlying = self.option.underlying.replace(initial_value=current_spot)
                bumped_option = type(self.option)(
                    name=f"{self.option.name}_hedge",
                    underlying=bumped_underlying,
                    spec=self.option.spec,
                    pricing_method=self.option.pricing_method,
                    params=self.option.params,
                )

                current_delta = bumped_option.delta()
                hedge_ratios[step, path_idx] = current_delta

                # Rebalance hedge
                target_shares = current_delta
                shares_to_trade = target_shares - shares

                # Transaction costs
                transaction_cost = abs(shares_to_trade) * current_spot * cost_rate
                cash -= shares_to_trade * current_spot + transaction_cost
                shares = target_shares

                # Theta decay (approximate)
                if step < num_steps - 1:
                    dt = calculate_year_fraction(
                        time_grid[step],
                        time_grid[step + 1],
                        day_count_convention=365,
                    )
                    theta_val = bumped_option.theta()
                    cash += theta_val * dt  # Theta benefit (we're short)

            # Final settlement
            final_spot = path[-1]
            bumped_underlying = self.option.underlying.replace(initial_value=final_spot)
            bumped_option = type(self.option)(
                name=f"{self.option.name}_final",
                underlying=bumped_underlying,
                spec=self.option.spec,
                pricing_method=self.option.pricing_method,
                params=self.option.params,
            )
            final_option_price = bumped_option.present_value()

            # Final PnL
            option_pnl = initial_option_price - final_option_price  # We're short
            stock_pnl = shares * (final_spot - path[0])
            pnl_total[path_idx] = option_pnl + stock_pnl + cash

            # Simplified attribution (for educational purposes)
            # In practice, this would be more sophisticated
            pnl_delta[path_idx] = shares * (final_spot - path[0])
            pnl_theta[path_idx] = (
                initial_option_price - final_option_price
            ) * 0.5  # Simplified
            pnl_gamma[path_idx] = pnl_total[path_idx] - pnl_delta[path_idx] - pnl_theta[path_idx]
            pnl_vega[path_idx] = 0.0  # Simplified

        # Create attribution DataFrame
        pnl_attribution = pd.DataFrame(
            {
                "delta": pnl_delta,
                "gamma": pnl_gamma,
                "theta": pnl_theta,
                "vega": pnl_vega,
                "total": pnl_total,
            }
        )

        return HedgingResult(
            pnl_total=pnl_total,
            pnl_attribution=pnl_attribution,
            hedge_ratios=hedge_ratios,
            final_pnl_mean=np.mean(pnl_total),
            final_pnl_std=np.std(pnl_total),
        )
