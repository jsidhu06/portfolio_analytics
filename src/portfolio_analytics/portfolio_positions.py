from typing import Sequence, Type
import numpy as np
import pandas as pd
from .stochastic_processes import (
    PathSimulation,
    GeometricBrownianMotion,
    SquareRootDiffusion,
    JumpDiffusion,
    GBMParams,
    JDParams,
    SRDParams,
    SimulationConfig,
)
from .valuation import OptionValuation, OptionSpec, UnderlyingConfig
from .enums import OptionType, ExerciseType, PricingMethod
from .market_environment import MarketEnvironment, MarketData, CorrelationContext
from .utils import sn_random_numbers

# models available for risk factor modeling
MODELS: dict[str, Type[PathSimulation]] = {
    "gbm": GeometricBrownianMotion,
    "jd": JumpDiffusion,
    "srd": SquareRootDiffusion,
}

# parameter types for each model
MODEL_PARAMS: dict[str, type] = {
    "gbm": GBMParams,
    "jd": JDParams,
    "srd": SRDParams,
}

# mapping of string option types to enums
OPTION_TYPES: dict[str, OptionType] = {
    "call": OptionType.CALL,
    "put": OptionType.PUT,
}

# mapping of string exercise types to enums
EXERCISE_TYPES: dict[str, ExerciseType] = {
    "european": ExerciseType.EUROPEAN,
    "american": ExerciseType.AMERICAN,
}


class DerivativesPosition:
    """Class to model a derivatives position.

    Attributes
    ==========
    name: str
        name of the object
    quantity: float
        number of assets/derivatives making up the position
    underlying: str
        name of asset/risk factor for the derivative
    spec: OptionSpec
        Contract specification (type, exercise type, strike, maturity, etc.)
    market_data: MarketData
        Market data (pricing date, discount curve, currency)
    """

    def __init__(
        self,
        name: str,
        quantity: int,
        underlying: str,
        spec: OptionSpec,
        market_data: MarketData,
    ):
        self.name = name
        self.quantity = quantity
        self.underlying = underlying
        self.spec = spec
        self.market_data = market_data

    def __str__(self) -> str:
        lines = []

        lines.append("NAME")
        lines.append(f"{self.name}\n")

        lines.append("QUANTITY")
        lines.append(f"{self.quantity}\n")

        lines.append("UNDERLYING")
        lines.append(f"{self.underlying}\n")

        lines.append("MARKET DATA")
        lines.append(f"Pricing Date: {self.market_data.pricing_date}")
        lines.append(f"Currency: {self.market_data.currency}")
        lines.append(f"Discount Curve: {self.market_data.discount_curve.name}")

        lines.append("\nOPTION SPECIFICATION")
        lines.append(f"Type: {self.spec.option_type.value}")
        lines.append(f"Exercise: {self.spec.exercise_type.value}")
        lines.append(f"Strike: {self.spec.strike}")
        lines.append(f"Maturity: {self.spec.maturity}")
        lines.append(f"Currency: {self.spec.currency}")
        lines.append(f"Contract Size: {self.spec.contract_size}")

        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.__str__()


class DerivativesPortfolio:
    """Class for modeling and valuing portfolios of derivatives positions.

    Attributes
    ==========
    name: str
        name of the object
    positions: dict[str,DerivativesPosition]
        dictionary of positions (instances of DerivativesPosition class)
    val_env: MarketEnvironment
        market environment for the valuation
    underlyings: dict[str,UnderlyingConfig]
        dictionary of underlying asset configurations
    correlations: Sequence[tuple[str, str, float]], optional
        list of tuples with pairwise correlations between assets
    random_seed: int, optional
        random number generator seed

    Methods
    =======
    get_positions:
        prints information about the single portfolio positions
    get_statistics:
        returns a pandas DataFrame object with portfolio statistics
    """

    def __init__(
        self,
        name: str,
        positions: dict[str, DerivativesPosition],
        val_env: MarketEnvironment,
        underlyings: dict[str, UnderlyingConfig],
        correlations: Sequence[tuple[str, str, float]] | None = None,
        random_seed: int | None = None,
    ):
        self.name = name
        self.positions = positions
        self.val_env = val_env
        self.underlyings_config = underlyings
        self.underlying_names: set[str] = set(underlyings.keys())
        self.correlations = correlations
        self.time_grid = None
        self.underlying_objects: dict[str, PathSimulation] = {}
        self.valuation_objects: dict[str, OptionValuation] = {}
        self.random_seed = random_seed
        self.special_dates = []
        for pos in self.positions:
            # determine earliest starting_date
            self.val_env.constants["starting_date"] = min(
                self.val_env.constants["starting_date"], positions[pos].market_data.pricing_date
            )
            # determine latest date of relevance
            self.val_env.constants["final_date"] = max(
                self.val_env.constants["final_date"], positions[pos].spec.maturity
            )

        # generate general time grid
        start = self.val_env.constants["starting_date"]
        end = self.val_env.constants["final_date"]
        time_grid = list(
            pd.date_range(
                start=start, end=end, freq=self.val_env.constants["frequency"]
            ).to_pydatetime()
        )
        for pos in self.positions:
            maturity_date = positions[pos].spec.maturity
            if maturity_date not in time_grid:
                time_grid.insert(0, maturity_date)
                self.special_dates.append(maturity_date)
        if start not in time_grid:
            time_grid.insert(0, start)
        if end not in time_grid:
            time_grid.append(end)
        # delete duplicate entries
        time_grid = list(set(time_grid))
        # sort dates in time_grid
        time_grid.sort()
        self.time_grid = np.array(time_grid)
        self.val_env.add_list("time_grid", self.time_grid)

        if correlations is not None:
            # take care of correlations
            ul_list = sorted(self.underlying_names)
            correlation_matrix = np.zeros((len(ul_list), len(ul_list)))
            np.fill_diagonal(correlation_matrix, 1.0)
            correlation_matrix = pd.DataFrame(correlation_matrix, index=ul_list, columns=ul_list)
            for i, j, corr in correlations:
                corr = min(corr, 0.999999999999)
                # fill correlation matrix
                correlation_matrix.loc[i, j] = corr
                correlation_matrix.loc[j, i] = corr
            # determine Cholesky matrix
            cholesky_matrix = np.linalg.cholesky(np.array(correlation_matrix))

            # dictionary with index positions for the
            # slice of the random number array to be used by
            # respective underlying
            rn_set = {asset: ul_list.index(asset) for asset in self.underlying_names}

            # random numbers array, to be used by
            # all underlyings (if correlations exist)
            random_numbers = sn_random_numbers(
                (len(rn_set), len(self.time_grid), self.val_env.constants["paths"]),
                random_seed=self.random_seed,
            )

            # add all to valuation environment that is
            # to be shared with every underlying
            self.val_env.add_list("cholesky_matrix", cholesky_matrix)
            self.val_env.add_list("random_numbers", random_numbers)
            self.val_env.add_list("rn_set", rn_set)

        for asset_name, asset_config in self.underlyings_config.items():
            # Get the model class for this underlying
            model = MODELS.get(asset_config.model)
            if model is None:
                raise ValueError(
                    f"Model '{asset_config.model}' not found. Must be one of {list(MODELS.keys())}"
                )

            # Construct process parameters based on model type
            if asset_config.model == "gbm":
                process_params = GBMParams(
                    initial_value=asset_config.initial_value,
                    volatility=asset_config.volatility,
                )
            elif asset_config.model == "jd":
                process_params = JDParams(
                    initial_value=asset_config.initial_value,
                    volatility=asset_config.volatility,
                    jump_intensity=asset_config.jump_intensity,
                    jump_mean=asset_config.jump_mean,
                    jump_std=asset_config.jump_std,
                )
            elif asset_config.model == "srd":
                process_params = SRDParams(
                    initial_value=asset_config.initial_value,
                    volatility=asset_config.volatility,
                    kappa=asset_config.kappa,
                    theta=asset_config.theta,
                )
            else:
                raise ValueError(f"Unknown model type: {asset_config.model}")

            # Construct SimulationConfig
            sim_config = SimulationConfig(
                paths=self.val_env.constants["paths"],
                frequency=self.val_env.constants["frequency"],
                final_date=self.val_env.constants["final_date"],
                day_count_convention=self.val_env.constants.get("day_count_convention", 365),
                time_grid=self.time_grid,
                special_dates=self.special_dates,
            )

            # Construct CorrelationContext if correlations exist
            if correlations is not None:
                corr_context = CorrelationContext(
                    cholesky_matrix=self.val_env.get_list("cholesky_matrix"),
                    random_numbers=self.val_env.get_list("random_numbers"),
                    rn_set=self.val_env.get_list("rn_set"),
                )
            else:
                corr_context = None

            # Instantiate the PathSimulation object
            self.underlying_objects[asset_name] = model(
                name=asset_name,
                market_data=asset_config.market_data,
                process_params=process_params,
                sim=sim_config,
                corr=corr_context,
            )

        for pos in positions:
            # instantiate valuation class with Monte Carlo pricing
            self.valuation_objects[pos] = OptionValuation(
                name=positions[pos].name,
                underlying=self.underlying_objects[positions[pos].underlying],
                spec=positions[pos].spec,
                pricing_method=PricingMethod.MONTE_CARLO,
            )

    def get_positions(self):
        """Convenience method to get information about
        all derivatives positions in a portfolio."""
        for pos in self.positions:
            bar = "\n" + 50 * "-"
            print(bar)
            print(self.positions[pos])
            print(bar)

    def get_statistics(self, random_seed: int | None = None):
        """Provides portfolio statistics."""
        res_list = []
        # iterate over all positions in portfolio
        for pos, value in self.valuation_objects.items():
            p: DerivativesPosition = self.positions[pos]
            pv = value.present_value(random_seed=random_seed)
            res_list.append(
                [
                    p.name,
                    p.quantity,
                    # calculate all present values for the single instruments
                    pv,
                    p.spec.currency,
                    # single instrument contact size
                    p.spec.contract_size,
                    # underlying spot price
                    self.underlyings_config[p.underlying].initial_value,
                    # single instrument value times quantity
                    pv * p.spec.contract_size * p.quantity,
                    # calculate Delta of position
                    value.delta(random_seed=random_seed) * p.spec.contract_size * p.quantity,
                    # calculate Vega of position
                    value.vega(random_seed=random_seed) * p.spec.contract_size * p.quantity,
                ]
            )
        # generate a pandas DataFrame object with all results
        res_df = pd.DataFrame(
            res_list,
            columns=[
                "name",
                "quantity",
                "value",
                "curr.",
                "contract_size",
                "underlying_price",
                "pos_value",
                "pos_delta",
                "pos_vega",
            ],
        )
        return res_df
