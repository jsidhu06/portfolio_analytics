from dataclasses import dataclass
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
from .valuation import OptionValuation, OptionSpec
from .valuation import MonteCarloParams
from .enums import OptionType, ExerciseType, PricingMethod
from .market_environment import CorrelationContext, ValuationEnvironment
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


@dataclass
class UnderlyingConfig:
    """Configuration for an underlying asset in portfolio simulations.

    Specifies the stochastic process model and its parameters for a particular
    underlying asset. Used to instantiate PathSimulation objects in portfolio context.

    Attributes
    ==========
    name: str
        Name/identifier of the underlying asset (e.g., 'STOCK', 'INDEX')
    model: str
        Stochastic process model type: 'gbm', 'jd', or 'srd'
    initial_value: float
        Initial spot price or rate
    volatility: float
        Volatility of the process
    lambd: float
        Jump intensity (for 'jd' model only)
    mu: float
        Mean of jump size log returns (for 'jd' model only)
    delta: float
        Standard deviation of jump size log returns (for 'jd' model only)
    kappa: float
        Mean reversion speed (for 'srd' model only)
    theta: float
        Long-run mean level (for 'srd' model only)
    """

    name: str
    model: str  # 'gbm', 'jd', 'srd'
    initial_value: float
    volatility: float
    # Optional JD (Jump Diffusion) parameters
    lambd: float | None = None
    mu: float | None = None
    delta: float | None = None
    # Optional SRD (Square Root Diffusion / CIR) parameters
    kappa: float | None = None
    theta: float | None = None

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name.strip():
            raise TypeError("UnderlyingConfig.name must be a non-empty string")
        if not isinstance(self.model, str) or not self.model.strip():
            raise TypeError("UnderlyingConfig.model must be a non-empty string")

        allowed_models = {"gbm", "jd", "srd"}
        if self.model not in allowed_models:
            raise ValueError(
                f"UnderlyingConfig.model must be one of {sorted(allowed_models)}, got '{self.model}'"
            )

        if self.initial_value is None:
            raise ValueError("UnderlyingConfig.initial_value must be not None")
        if self.volatility is None:
            raise ValueError("UnderlyingConfig.volatility must be not None")

        try:
            self.initial_value = float(self.initial_value)
            self.volatility = float(self.volatility)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                "UnderlyingConfig.initial_value and volatility must be numeric"
            ) from exc

        if not np.isfinite(self.initial_value):
            raise ValueError("UnderlyingConfig.initial_value must be finite")
        if not np.isfinite(self.volatility):
            raise ValueError("UnderlyingConfig.volatility must be finite")
        if self.volatility < 0.0:
            raise ValueError("UnderlyingConfig.volatility must be >= 0")

        # Model-specific validation
        if self.model == "jd":
            missing = [
                name
                for name, value in (
                    ("lambd", self.lambd),
                    ("mu", self.mu),
                    ("delta", self.delta),
                )
                if value is None
            ]
            if missing:
                raise ValueError(
                    f"UnderlyingConfig for model='jd' requires {', '.join(missing)} to be set"
                )
            self.lambd = float(self.lambd)
            self.mu = float(self.mu)
            self.delta = float(self.delta)
            if not np.isfinite(self.lambd):
                raise ValueError("UnderlyingConfig.lambd must be finite")
            if not np.isfinite(self.mu):
                raise ValueError("UnderlyingConfig.mu must be finite")
            if not np.isfinite(self.delta):
                raise ValueError("UnderlyingConfig.delta must be finite")
            if self.lambd < 0.0:
                raise ValueError("UnderlyingConfig.lambd must be >= 0")
            if self.delta < 0.0:
                raise ValueError("UnderlyingConfig.delta must be >= 0")

        if self.model == "srd":
            missing = [
                name
                for name, value in (("kappa", self.kappa), ("theta", self.theta))
                if value is None
            ]
            if missing:
                raise ValueError(
                    f"UnderlyingConfig for model='srd' requires {', '.join(missing)} to be set"
                )
            self.kappa = float(self.kappa)
            self.theta = float(self.theta)
            if not np.isfinite(self.kappa):
                raise ValueError("UnderlyingConfig.kappa must be finite")
            if not np.isfinite(self.theta):
                raise ValueError("UnderlyingConfig.theta must be finite")
            if self.kappa < 0.0:
                raise ValueError("UnderlyingConfig.kappa must be >= 0")
            if self.theta < 0.0:
                raise ValueError("UnderlyingConfig.theta must be >= 0")


class DerivativesPosition:
    """Class to model a derivatives position.

    A position represents a holding of a derivatives contract: a quantity of
    a specific underlying asset, with contract terms defined by spec.

    Attributes
    ==========
    name: str
        name/identifier of the position
    quantity: int
        number of contracts held
    underlying: str
        name of the risk factor/asset (must match a key in portfolio.underlyings)
    spec: OptionSpec
        Contract specification (type, exercise type, strike, maturity, currency, etc.)
    """

    def __init__(
        self,
        name: str,
        quantity: int,
        underlying: str,
        spec: OptionSpec,
    ):
        self.name = name
        self.quantity = quantity
        self.underlying = underlying
        self.spec = spec

    def __str__(self) -> str:
        lines = []

        lines.append("NAME")
        lines.append(f"{self.name}\n")

        lines.append("QUANTITY")
        lines.append(f"{self.quantity}\n")

        lines.append("UNDERLYING")
        lines.append(f"{self.underlying}\n")

        lines.append("OPTION SPECIFICATION")
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
    val_env: ValuationEnvironment
        valuation environment for the portfolio
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
        val_env: ValuationEnvironment,
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
        self.pricing_date = val_env.market_data.pricing_date
        self.end_date = None
        self.time_grid = None
        self.underlying_objects: dict[str, PathSimulation] = {}
        self.valuation_objects: dict[str, OptionValuation] = {}
        self.random_seed = random_seed
        self.special_dates = set()

        # Validate inputs + derive portfolio schedule
        if not positions:
            raise ValueError("positions dict must not be empty")
        if not underlyings:
            raise ValueError("underlyings dict must not be empty")

        # Validate that each position references a valid underlying
        for position_name, position in positions.items():
            if position.underlying not in underlyings:
                raise ValueError(
                    f"Position '{position_name}' references underlying '{position.underlying}' "
                    f"which is not in the underlyings dict. Available underlyings: {list(underlyings.keys())}"
                )

        position_maturities = [p.spec.maturity for p in positions.values()]

        self.end_date = max(position_maturities)

        # Base grid from starting_date to end_date
        time_grid = list(
            pd.date_range(
                start=self.pricing_date,
                end=self.end_date,
                freq=self.val_env.frequency,
            ).to_pydatetime()
        )

        # special dates are maturity dates that are not in the regular time grid; we add these
        # to the time grid
        self.special_dates = set(position_maturities).difference(time_grid)

        # Ensure all key dates are included (pricing dates + maturities)
        required_dates = set([self.pricing_date, self.end_date] + position_maturities)
        time_grid.extend(required_dates)

        # Delete duplicates and sort
        time_grid = sorted(set(time_grid))
        self.time_grid = np.array(time_grid)

        # Construct SimulationConfig once (shared across underlyings)
        sim_config = SimulationConfig(
            paths=self.val_env.paths,
            day_count_convention=self.val_env.day_count_convention,
            time_grid=self.time_grid,
        )

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
                (len(rn_set), len(self.time_grid), self.val_env.paths),
                random_seed=self.random_seed,
            )

            # Create CorrelationContext
            corr_context = CorrelationContext(
                cholesky_matrix=cholesky_matrix,
                random_numbers=random_numbers,
                rn_set=rn_set,
            )
        else:
            corr_context = None

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
                    lambd=asset_config.lambd,
                    mu=asset_config.mu,
                    delta=asset_config.delta,
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

            # Instantiate the PathSimulation object
            self.underlying_objects[asset_name] = model(
                name=asset_name,
                market_data=self.val_env.market_data,
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
            params = MonteCarloParams(random_seed=random_seed)
            pv = value.present_value(params=params)
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
                    value.delta(params=params) * p.spec.contract_size * p.quantity,
                    # calculate Vega of position
                    value.vega(params=params) * p.spec.contract_size * p.quantity,
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
