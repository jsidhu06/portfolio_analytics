from typing import Optional
import numpy as np
import pandas as pd
from .stochastic_processes import GeometricBrownianMotion, SquareRootDiffusion, JumpDiffusion
from .valuation import ValuationMCSEuropean, ValuationMCSAmerican
from .utils import sn_random_numbers

# models available for risk factor modeling
MODELS = {"gbm": GeometricBrownianMotion, "jd": JumpDiffusion, "srd": SquareRootDiffusion}

# allowed exercise types
OTYPES = {"European": ValuationMCSEuropean, "American": ValuationMCSAmerican}


class DerivativesPosition:
    """Class to model a derivatives position.

    Attributes
    ==========

    name: str
        name of the object
    quantity: float
        number of assets/derivatives making up the position
    contract_size: int
        size of one contract (e.g. number of shares per option contract)
    underlying: str
        name of asset/risk factor for the derivative
    mar_env: instance of market_environment
        constants, lists, and curves relevant for valuation_class
    otype: str
        valuation class to use
    side: str
        'call' or 'put' for options

    Methods
    =======
    get_info:
        prints information about the derivative position
    """

    def __init__(self, name, quantity, contract_size, underlying, mar_env, otype, side):
        self.name = name
        self.quantity = quantity
        self.contract_size = contract_size
        self.underlying = underlying
        self.mar_env = mar_env
        self.otype = otype
        self.side = side

    def __str__(self) -> str:
        lines = []

        lines.append("NAME")
        lines.append(f"{self.name}\n")

        lines.append("QUANTITY")
        lines.append(f"{self.quantity}\n")

        lines.append("UNDERLYING")
        lines.append(f"{self.underlying}\n")

        lines.append("MARKET ENVIRONMENT")

        lines.append("\n**Constants**")
        for key, value in self.mar_env.constants.items():
            lines.append(f"{key}: {value}")

        lines.append("\n**Lists**")
        for key, value in self.mar_env.lists.items():
            lines.append(f"{key}: {value}")

        lines.append("\n**Curves**")
        for key in self.mar_env.curves.keys():
            lines.append(f"{key}")

        lines.append("\nOPTION TYPE")
        lines.append(f"{self.otype}\n")

        lines.append("SIDE")
        lines.append(f"{self.side}")

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
    val_env: market_environment
        market environment for the valuation
    assets: dict
        dictionary of market environments for the assets
    correlations: list
        correlations between assets
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
        self, name, positions, val_env, assets, correlations=None, random_seed: Optional[int] = None
    ):
        self.name = name
        self.positions = positions
        self.val_env = val_env
        self.assets = assets
        self.underlyings = set()
        self.correlations = correlations
        self.time_grid = None
        self.underlying_objects = {}  # Dict[str, PathSimulation]
        self.valuation_objects = {}  # Dict[str,OptionValuation]
        self.random_seed = random_seed
        self.special_dates = []
        for pos in self.positions:
            # determine earliest starting_date
            self.val_env.constants["starting_date"] = min(
                self.val_env.constants["starting_date"], positions[pos].mar_env.pricing_date
            )
            # determine latest date of relevance
            self.val_env.constants["final_date"] = max(
                self.val_env.constants["final_date"], positions[pos].mar_env.constants["maturity"]
            )
            # collect all underlyings and
            # add to set (avoids redundancy)
            self.underlyings.add(positions[pos].underlying)

        # generate general time grid
        start = self.val_env.constants["starting_date"]
        end = self.val_env.constants["final_date"]
        time_grid = list(
            pd.date_range(
                start=start, end=end, freq=self.val_env.constants["frequency"]
            ).to_pydatetime()
        )
        for pos in self.positions:
            maturity_date = positions[pos].mar_env.constants["maturity"]
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
            ul_list = sorted(self.underlyings)
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
            rn_set = {asset: ul_list.index(asset) for asset in self.underlyings}

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

        for asset in self.underlyings:
            # select market environment of asset
            mar_env = self.assets[asset]
            # add valuation environment to market environment
            mar_env.add_environment(val_env)
            # select right simulation class
            model = MODELS.get(mar_env.constants["model"])
            if model is None:
                raise ValueError(f"Model must be one of {list(MODELS.keys())}")
            # instantiate simulation object
            if correlations is not None:
                self.underlying_objects[asset] = model(asset, mar_env, corr=True)
            else:
                self.underlying_objects[asset] = model(asset, mar_env, corr=False)

        for pos in positions:
            # select right valuation class (European, American)
            val_class = OTYPES[positions[pos].otype]
            # pick market environment and add valuation environment
            mar_env = positions[pos].mar_env
            mar_env.add_environment(self.val_env)
            # instantiate valuation class
            self.valuation_objects[pos] = val_class(
                name=positions[pos].name,
                mar_env=mar_env,
                underlying=self.underlying_objects[positions[pos].underlying],
                side=positions[pos].side,
            )

    def get_positions(self):
        """Convenience method to get information about
        all derivatives positions in a portfolio."""
        for pos in self.positions:
            bar = "\n" + 50 * "-"
            print(bar)
            print(self.positions[pos])
            print(bar)

    def get_statistics(self, random_seed: Optional[int] = None):
        """Provides portfolio statistics."""
        res_list = []
        # iterate over all positions in portfolio
        for pos, value in self.valuation_objects.items():  # Dict[str,OptionValuation]
            p: DerivativesPosition = self.positions[pos]
            pv = value.present_value(random_seed=random_seed)
            res_list.append(
                [
                    p.name,
                    p.quantity,
                    # calculate all present values for the single instruments
                    pv,
                    value.currency,
                    # single instrument contact size
                    p.contract_size,
                    # underlying spot price
                    self.assets[p.underlying].get_constant("initial_value"),
                    # single instrument value times quantity
                    pv * p.contract_size * p.quantity,
                    # calculate Delta of position
                    value.delta(random_seed=random_seed) * p.contract_size * p.quantity,
                    # calculate Vega of position
                    value.vega(random_seed=random_seed) * p.contract_size * p.quantity,
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
