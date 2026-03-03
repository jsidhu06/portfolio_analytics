"""Tests for Asian option pricing — analytical, Monte Carlo, and binomial.

Covers:
- Geometric: Kemna-Vorst (1990) exact closed-form
- Arithmetic: Turnbull-Wakeman (1991) moment-matching (Hull §26.13)
- Seasoned Asians (Hull K* adjustment)
- Binomial Hull tree, Binomial MC
- American Asian (Longstaff-Schwartz LSM)
- Fixing-dates and economic-property tests
"""

import datetime as dt
import logging
from typing import Sequence

import numpy as np
import pytest
from scipy.stats import norm

from portfolio_analytics.enums import (
    AsianAveraging,
    DayCountConvention,
    ExerciseType,
    OptionType,
    PricingMethod,
)
from portfolio_analytics.exceptions import ValidationError
from portfolio_analytics.market_environment import MarketData
from portfolio_analytics.rates import DiscountCurve
from portfolio_analytics.tests.helpers import flat_curve
from portfolio_analytics.stochastic_processes import (
    GBMParams,
    GBMProcess,
    SimulationConfig,
)
from portfolio_analytics.utils import calculate_year_fraction, pv_discrete_dividends
from portfolio_analytics.valuation import OptionSpec, OptionValuation, UnderlyingPricingData
from portfolio_analytics.valuation.asian_analytical import (
    _asian_arithmetic_analytical,
    _asian_geometric_analytical,
)
from portfolio_analytics.valuation.core import AsianOptionSpec
from portfolio_analytics.valuation.params import BinomialParams, MonteCarloParams


logger = logging.getLogger(__name__)


PRICING_DATE = dt.datetime(2025, 1, 1)
CURRENCY = "USD"
MC_PATHS = 200_000
MC_SEED = 42
NUM_STEPS = 60
BINOM_STEPS = 100
ASIAN_TREE_AVERAGES = 100


def _market_data(short_rate: float, maturity: dt.datetime) -> MarketData:
    curve = flat_curve(PRICING_DATE, maturity, short_rate)
    return MarketData(PRICING_DATE, curve, currency=CURRENCY)


def _flat_dividend_curve(dividend_yield: float, maturity: dt.datetime) -> DiscountCurve | None:
    if dividend_yield == 0.0:
        return None
    ttm = calculate_year_fraction(PRICING_DATE, maturity)
    return DiscountCurve.flat(dividend_yield, end_time=ttm)


def _underlying(
    *,
    spot: float,
    vol: float,
    short_rate: float,
    maturity: dt.datetime,
    dividend_curve: DiscountCurve | None = None,
    discrete_dividends: Sequence[tuple[dt.datetime, float]] | None = None,
) -> UnderlyingPricingData:
    return UnderlyingPricingData(
        initial_value=spot,
        volatility=vol,
        market_data=_market_data(short_rate, maturity),
        dividend_curve=dividend_curve,
        discrete_dividends=discrete_dividends,
    )


def _gbm_underlying(
    *,
    spot: float,
    vol: float,
    short_rate: float,
    maturity: dt.datetime,
    paths: int,
    num_steps: int,
    dividend_curve: DiscountCurve | None = None,
    discrete_dividends: Sequence[tuple[dt.datetime, float]] | None = None,
) -> GBMProcess:
    sim_config = SimulationConfig(
        paths=paths,
        day_count_convention=DayCountConvention.ACT_365F,
        num_steps=num_steps,
        end_date=maturity,
    )
    gbm_params = GBMParams(
        initial_value=spot,
        volatility=vol,
        dividend_curve=dividend_curve,
        discrete_dividends=discrete_dividends,
    )
    return GBMProcess(
        _market_data(short_rate, maturity),
        gbm_params,
        sim_config,
    )


def _asian_spec(
    *,
    strike: float,
    maturity: dt.datetime,
    call_put: OptionType,
    exercise_type: ExerciseType = ExerciseType.EUROPEAN,
    averaging: AsianAveraging = AsianAveraging.ARITHMETIC,
    num_steps: int | None = None,
    averaging_start: dt.datetime | None = None,
) -> AsianOptionSpec:
    return AsianOptionSpec(
        averaging=averaging,
        call_put=call_put,
        strike=strike,
        maturity=maturity,
        currency=CURRENCY,
        exercise_type=exercise_type,
        num_steps=num_steps,
        averaging_start=averaging_start,
    )


@pytest.mark.parametrize(
    "spot,strike,vol,short_rate,dividend_yield,days,call_put",
    [
        (100.0, 100.0, 0.2, 0.03, 0.0, 365, OptionType.CALL),
        (100.0, 100.0, 0.2, 0.03, 0.02, 365, OptionType.CALL),
        (110.0, 100.0, 0.25, 0.01, 0.0, 270, OptionType.PUT),
        (110.0, 100.0, 0.25, 0.01, 0.015, 270, OptionType.PUT),
        (95.0, 90.0, 0.3, 0.05, 0.0, 540, OptionType.CALL),
        (95.0, 90.0, 0.3, 0.05, 0.025, 540, OptionType.CALL),
        (105.0, 110.0, 0.18, 0.02, 0.0, 180, OptionType.PUT),
        (105.0, 110.0, 0.18, 0.02, 0.01, 180, OptionType.PUT),
    ],
)
def test_asian_binomial_hull_close_to_mc(
    spot, strike, vol, short_rate, dividend_yield, days, call_put
):
    maturity = PRICING_DATE + dt.timedelta(days=days)
    spec = _asian_spec(strike=strike, maturity=maturity, call_put=call_put)
    q_curve = _flat_dividend_curve(dividend_yield, maturity)

    mc_underlying = _gbm_underlying(
        spot=spot,
        vol=vol,
        short_rate=short_rate,
        dividend_curve=q_curve,
        maturity=maturity,
        paths=MC_PATHS,
        num_steps=NUM_STEPS,
    )
    mc_pv = OptionValuation(
        mc_underlying,
        spec,
        PricingMethod.MONTE_CARLO,
        params=MonteCarloParams(random_seed=MC_SEED),
    ).present_value()

    binom_underlying = _underlying(
        spot=spot,
        vol=vol,
        short_rate=short_rate,
        dividend_curve=q_curve,
        maturity=maturity,
    )

    binom_mc_pv = OptionValuation(
        binom_underlying,
        spec,
        PricingMethod.BINOMIAL,
        params=BinomialParams(
            num_steps=NUM_STEPS * 2,
            mc_paths=MC_PATHS,
            random_seed=MC_SEED,
        ),
    ).present_value()

    hull_pv = OptionValuation(
        binom_underlying,
        spec,
        PricingMethod.BINOMIAL,
        params=BinomialParams(
            num_steps=NUM_STEPS,
            asian_tree_averages=ASIAN_TREE_AVERAGES,
        ),
    ).present_value()

    logger.info(
        "Asian %s S=%.2f K=%.2f vol=%.2f r=%.2f q=%.2f days=%d\nMC=%.6f Hull=%.6f BinomMC=%.6f",
        call_put.value,
        spot,
        strike,
        vol,
        short_rate,
        dividend_yield,
        days,
        mc_pv,
        hull_pv,
        binom_mc_pv,
    )

    assert np.isclose(mc_pv, hull_pv, rtol=0.02)
    assert np.isclose(binom_mc_pv, hull_pv, rtol=0.02)


@pytest.mark.parametrize(
    "div_days,div_amt,rtol_mc_adj,mc_paths",
    [
        ([90, 270], 0.5, 0.03, 100_000),
        ([30, 330], 0.5, 0.05, 50_000),
        ([60, 300], 1.0, 0.06, 50_000),
        ([30, 120, 210, 330], 0.25, 0.05, 50_000),
    ],
)
def test_asian_discrete_dividends_binomial_hull_vs_mc(div_days, div_amt, rtol_mc_adj, mc_paths):
    spot = 52.0
    strike = 50.0
    vol = 0.4
    short_rate = 0.1
    days = 365
    maturity = PRICING_DATE + dt.timedelta(days=days)
    divs = [(PRICING_DATE + dt.timedelta(days=day), div_amt) for day in div_days]

    spec = _asian_spec(strike=strike, maturity=maturity, call_put=OptionType.CALL)

    mc_underlying = _gbm_underlying(
        spot=spot,
        vol=vol,
        short_rate=short_rate,
        discrete_dividends=divs,
        maturity=maturity,
        paths=mc_paths,
        num_steps=NUM_STEPS,
    )
    mc_pv = OptionValuation(
        mc_underlying,
        spec,
        PricingMethod.MONTE_CARLO,
        params=MonteCarloParams(random_seed=MC_SEED),
    ).present_value()

    binom_underlying = _underlying(
        spot=spot,
        vol=vol,
        short_rate=short_rate,
        discrete_dividends=divs,
        maturity=maturity,
    )

    binom_mc_pv = OptionValuation(
        binom_underlying,
        spec,
        PricingMethod.BINOMIAL,
        params=BinomialParams(
            num_steps=NUM_STEPS * 2,
            mc_paths=MC_PATHS,
            random_seed=MC_SEED,
        ),
    ).present_value()

    hull_pv = OptionValuation(
        binom_underlying,
        spec,
        PricingMethod.BINOMIAL,
        params=BinomialParams(
            num_steps=NUM_STEPS,
            asian_tree_averages=ASIAN_TREE_AVERAGES,
        ),
    ).present_value()

    pv_divs = pv_discrete_dividends(
        dividends=divs,
        curve_date=PRICING_DATE,
        end_date=maturity,
        discount_curve=binom_underlying.discount_curve,
    )
    vol_multiplier = spot / (spot - pv_divs)
    adjusted_underlying = binom_underlying.replace(
        volatility=binom_underlying.volatility * vol_multiplier
    )
    adjusted_hull_pv = OptionValuation(
        adjusted_underlying,
        spec,
        PricingMethod.BINOMIAL,
        params=BinomialParams(
            num_steps=NUM_STEPS,
            asian_tree_averages=ASIAN_TREE_AVERAGES,
        ),
    ).present_value()

    logger.info(
        "Asian disc-div S=%.2f K=%.2f vol=%.2f r=%.2f divs=%d\n"
        "MC=%.6f Hull=%.6f BinomMC=%.6f HullAdj=%.6f",
        spot,
        strike,
        vol,
        short_rate,
        len(divs),
        mc_pv,
        hull_pv,
        binom_mc_pv,
        adjusted_hull_pv,
    )

    assert np.isclose(binom_mc_pv, hull_pv, rtol=0.02)
    assert np.isclose(mc_pv, adjusted_hull_pv, rtol=rtol_mc_adj)


@pytest.mark.parametrize(
    "spot,strike,vol,short_rate,days,call_put",
    [
        (100.0, 100.0, 0.2, 0.03, 365, OptionType.CALL),
        (100.0, 100.0, 0.2, 0.03, 365, OptionType.PUT),
        (110.0, 100.0, 0.25, 0.01, 270, OptionType.CALL),
        (110.0, 100.0, 0.25, 0.01, 270, OptionType.PUT),
    ],
)
def test_asian_american_at_least_european_hull(spot, strike, vol, short_rate, days, call_put):
    maturity = PRICING_DATE + dt.timedelta(days=days)
    euro_spec = _asian_spec(
        strike=strike, maturity=maturity, call_put=call_put, exercise_type=ExerciseType.EUROPEAN
    )
    amer_spec = _asian_spec(
        strike=strike, maturity=maturity, call_put=call_put, exercise_type=ExerciseType.AMERICAN
    )

    binom_underlying = _underlying(
        spot=spot,
        vol=vol,
        short_rate=short_rate,
        dividend_curve=None,
        maturity=maturity,
    )

    euro_pv = OptionValuation(
        binom_underlying,
        euro_spec,
        PricingMethod.BINOMIAL,
        params=BinomialParams(
            num_steps=NUM_STEPS,
            asian_tree_averages=ASIAN_TREE_AVERAGES,
        ),
    ).present_value()

    amer_pv = OptionValuation(
        binom_underlying,
        amer_spec,
        PricingMethod.BINOMIAL,
        params=BinomialParams(
            num_steps=NUM_STEPS,
            asian_tree_averages=ASIAN_TREE_AVERAGES,
        ),
    ).present_value()

    assert amer_pv >= euro_pv - 1e-8


# ---------------------------------------------------------------------------
# Geometric Averaging Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "spot,strike,vol,short_rate,days,call_put",
    [
        (100.0, 100.0, 0.2, 0.03, 365, OptionType.CALL),
        (100.0, 100.0, 0.2, 0.03, 365, OptionType.PUT),
        (110.0, 100.0, 0.25, 0.01, 270, OptionType.CALL),
        (95.0, 90.0, 0.3, 0.05, 540, OptionType.CALL),
        (105.0, 110.0, 0.18, 0.02, 180, OptionType.PUT),
    ],
)
def test_geometric_asian_mc_positive_value(spot, strike, vol, short_rate, days, call_put):
    """Geometric Asian options should produce positive option values."""
    maturity = PRICING_DATE + dt.timedelta(days=days)
    spec = _asian_spec(
        strike=strike,
        maturity=maturity,
        call_put=call_put,
        averaging=AsianAveraging.GEOMETRIC,
    )

    mc_underlying = _gbm_underlying(
        spot=spot,
        vol=vol,
        short_rate=short_rate,
        maturity=maturity,
        paths=MC_PATHS,
        num_steps=NUM_STEPS,
    )
    pv = OptionValuation(
        mc_underlying,
        spec,
        PricingMethod.MONTE_CARLO,
        params=MonteCarloParams(random_seed=MC_SEED),
    ).present_value()

    assert pv > 0.0


@pytest.mark.parametrize(
    "spot,strike,vol,short_rate,days,call_put",
    [
        (100.0, 100.0, 0.2, 0.03, 365, OptionType.CALL),
        (100.0, 100.0, 0.2, 0.03, 365, OptionType.PUT),
        (110.0, 100.0, 0.25, 0.01, 270, OptionType.CALL),
    ],
)
def test_geometric_leq_arithmetic_asian(spot, strike, vol, short_rate, days, call_put):
    """Geometric average ≤ arithmetic average ⟹ geometric call ≤ arithmetic call
    (and geometric put ≥ arithmetic put for the same strike).

    For calls: geometric payoff ≤ arithmetic payoff (AM-GM inequality).
    For puts:  geometric payoff ≥ arithmetic payoff.

    In expectation (hence PV), the same ordering holds.
    """
    maturity = PRICING_DATE + dt.timedelta(days=days)

    arith_spec = _asian_spec(
        strike=strike,
        maturity=maturity,
        call_put=call_put,
        averaging=AsianAveraging.ARITHMETIC,
    )
    geom_spec = _asian_spec(
        strike=strike,
        maturity=maturity,
        call_put=call_put,
        averaging=AsianAveraging.GEOMETRIC,
    )

    mc_underlying = _gbm_underlying(
        spot=spot,
        vol=vol,
        short_rate=short_rate,
        maturity=maturity,
        paths=MC_PATHS,
        num_steps=NUM_STEPS,
    )

    arith_pv = OptionValuation(
        mc_underlying,
        arith_spec,
        PricingMethod.MONTE_CARLO,
        params=MonteCarloParams(random_seed=MC_SEED),
    ).present_value()

    geom_pv = OptionValuation(
        mc_underlying,
        geom_spec,
        PricingMethod.MONTE_CARLO,
        params=MonteCarloParams(random_seed=MC_SEED),
    ).present_value()

    if call_put is OptionType.CALL:
        # Geometric call ≤ arithmetic call (AM-GM)
        assert geom_pv <= arith_pv + 1e-8
    else:
        # Geometric put ≥ arithmetic put (AM-GM)
        assert geom_pv >= arith_pv - 1e-8


@pytest.mark.parametrize(
    "spot,strike,vol,short_rate,days,call_put",
    [
        (100.0, 100.0, 0.2, 0.03, 365, OptionType.CALL),
        (110.0, 100.0, 0.25, 0.01, 270, OptionType.CALL),
        (100.0, 100.0, 0.2, 0.03, 365, OptionType.PUT),
    ],
)
def test_geometric_asian_binomial_hull_close_to_mc(spot, strike, vol, short_rate, days, call_put):
    """Binomial Hull tree with geometric averaging should match MC geometric."""
    maturity = PRICING_DATE + dt.timedelta(days=days)
    spec = _asian_spec(
        strike=strike,
        maturity=maturity,
        call_put=call_put,
        averaging=AsianAveraging.GEOMETRIC,
    )

    mc_underlying = _gbm_underlying(
        spot=spot,
        vol=vol,
        short_rate=short_rate,
        maturity=maturity,
        paths=MC_PATHS,
        num_steps=NUM_STEPS,
    )
    mc_pv = OptionValuation(
        mc_underlying,
        spec,
        PricingMethod.MONTE_CARLO,
        params=MonteCarloParams(random_seed=MC_SEED),
    ).present_value()

    binom_underlying = _underlying(
        spot=spot,
        vol=vol,
        short_rate=short_rate,
        maturity=maturity,
    )
    hull_pv = OptionValuation(
        binom_underlying,
        spec,
        PricingMethod.BINOMIAL,
        params=BinomialParams(
            num_steps=NUM_STEPS,
            asian_tree_averages=ASIAN_TREE_AVERAGES,
        ),
    ).present_value()

    logger.info(
        "Geometric Asian %s S=%.2f K=%.2f | MC=%.6f Hull=%.6f",
        call_put.value,
        spot,
        strike,
        mc_pv,
        hull_pv,
    )
    assert np.isclose(mc_pv, hull_pv, rtol=0.03)


# ---------------------------------------------------------------------------
# American Asian MC (Longstaff-Schwartz) Tests
# ---------------------------------------------------------------------------


class TestAmericanAsianMC:
    """Tests for Longstaff-Schwartz American Asian option pricing."""

    SPOT = 100.0
    STRIKE = 100.0
    VOL = 0.2
    SHORT_RATE = 0.03
    DAYS = 365
    PATHS = 100_000
    SEED = 42
    NUM_STEPS = 60

    @property
    def maturity(self) -> dt.datetime:
        return PRICING_DATE + dt.timedelta(days=self.DAYS)

    def _mc_underlying(
        self,
        *,
        spot: float | None = None,
        vol: float | None = None,
        short_rate: float | None = None,
        maturity: dt.datetime | None = None,
        paths: int | None = None,
        dividend_curve: DiscountCurve | None = None,
    ) -> GBMProcess:
        return _gbm_underlying(
            spot=spot or self.SPOT,
            vol=vol or self.VOL,
            short_rate=short_rate or self.SHORT_RATE,
            maturity=maturity or self.maturity,
            paths=paths or self.PATHS,
            num_steps=self.NUM_STEPS,
            dividend_curve=dividend_curve,
        )

    def _price(
        self,
        call_put: OptionType,
        exercise_type: ExerciseType,
        averaging: AsianAveraging = AsianAveraging.ARITHMETIC,
        *,
        spot: float | None = None,
        strike: float | None = None,
        vol: float | None = None,
        short_rate: float | None = None,
        maturity: dt.datetime | None = None,
        paths: int | None = None,
        seed: int | None = None,
        dividend_curve: DiscountCurve | None = None,
    ) -> float:
        mat = maturity or self.maturity
        spec = _asian_spec(
            strike=strike or self.STRIKE,
            maturity=mat,
            call_put=call_put,
            exercise_type=exercise_type,
            averaging=averaging,
        )
        underlying = self._mc_underlying(
            spot=spot,
            vol=vol,
            short_rate=short_rate,
            maturity=mat,
            paths=paths,
            dividend_curve=dividend_curve,
        )
        return OptionValuation(
            underlying,
            spec,
            PricingMethod.MONTE_CARLO,
            params=MonteCarloParams(random_seed=seed or self.SEED),
        ).present_value()

    # -- American >= European for puts (early exercise premium) --

    @pytest.mark.parametrize("call_put", [OptionType.PUT])
    @pytest.mark.parametrize("averaging", [AsianAveraging.ARITHMETIC, AsianAveraging.GEOMETRIC])
    def test_american_geq_european_put(self, call_put, averaging):
        """American Asian put >= European Asian put (early exercise premium)."""
        euro = self._price(call_put, ExerciseType.EUROPEAN, averaging)
        amer = self._price(call_put, ExerciseType.AMERICAN, averaging)
        assert amer >= euro - 1e-6, (
            f"American ({amer:.6f}) < European ({euro:.6f}) for {averaging.value} {call_put.value}"
        )

    # -- American >= European for calls with dividends --

    def test_american_geq_european_call_with_dividends(self):
        """American Asian call >= European when dividends present."""
        q_curve = flat_curve(PRICING_DATE, self.maturity, 0.04)
        euro = self._price(OptionType.CALL, ExerciseType.EUROPEAN, dividend_curve=q_curve)
        amer = self._price(OptionType.CALL, ExerciseType.AMERICAN, dividend_curve=q_curve)
        assert amer >= euro - 1e-6

    # -- Positive prices --

    @pytest.mark.parametrize("call_put", [OptionType.CALL, OptionType.PUT])
    @pytest.mark.parametrize("averaging", [AsianAveraging.ARITHMETIC, AsianAveraging.GEOMETRIC])
    def test_positive_price(self, call_put, averaging):
        pv = self._price(call_put, ExerciseType.AMERICAN, averaging)
        assert pv > 0.0

    # -- Price increases with volatility --

    @pytest.mark.parametrize("call_put", [OptionType.CALL, OptionType.PUT])
    def test_price_increases_with_vol(self, call_put):
        low = self._price(call_put, ExerciseType.AMERICAN, vol=0.15)
        high = self._price(call_put, ExerciseType.AMERICAN, vol=0.35)
        assert high > low

    # -- MC American vs Binomial Hull American --

    @pytest.mark.parametrize(
        "call_put,averaging",
        [
            (OptionType.CALL, AsianAveraging.ARITHMETIC),
            (OptionType.PUT, AsianAveraging.ARITHMETIC),
            (OptionType.CALL, AsianAveraging.GEOMETRIC),
            (OptionType.PUT, AsianAveraging.GEOMETRIC),
        ],
    )
    def test_mc_american_close_to_hull_american(self, call_put, averaging):
        """MC LSM American Asian should be close to Hull binomial American Asian."""
        mat = self.maturity
        spec = _asian_spec(
            strike=self.STRIKE,
            maturity=mat,
            call_put=call_put,
            exercise_type=ExerciseType.AMERICAN,
            averaging=averaging,
        )

        # MC American
        mc_underlying = self._mc_underlying()
        mc_pv = OptionValuation(
            mc_underlying,
            spec,
            PricingMethod.MONTE_CARLO,
            params=MonteCarloParams(random_seed=self.SEED),
        ).present_value()

        # Hull binomial American
        binom_underlying = _underlying(
            spot=self.SPOT,
            vol=self.VOL,
            short_rate=self.SHORT_RATE,
            maturity=mat,
        )
        hull_pv = OptionValuation(
            binom_underlying,
            spec,
            PricingMethod.BINOMIAL,
            params=BinomialParams(
                num_steps=NUM_STEPS,
                asian_tree_averages=ASIAN_TREE_AVERAGES,
            ),
        ).present_value()

        logger.info(
            "American Asian %s %s | MC=%.6f Hull=%.6f",
            averaging.value,
            call_put.value,
            mc_pv,
            hull_pv,
        )
        assert np.isclose(mc_pv, hull_pv, rtol=0.03), (
            f"MC ({mc_pv:.6f}) vs Hull ({hull_pv:.6f}) too far apart"
        )

    # -- solve() returns expected shapes --

    def test_solve_returns_correct_shapes(self):
        mat = self.maturity
        spec = _asian_spec(
            strike=self.STRIKE,
            maturity=mat,
            call_put=OptionType.CALL,
            exercise_type=ExerciseType.AMERICAN,
        )
        n_paths = 1000
        underlying = _gbm_underlying(
            spot=self.SPOT,
            vol=self.VOL,
            short_rate=self.SHORT_RATE,
            maturity=mat,
            paths=n_paths,
            num_steps=self.NUM_STEPS,
        )
        ov = OptionValuation(
            underlying,
            spec,
            PricingMethod.MONTE_CARLO,
            params=MonteCarloParams(random_seed=self.SEED),
        )
        avg_paths, running_avg, intrinsic = ov.solve()
        assert avg_paths.shape == running_avg.shape == intrinsic.shape
        assert avg_paths.shape[1] == n_paths
        # All intrinsic values should be non-negative
        assert np.all(intrinsic >= 0.0)

    # -- BSM American Asian raises --

    def test_bsm_american_asian_raises(self):
        """BSM does not support American exercise for Asian options."""
        spec = _asian_spec(
            strike=self.STRIKE,
            maturity=self.maturity,
            call_put=OptionType.CALL,
            exercise_type=ExerciseType.AMERICAN,
        )
        binom_underlying = _underlying(
            spot=self.SPOT,
            vol=self.VOL,
            short_rate=self.SHORT_RATE,
            maturity=self.maturity,
        )
        with pytest.raises(ValidationError, match="AMERICAN.*BSM|BSM.*AMERICAN"):
            OptionValuation(
                binom_underlying,
                spec,
                PricingMethod.BSM,
            )


# ═══════════════════════════════════════════════════════════════════════════
# Fixing-dates & economic-property tests
# ═══════════════════════════════════════════════════════════════════════════

_FD_SPOT = 100.0
_FD_STRIKE = 100.0
_FD_VOL = 0.20
_FD_RATE = 0.05
_FD_MATURITY = PRICING_DATE + dt.timedelta(days=365)
_FD_PATHS = 200_000
_FD_SEED = 42
_FD_STEPS = 60


def _fd_market_data(
    r_curve: DiscountCurve | None = None,
) -> MarketData:
    curve = r_curve if r_curve is not None else flat_curve(PRICING_DATE, _FD_MATURITY, _FD_RATE)
    return MarketData(PRICING_DATE, curve, currency=CURRENCY)


def _fd_gbm(
    *,
    dividend_curve: DiscountCurve | None = None,
) -> GBMProcess:
    md = _fd_market_data()
    params = GBMParams(
        initial_value=_FD_SPOT,
        volatility=_FD_VOL,
        dividend_curve=dividend_curve,
    )
    sim_cfg = SimulationConfig(
        paths=_FD_PATHS,
        end_date=_FD_MATURITY,
        num_steps=_FD_STEPS,
    )
    return GBMProcess(md, params, sim_cfg)


# Monthly fixing dates used as baseline
_MONTHLY_FIXINGS = tuple(
    dt.datetime(2025, m, 1) if m <= 12 else dt.datetime(2026, m - 12, 1) for m in range(2, 14)
)


def _fd_asian_pv(
    call_put: OptionType,
    averaging: AsianAveraging = AsianAveraging.ARITHMETIC,
    exercise_type: ExerciseType = ExerciseType.EUROPEAN,
    fixing_dates: tuple[dt.datetime, ...] = _MONTHLY_FIXINGS,
    dividend_curve: DiscountCurve | None = None,
    seed: int = _FD_SEED,
) -> float:
    """Helper: MC Asian PV using explicit fixing dates."""
    spec = AsianOptionSpec(
        averaging=averaging,
        call_put=call_put,
        strike=_FD_STRIKE,
        maturity=_FD_MATURITY,
        currency=CURRENCY,
        exercise_type=exercise_type,
        fixing_dates=fixing_dates,
    )
    gbm = _fd_gbm(dividend_curve=dividend_curve)
    return OptionValuation(
        gbm,
        spec,
        PricingMethod.MONTE_CARLO,
        params=MonteCarloParams(random_seed=seed),
    ).present_value()


class TestDividendEffectOnAsians:
    """Dividends should reduce call value and increase put value."""

    DIV_YIELD = 0.04

    @property
    def div_curve(self) -> DiscountCurve:
        return flat_curve(PRICING_DATE, _FD_MATURITY, self.DIV_YIELD)

    def test_dividends_reduce_asian_call(self):
        base = _fd_asian_pv(OptionType.CALL)
        with_div = _fd_asian_pv(OptionType.CALL, dividend_curve=self.div_curve)
        assert with_div < base, (
            f"Dividend call ({with_div:.4f}) should be < no-div call ({base:.4f})"
        )

    def test_dividends_increase_asian_put(self):
        base = _fd_asian_pv(OptionType.PUT)
        with_div = _fd_asian_pv(OptionType.PUT, dividend_curve=self.div_curve)
        assert with_div > base, f"Dividend put ({with_div:.4f}) should be > no-div put ({base:.4f})"

    def test_dividends_reduce_geometric_call(self):
        base = _fd_asian_pv(OptionType.CALL, averaging=AsianAveraging.GEOMETRIC)
        with_div = _fd_asian_pv(
            OptionType.CALL,
            averaging=AsianAveraging.GEOMETRIC,
            dividend_curve=self.div_curve,
        )
        assert with_div < base

    def test_dividends_increase_geometric_put(self):
        base = _fd_asian_pv(OptionType.PUT, averaging=AsianAveraging.GEOMETRIC)
        with_div = _fd_asian_pv(
            OptionType.PUT,
            averaging=AsianAveraging.GEOMETRIC,
            dividend_curve=self.div_curve,
        )
        assert with_div > base


class TestAmericanAsianPremium:
    """American Asians should have non-negative early-exercise premium."""

    @pytest.mark.parametrize("call_put", [OptionType.CALL, OptionType.PUT])
    @pytest.mark.parametrize("averaging", [AsianAveraging.ARITHMETIC, AsianAveraging.GEOMETRIC])
    def test_american_geq_european_with_fixing_dates(self, call_put, averaging):
        """American Asian PV >= European Asian PV (using fixing dates)."""
        euro = _fd_asian_pv(call_put, averaging, ExerciseType.EUROPEAN)
        amer = _fd_asian_pv(call_put, averaging, ExerciseType.AMERICAN)
        assert amer >= euro - 1e-6, (
            f"American ({amer:.6f}) < European ({euro:.6f}) for {averaging.value} {call_put.value}"
        )


class TestFixingDatesValidation:
    """Validation rules for fixing_dates on AsianOptionSpec."""

    def test_empty_fixing_dates_raises(self):
        with pytest.raises(ValidationError, match="non-empty"):
            AsianOptionSpec(
                averaging=AsianAveraging.ARITHMETIC,
                call_put=OptionType.CALL,
                strike=100.0,
                maturity=_FD_MATURITY,
                fixing_dates=(),
            )

    def test_unsorted_fixing_dates_raises(self):
        with pytest.raises(ValidationError, match="ascending"):
            AsianOptionSpec(
                averaging=AsianAveraging.ARITHMETIC,
                call_put=OptionType.CALL,
                strike=100.0,
                maturity=_FD_MATURITY,
                fixing_dates=(
                    dt.datetime(2025, 6, 1),
                    dt.datetime(2025, 3, 1),
                ),
            )

    def test_fixing_dates_beyond_maturity_raises(self):
        with pytest.raises(ValidationError, match="maturity"):
            AsianOptionSpec(
                averaging=AsianAveraging.ARITHMETIC,
                call_put=OptionType.CALL,
                strike=100.0,
                maturity=_FD_MATURITY,
                fixing_dates=(
                    dt.datetime(2025, 6, 1),
                    dt.datetime(2026, 6, 1),  # past maturity
                ),
            )

    def test_fixing_dates_before_averaging_start_raises(self):
        with pytest.raises(ValidationError, match="averaging_start"):
            AsianOptionSpec(
                averaging=AsianAveraging.ARITHMETIC,
                call_put=OptionType.CALL,
                strike=100.0,
                maturity=_FD_MATURITY,
                averaging_start=dt.datetime(2025, 4, 1),
                fixing_dates=(
                    dt.datetime(2025, 3, 1),  # before averaging_start
                    dt.datetime(2025, 6, 1),
                ),
            )

    def test_valid_fixing_dates_accepted(self):
        """No error raised for well-formed fixing dates."""
        spec = AsianOptionSpec(
            averaging=AsianAveraging.ARITHMETIC,
            call_put=OptionType.CALL,
            strike=100.0,
            maturity=_FD_MATURITY,
            fixing_dates=_MONTHLY_FIXINGS,
        )
        assert spec.fixing_dates == _MONTHLY_FIXINGS


# ---------------------------------------------------------------------------
# N=1 should reproduce vanilla BSM exactly
# ---------------------------------------------------------------------------


class TestSmallObservationCounts:
    """With N=1 the average is over {S₀, S(T)} (2 prices).

    Averaging reduces effective variance, so the Asian price is strictly
    less than the vanilla BSM price for both calls and puts.
    """

    @pytest.mark.parametrize(
        "spot,strike,vol,r,q,days,call_put",
        [
            (100, 100, 0.20, 0.05, 0.00, 365, OptionType.CALL),
            (100, 100, 0.20, 0.05, 0.00, 365, OptionType.PUT),
            (110, 90, 0.30, 0.03, 0.02, 180, OptionType.CALL),
            (90, 110, 0.25, 0.08, 0.01, 540, OptionType.PUT),
        ],
    )
    def test_n1_less_than_bsm(self, spot, strike, vol, r, q, days, call_put):
        maturity = PRICING_DATE + dt.timedelta(days=days)
        und = _underlying(
            spot=spot,
            vol=vol,
            short_rate=r,
            maturity=maturity,
            dividend_curve=_flat_dividend_curve(q, maturity),
        )

        asian_pv = OptionValuation(
            und,
            _asian_spec(
                strike=strike,
                maturity=maturity,
                call_put=call_put,
                num_steps=1,
                averaging=AsianAveraging.GEOMETRIC,
            ),
            PricingMethod.BSM,
        ).present_value()

        vanilla_spec = OptionSpec(
            option_type=call_put,
            exercise_type=ExerciseType.EUROPEAN,
            strike=strike,
            maturity=maturity,
            currency=CURRENCY,
        )
        bsm_pv = OptionValuation(
            und,
            vanilla_spec,
            PricingMethod.BSM,
        ).present_value()

        assert asian_pv < bsm_pv, f"N=1 Asian={asian_pv:.8f} should be < BSM={bsm_pv:.8f}"
        assert asian_pv > 0.0


# ---------------------------------------------------------------------------
# Put-call parity for geometric Asians
# ---------------------------------------------------------------------------


class TestGeometricAsianPutCallParity:
    """For European geometric Asians: C - P = e^{-rT}(E[G] - K)."""

    @pytest.mark.parametrize(
        "spot,strike,vol,r,q,days,num_obs",
        [
            (100, 100, 0.20, 0.05, 0.00, 365, 12),
            (100, 100, 0.20, 0.05, 0.02, 365, 52),
            (110, 90, 0.30, 0.03, 0.00, 180, 6),
            (90, 110, 0.25, 0.08, 0.01, 540, 30),
        ],
    )
    def test_put_call_parity(self, spot, strike, vol, r, q, days, num_obs):
        maturity = PRICING_DATE + dt.timedelta(days=days)
        und = _underlying(
            spot=spot,
            vol=vol,
            short_rate=r,
            maturity=maturity,
            dividend_curve=_flat_dividend_curve(q, maturity),
        )

        call_pv = OptionValuation(
            und,
            _asian_spec(
                strike=strike,
                maturity=maturity,
                call_put=OptionType.CALL,
                num_steps=num_obs,
                averaging=AsianAveraging.GEOMETRIC,
            ),
            PricingMethod.BSM,
        ).present_value()

        put_pv = OptionValuation(
            und,
            _asian_spec(
                strike=strike,
                maturity=maturity,
                call_put=OptionType.PUT,
                num_steps=num_obs,
                averaging=AsianAveraging.GEOMETRIC,
            ),
            PricingMethod.BSM,
        ).present_value()

        # E[G] via the formula internals (N intervals → M=N+1 prices)
        T = days / 365.0
        N = num_obs
        M = N + 1
        delta = T / N
        t_bar = N * delta / 2.0
        M1 = np.log(spot) + (r - q - 0.5 * vol**2) * t_bar
        M2 = vol**2 * (delta * N * (2 * N + 1) / (6.0 * M))
        F_G = np.exp(M1 + 0.5 * M2)
        df = np.exp(-r * T)

        parity_rhs = df * (F_G - strike)
        assert np.isclose(call_pv - put_pv, parity_rhs, atol=1e-10), (
            f"C-P={call_pv - put_pv:.10f} vs df*(F_G-K)={parity_rhs:.10f}"
        )


# ---------------------------------------------------------------------------
# Analytical vs Monte Carlo convergence
# ---------------------------------------------------------------------------


class TestAnalyticalVsMC:
    """Analytical geometric Asian should match MC geometric within sampling noise."""

    @pytest.mark.parametrize(
        "spot,strike,vol,r,q,days,call_put",
        [
            (100, 100, 0.20, 0.05, 0.00, 365, OptionType.CALL),
            (100, 100, 0.20, 0.05, 0.00, 365, OptionType.PUT),
            (110, 100, 0.25, 0.03, 0.00, 270, OptionType.CALL),
            (100, 100, 0.20, 0.05, 0.02, 365, OptionType.CALL),
            (95, 90, 0.30, 0.05, 0.01, 540, OptionType.PUT),
        ],
    )
    def test_analytical_close_to_mc(self, spot, strike, vol, r, q, days, call_put):
        maturity = PRICING_DATE + dt.timedelta(days=days)
        num_steps = 60

        # Analytical
        und = _underlying(
            spot=spot,
            vol=vol,
            short_rate=r,
            maturity=maturity,
            dividend_curve=_flat_dividend_curve(q, maturity),
        )
        analytical_pv = OptionValuation(
            und,
            _asian_spec(
                strike=strike,
                maturity=maturity,
                call_put=call_put,
                num_steps=num_steps,
                averaging=AsianAveraging.GEOMETRIC,
            ),
            PricingMethod.BSM,
        ).present_value()

        # MC geometric (same number of steps so observations match approximately)
        mc_und = _gbm_underlying(
            spot=spot,
            vol=vol,
            short_rate=r,
            maturity=maturity,
            paths=300_000,
            num_steps=num_steps,
            dividend_curve=_flat_dividend_curve(q, maturity),
        )
        mc_pv = OptionValuation(
            mc_und,
            _asian_spec(
                strike=strike,
                maturity=maturity,
                call_put=call_put,
                averaging=AsianAveraging.GEOMETRIC,
            ),
            PricingMethod.MONTE_CARLO,
            params=MonteCarloParams(random_seed=42),
        ).present_value()

        logger.info(
            "Geom Asian %s S=%.0f K=%.0f analytical=%.6f MC=%.6f",
            call_put.value,
            spot,
            strike,
            analytical_pv,
            mc_pv,
        )
        # Both analytical and MC include S₀ in the average (N+1 prices).
        # Tolerance absorbs MC sampling noise.
        assert np.isclose(analytical_pv, mc_pv, rtol=0.02), (
            f"analytical={analytical_pv:.6f} MC={mc_pv:.6f}"
        )


# ---------------------------------------------------------------------------
# Monotonicity and ordering properties
# ---------------------------------------------------------------------------


class TestGeometricAsianProperties:
    """Sanity checks on the analytical price."""

    def test_positive_price(self):
        maturity = PRICING_DATE + dt.timedelta(days=365)
        und = _underlying(spot=100, vol=0.2, short_rate=0.05, maturity=maturity)
        pv = OptionValuation(
            und,
            _asian_spec(
                strike=100,
                maturity=maturity,
                call_put=OptionType.CALL,
                num_steps=12,
                averaging=AsianAveraging.GEOMETRIC,
            ),
            PricingMethod.BSM,
        ).present_value()
        assert pv > 0.0

    def test_call_increases_with_spot(self):
        maturity = PRICING_DATE + dt.timedelta(days=365)
        pvs = []
        for spot in (90, 100, 110):
            und = _underlying(spot=spot, vol=0.2, short_rate=0.05, maturity=maturity)
            pv = OptionValuation(
                und,
                _asian_spec(
                    strike=100,
                    maturity=maturity,
                    call_put=OptionType.CALL,
                    num_steps=12,
                    averaging=AsianAveraging.GEOMETRIC,
                ),
                PricingMethod.BSM,
            ).present_value()
            pvs.append(pv)
        assert pvs[0] < pvs[1] < pvs[2]

    def test_put_decreases_with_spot(self):
        maturity = PRICING_DATE + dt.timedelta(days=365)
        pvs = []
        for spot in (90, 100, 110):
            und = _underlying(spot=spot, vol=0.2, short_rate=0.05, maturity=maturity)
            pv = OptionValuation(
                und,
                _asian_spec(
                    strike=100,
                    maturity=maturity,
                    call_put=OptionType.PUT,
                    num_steps=12,
                    averaging=AsianAveraging.GEOMETRIC,
                ),
                PricingMethod.BSM,
            ).present_value()
            pvs.append(pv)
        assert pvs[0] > pvs[1] > pvs[2]

    def test_more_steps_increases_effective_variance(self):
        """With S₀ included, M₂ = σ²T·(2N+1)/(6(N+1)) is increasing in N.

        The known S₀ observation contributes zero variance; adding more
        future observations dilutes its weight, raising the effective vol
        of the geometric average toward σ/√3.  ATM call price therefore
        increases with the number of steps.
        """
        maturity = PRICING_DATE + dt.timedelta(days=365)
        und = _underlying(spot=100, vol=0.3, short_rate=0.05, maturity=maturity)
        pv_4 = OptionValuation(
            und,
            _asian_spec(
                strike=100,
                maturity=maturity,
                call_put=OptionType.CALL,
                num_steps=4,
                averaging=AsianAveraging.GEOMETRIC,
            ),
            PricingMethod.BSM,
        ).present_value()
        pv_252 = OptionValuation(
            und,
            _asian_spec(
                strike=100,
                maturity=maturity,
                call_put=OptionType.CALL,
                num_steps=252,
                averaging=AsianAveraging.GEOMETRIC,
            ),
            PricingMethod.BSM,
        ).present_value()
        assert pv_252 > pv_4

    def test_geometric_call_leq_vanilla_bsm(self):
        """Geometric average call ≤ vanilla European call (averaging reduces variance)."""
        maturity = PRICING_DATE + dt.timedelta(days=365)
        und = _underlying(spot=100, vol=0.25, short_rate=0.05, maturity=maturity)

        geom_pv = OptionValuation(
            und,
            _asian_spec(
                strike=100,
                maturity=maturity,
                call_put=OptionType.CALL,
                num_steps=52,
                averaging=AsianAveraging.GEOMETRIC,
            ),
            PricingMethod.BSM,
        ).present_value()

        vanilla_spec = OptionSpec(
            option_type=OptionType.CALL,
            exercise_type=ExerciseType.EUROPEAN,
            strike=100,
            maturity=maturity,
            currency=CURRENCY,
        )
        vanilla_pv = OptionValuation(
            und,
            vanilla_spec,
            PricingMethod.BSM,
        ).present_value()

        assert geom_pv < vanilla_pv

    def test_geometric_put_leq_vanilla_bsm(self):
        """Geometric average put ≤ vanilla European put (averaging reduces variance)."""
        maturity = PRICING_DATE + dt.timedelta(days=365)
        und = _underlying(spot=100, vol=0.25, short_rate=0.05, maturity=maturity)

        geom_pv = OptionValuation(
            und,
            _asian_spec(
                strike=100,
                maturity=maturity,
                call_put=OptionType.PUT,
                num_steps=52,
                averaging=AsianAveraging.GEOMETRIC,
            ),
            PricingMethod.BSM,
        ).present_value()

        vanilla_spec = OptionSpec(
            option_type=OptionType.PUT,
            exercise_type=ExerciseType.EUROPEAN,
            strike=100,
            maturity=maturity,
            currency=CURRENCY,
        )
        vanilla_pv = OptionValuation(
            und,
            vanilla_spec,
            PricingMethod.BSM,
        ).present_value()

        assert geom_pv < vanilla_pv


# ---------------------------------------------------------------------------
# Continuous limit: N → ∞ gives σ_a = σ/√3
# ---------------------------------------------------------------------------


class TestContinuousLimit:
    """For large N the discrete formula should approach the continuous result."""

    def test_large_n_approaches_continuous(self):
        T, S0, K, sigma, r, q = 1.0, 100.0, 100.0, 0.25, 0.05, 0.02

        # Continuous closed form: use σ_a = σ/√3, adjusted growth rate
        t_bar_cont = T / 2.0
        M1_cont = np.log(S0) + (r - q - 0.5 * sigma**2) * t_bar_cont
        M2_cont = sigma**2 * T / 3.0
        F_G = np.exp(M1_cont + 0.5 * M2_cont)
        vol_sqrt = np.sqrt(M2_cont)
        d1 = (np.log(F_G / K) + 0.5 * M2_cont) / vol_sqrt
        d2 = d1 - vol_sqrt
        continuous_call = np.exp(-r * T) * (F_G * norm.cdf(d1) - K * norm.cdf(d2))

        # Discrete with N=10000
        discrete_call = _asian_geometric_analytical(
            spot=S0,
            strike=K,
            time_to_maturity=T,
            volatility=sigma,
            risk_free_rate=r,
            dividend_yield=q,
            option_type=OptionType.CALL,
            num_steps=10_000,
        )

        assert np.isclose(discrete_call, continuous_call, rtol=1e-4)


# ---------------------------------------------------------------------------
# Dividend yield support
# ---------------------------------------------------------------------------


class TestDividendYield:
    """Continuous dividend yield should reduce call value, increase put value."""

    def test_dividend_reduces_call(self):
        maturity = PRICING_DATE + dt.timedelta(days=365)
        pv_no_q = OptionValuation(
            _underlying(spot=100, vol=0.2, short_rate=0.05, maturity=maturity),
            _asian_spec(
                strike=100,
                maturity=maturity,
                call_put=OptionType.CALL,
                num_steps=12,
                averaging=AsianAveraging.GEOMETRIC,
            ),
            PricingMethod.BSM,
        ).present_value()

        pv_with_q = OptionValuation(
            _underlying(
                spot=100,
                vol=0.2,
                short_rate=0.05,
                maturity=maturity,
                dividend_curve=_flat_dividend_curve(0.03, maturity),
            ),
            _asian_spec(
                strike=100,
                maturity=maturity,
                call_put=OptionType.CALL,
                num_steps=12,
                averaging=AsianAveraging.GEOMETRIC,
            ),
            PricingMethod.BSM,
        ).present_value()

        assert pv_with_q < pv_no_q


# ---------------------------------------------------------------------------
# Seasoned Asian (observed_average / observed_count — Hull K* adjustment)
# ---------------------------------------------------------------------------


class TestSeasonedAsian:
    """Test Hull's K* strike-adjustment for seasoned Asian options.

    When n₁ fixings have already been observed with average S̄, the payoff is:

        max((n₁·S̄ + n₂·S_avg) / (n₁+n₂) − K, 0)

    This equals ``(n₂/(n₁+n₂)) · max(S_avg − K*, 0)`` with:

        K* = ((n₁+n₂)/n₂) · K  −  (n₁/n₂) · S̄

    See Hull, "Options, Futures, and Other Derivatives", Section 26.13.
    """

    SPOT = 50.0
    VOL = 0.40
    RATE = 0.10
    MATURITY = PRICING_DATE + dt.timedelta(days=365)

    def _ud(self) -> UnderlyingPricingData:
        return _underlying(
            spot=self.SPOT,
            vol=self.VOL,
            short_rate=self.RATE,
            maturity=self.MATURITY,
        )

    # ── Validation ────────────────────────────────────────────────────────

    def test_observed_average_requires_observed_count(self):
        with pytest.raises(Exception, match="observed_average and observed_count"):
            AsianOptionSpec(
                averaging=AsianAveraging.ARITHMETIC,
                call_put=OptionType.CALL,
                strike=50.0,
                maturity=self.MATURITY,
                currency=CURRENCY,
                num_steps=5,
                observed_average=52.0,
            )

    def test_observed_count_requires_observed_average(self):
        with pytest.raises(Exception, match="observed_average and observed_count"):
            AsianOptionSpec(
                averaging=AsianAveraging.ARITHMETIC,
                call_put=OptionType.CALL,
                strike=50.0,
                maturity=self.MATURITY,
                currency=CURRENCY,
                num_steps=5,
                observed_count=6,
            )

    def test_observed_average_must_be_positive(self):
        with pytest.raises(Exception, match="observed_average must be > 0"):
            AsianOptionSpec(
                averaging=AsianAveraging.ARITHMETIC,
                call_put=OptionType.CALL,
                strike=50.0,
                maturity=self.MATURITY,
                currency=CURRENCY,
                num_steps=5,
                observed_average=-1.0,
                observed_count=6,
            )

    def test_observed_count_must_be_positive_int(self):
        with pytest.raises(Exception, match="observed_count must be a positive integer"):
            AsianOptionSpec(
                averaging=AsianAveraging.ARITHMETIC,
                call_put=OptionType.CALL,
                strike=50.0,
                maturity=self.MATURITY,
                currency=CURRENCY,
                num_steps=5,
                observed_average=52.0,
                observed_count=0,
            )

    # ── K* > 0: reduces to fresh Asian ───────────────────────────────────

    @pytest.mark.parametrize("averaging", [AsianAveraging.ARITHMETIC, AsianAveraging.GEOMETRIC])
    @pytest.mark.parametrize("call_put", [OptionType.CALL, OptionType.PUT])
    def test_seasoned_matches_manual_k_star(self, averaging, call_put):
        """Seasoned PV should equal scale * fresh_PV(K=K*) when K* > 0."""
        n1, S_bar, K = 6, 52.0, 50.0
        n2_steps = 5  # n₂ = 6 future observations
        n2 = n2_steps + 1
        n_total = n1 + n2

        K_star = (n_total / n2) * K - (n1 / n2) * S_bar
        assert K_star > 0, "This test requires K* > 0"
        scale = n2 / n_total

        # Manual: price fresh Asian with K*
        fresh_spec = AsianOptionSpec(
            averaging=averaging,
            call_put=call_put,
            strike=K_star,
            maturity=self.MATURITY,
            currency=CURRENCY,
            num_steps=n2_steps,
        )
        fresh_pv = OptionValuation(
            self._ud(),
            fresh_spec,
            PricingMethod.BSM,
        ).present_value()

        # Library seasoned
        seasoned_spec = AsianOptionSpec(
            averaging=averaging,
            call_put=call_put,
            strike=K,
            maturity=self.MATURITY,
            currency=CURRENCY,
            num_steps=n2_steps,
            observed_average=S_bar,
            observed_count=n1,
        )
        seasoned_pv = OptionValuation(
            self._ud(),
            seasoned_spec,
            PricingMethod.BSM,
        ).present_value()

        assert np.isclose(seasoned_pv, scale * fresh_pv, rtol=1e-12)

    # ── K* <= 0: certain exercise (call → forward, put → 0) ─────────────

    def test_k_star_negative_call_is_forward(self):
        """When K* <= 0, the call is certain to be exercised."""
        n1, S_bar, K = 6, 120.0, 50.0
        n2_steps = 5
        n2 = n2_steps + 1
        n_total = n1 + n2
        K_star = (n_total / n2) * K - (n1 / n2) * S_bar
        assert K_star < 0, "This test requires K* < 0"

        seasoned_spec = AsianOptionSpec(
            averaging=AsianAveraging.ARITHMETIC,
            call_put=OptionType.CALL,
            strike=K,
            maturity=self.MATURITY,
            currency=CURRENCY,
            num_steps=n2_steps,
            observed_average=S_bar,
            observed_count=n1,
        )
        pv = OptionValuation(
            self._ud(),
            seasoned_spec,
            PricingMethod.BSM,
        ).present_value()

        # Must be strictly positive
        assert pv > 0.0

        # Manual: scale * (disc_M1 - K* * df)
        ttm = 1.0
        df = np.exp(-self.RATE * ttm)
        zero_spec = AsianOptionSpec(
            averaging=AsianAveraging.ARITHMETIC,
            call_put=OptionType.CALL,
            strike=0.0,
            maturity=self.MATURITY,
            currency=CURRENCY,
            num_steps=n2_steps,
        )
        disc_M1 = OptionValuation(
            self._ud(),
            zero_spec,
            PricingMethod.BSM,
        ).present_value()
        scale = n2 / n_total
        expected = scale * (disc_M1 - K_star * df)
        assert np.isclose(pv, expected, rtol=1e-12)

    def test_k_star_negative_put_is_zero(self):
        """When K* <= 0, a put is worthless (average > 0 > K*)."""
        seasoned_spec = AsianOptionSpec(
            averaging=AsianAveraging.ARITHMETIC,
            call_put=OptionType.PUT,
            strike=50.0,
            maturity=self.MATURITY,
            currency=CURRENCY,
            num_steps=5,
            observed_average=120.0,
            observed_count=6,
        )
        pv = OptionValuation(
            self._ud(),
            seasoned_spec,
            PricingMethod.BSM,
        ).present_value()

        assert pv == 0.0

    # ── Monotonicity and sanity ──────────────────────────────────────────

    def test_seasoned_less_than_fresh_when_atm(self):
        """A seasoned ATM call (S̄ = S₀ = K) should be worth less than a fresh one.

        With S̄ = K, some of the averaging period has elapsed at-the-money,
        leaving the remaining average with less optionality.
        """
        K = self.SPOT  # ATM
        n2_steps = 5
        n1 = 6

        fresh_spec = AsianOptionSpec(
            averaging=AsianAveraging.ARITHMETIC,
            call_put=OptionType.CALL,
            strike=K,
            maturity=self.MATURITY,
            currency=CURRENCY,
            num_steps=n2_steps,
        )
        fresh_pv = OptionValuation(
            self._ud(),
            fresh_spec,
            PricingMethod.BSM,
        ).present_value()

        seasoned_spec = AsianOptionSpec(
            averaging=AsianAveraging.ARITHMETIC,
            call_put=OptionType.CALL,
            strike=K,
            maturity=self.MATURITY,
            currency=CURRENCY,
            num_steps=n2_steps,
            observed_average=K,  # ATM: S̄ = K
            observed_count=n1,
        )
        seasoned_pv = OptionValuation(
            self._ud(),
            seasoned_spec,
            PricingMethod.BSM,
        ).present_value()

        assert seasoned_pv < fresh_pv

    def test_higher_observed_average_increases_call_value(self):
        """A higher observed average should increase the seasoned call value."""
        n2_steps = 5
        n1 = 6

        def _seasoned_call(s_bar: float) -> float:
            spec = AsianOptionSpec(
                averaging=AsianAveraging.ARITHMETIC,
                call_put=OptionType.CALL,
                strike=self.SPOT,
                maturity=self.MATURITY,
                currency=CURRENCY,
                num_steps=n2_steps,
                observed_average=s_bar,
                observed_count=n1,
            )
            return OptionValuation(
                self._ud(),
                spec,
                PricingMethod.BSM,
            ).present_value()

        assert _seasoned_call(55.0) > _seasoned_call(50.0) > _seasoned_call(45.0)

    def test_higher_observed_average_decreases_put_value(self):
        """A higher observed average should decrease the seasoned put value."""
        n2_steps = 5
        n1 = 6

        def _seasoned_put(s_bar: float) -> float:
            spec = AsianOptionSpec(
                averaging=AsianAveraging.ARITHMETIC,
                call_put=OptionType.PUT,
                strike=self.SPOT,
                maturity=self.MATURITY,
                currency=CURRENCY,
                num_steps=n2_steps,
                observed_average=s_bar,
                observed_count=n1,
            )
            return OptionValuation(
                self._ud(),
                spec,
                PricingMethod.BSM,
            ).present_value()

        assert _seasoned_put(45.0) > _seasoned_put(50.0) > _seasoned_put(55.0)

    def test_more_observed_reduces_optionality(self):
        """More past observations (with S̄ = K) → less optionality → lower call value."""
        n2_steps = 5
        K = self.SPOT

        def _seasoned_call(n1: int) -> float:
            spec = AsianOptionSpec(
                averaging=AsianAveraging.ARITHMETIC,
                call_put=OptionType.CALL,
                strike=K,
                maturity=self.MATURITY,
                currency=CURRENCY,
                num_steps=n2_steps,
                observed_average=K,
                observed_count=n1,
            )
            return OptionValuation(
                self._ud(),
                spec,
                PricingMethod.BSM,
            ).present_value()

        # More past ATM observations → the average is "pinned" closer to K
        assert _seasoned_call(1) > _seasoned_call(6) > _seasoned_call(50)

    # ── Cross-engine consistency (Binomial / MC) ─────────────────────────

    def test_binomial_seasoned_matches_analytical(self):
        """Binomial Hull tree seasoned Asian should converge to analytical."""
        n1, S_bar, K = 6, 52.0, 50.0
        n2_steps = 60

        seasoned_spec = AsianOptionSpec(
            averaging=AsianAveraging.ARITHMETIC,
            call_put=OptionType.CALL,
            strike=K,
            maturity=self.MATURITY,
            currency=CURRENCY,
            num_steps=n2_steps,
            observed_average=S_bar,
            observed_count=n1,
        )

        bsm_pv = OptionValuation(
            self._ud(),
            seasoned_spec,
            PricingMethod.BSM,
        ).present_value()

        binom_pv = OptionValuation(
            self._ud(),
            seasoned_spec,
            PricingMethod.BINOMIAL,
            params=BinomialParams(num_steps=n2_steps, asian_tree_averages=100),
        ).present_value()

        assert np.isclose(binom_pv, bsm_pv, rtol=0.01), (
            f"Binomial={binom_pv:.6f} vs BSM={bsm_pv:.6f}"
        )

    def test_mc_seasoned_matches_analytical(self):
        """MC seasoned Asian should converge to analytical."""
        n1, S_bar, K = 6, 52.0, 50.0
        n2_steps = 60

        seasoned_spec = AsianOptionSpec(
            averaging=AsianAveraging.ARITHMETIC,
            call_put=OptionType.CALL,
            strike=K,
            maturity=self.MATURITY,
            currency=CURRENCY,
            num_steps=n2_steps,
            observed_average=S_bar,
            observed_count=n1,
        )

        bsm_pv = OptionValuation(
            self._ud(),
            seasoned_spec,
            PricingMethod.BSM,
        ).present_value()

        gbm = _gbm_underlying(
            spot=self.SPOT,
            vol=self.VOL,
            short_rate=self.RATE,
            maturity=self.MATURITY,
            paths=200_000,
            num_steps=n2_steps,
        )
        mc_pv = OptionValuation(
            gbm,
            seasoned_spec,
            PricingMethod.MONTE_CARLO,
            params=MonteCarloParams(random_seed=42),
        ).present_value()

        assert np.isclose(mc_pv, bsm_pv, rtol=0.01), f"MC={mc_pv:.6f} vs BSM={bsm_pv:.6f}"

    def test_dividend_increases_put(self):
        maturity = PRICING_DATE + dt.timedelta(days=365)
        pv_no_q = OptionValuation(
            _underlying(spot=100, vol=0.2, short_rate=0.05, maturity=maturity),
            _asian_spec(
                strike=100,
                maturity=maturity,
                call_put=OptionType.PUT,
                num_steps=12,
                averaging=AsianAveraging.GEOMETRIC,
            ),
            PricingMethod.BSM,
        ).present_value()

        pv_with_q = OptionValuation(
            _underlying(
                spot=100,
                vol=0.2,
                short_rate=0.05,
                maturity=maturity,
                dividend_curve=_flat_dividend_curve(0.03, maturity),
            ),
            _asian_spec(
                strike=100,
                maturity=maturity,
                call_put=OptionType.PUT,
                num_steps=12,
                averaging=AsianAveraging.GEOMETRIC,
            ),
            PricingMethod.BSM,
        ).present_value()

        assert pv_with_q > pv_no_q


# ---------------------------------------------------------------------------
# Validation / error paths
# ---------------------------------------------------------------------------


class TestValidation:
    """Error handling for analytical Asian pricing."""

    def test_arithmetic_bsm_returns_positive_price(self):
        """Arithmetic averaging is now supported via Turnbull-Wakeman."""
        maturity = PRICING_DATE + dt.timedelta(days=365)
        und = _underlying(spot=100, vol=0.2, short_rate=0.05, maturity=maturity)
        pv = OptionValuation(
            und,
            _asian_spec(
                strike=100,
                maturity=maturity,
                call_put=OptionType.CALL,
                num_steps=12,
                averaging=AsianAveraging.ARITHMETIC,
            ),
            PricingMethod.BSM,
        ).present_value()
        assert pv > 0.0

    def test_missing_num_steps_raises(self):
        maturity = PRICING_DATE + dt.timedelta(days=365)
        und = _underlying(spot=100, vol=0.2, short_rate=0.05, maturity=maturity)
        with pytest.raises(Exception, match="num_steps"):
            OptionValuation(
                und,
                _asian_spec(
                    strike=100,
                    maturity=maturity,
                    call_put=OptionType.CALL,
                    averaging=AsianAveraging.GEOMETRIC,
                ),
                PricingMethod.BSM,
            )

    def test_invalid_num_steps_on_spec(self):
        maturity = PRICING_DATE + dt.timedelta(days=365)
        with pytest.raises(Exception, match="num_steps"):
            AsianOptionSpec(
                averaging=AsianAveraging.GEOMETRIC,
                call_put=OptionType.CALL,
                strike=100,
                maturity=maturity,
                currency=CURRENCY,
                num_steps=0,
            )

    def test_pure_function_validation(self):
        with pytest.raises(Exception, match="time_to_maturity"):
            _asian_geometric_analytical(
                spot=100,
                strike=100,
                time_to_maturity=-1,
                volatility=0.2,
                risk_free_rate=0.05,
                dividend_yield=0.0,
                option_type=OptionType.CALL,
                num_steps=12,
            )
        with pytest.raises(Exception, match="volatility"):
            _asian_geometric_analytical(
                spot=100,
                strike=100,
                time_to_maturity=1,
                volatility=-0.2,
                risk_free_rate=0.05,
                dividend_yield=0.0,
                option_type=OptionType.CALL,
                num_steps=12,
            )
        with pytest.raises(Exception, match="num_steps"):
            _asian_geometric_analytical(
                spot=100,
                strike=100,
                time_to_maturity=1,
                volatility=0.2,
                risk_free_rate=0.05,
                dividend_yield=0.0,
                option_type=OptionType.CALL,
                num_steps=0,
            )


# ---------------------------------------------------------------------------
# Averaging start in the future
# ---------------------------------------------------------------------------


class TestAveragingStart:
    """When averaging_start is after pricing_date, observations span [t_s, T]."""

    def test_later_start_changes_price(self):
        """Pushing the averaging window forward should change the price."""
        maturity = PRICING_DATE + dt.timedelta(days=365)
        avg_start = PRICING_DATE + dt.timedelta(days=90)
        und = _underlying(spot=100, vol=0.2, short_rate=0.05, maturity=maturity)

        pv_full = OptionValuation(
            und,
            _asian_spec(
                strike=100,
                maturity=maturity,
                call_put=OptionType.CALL,
                num_steps=12,
                averaging=AsianAveraging.GEOMETRIC,
            ),
            PricingMethod.BSM,
        ).present_value()

        pv_late = OptionValuation(
            und,
            _asian_spec(
                strike=100,
                maturity=maturity,
                call_put=OptionType.CALL,
                num_steps=12,
                averaging_start=avg_start,
                averaging=AsianAveraging.GEOMETRIC,
            ),
            PricingMethod.BSM,
        ).present_value()

        # Shorter averaging window → less variance reduction → more expensive call
        assert pv_late > pv_full

    def test_averaging_start_put_call_parity(self):
        """Put-call parity holds even with a deferred averaging start."""
        maturity = PRICING_DATE + dt.timedelta(days=365)
        avg_start = PRICING_DATE + dt.timedelta(days=60)
        und = _underlying(spot=100, vol=0.2, short_rate=0.05, maturity=maturity)

        call_pv = OptionValuation(
            und,
            _asian_spec(
                strike=100,
                maturity=maturity,
                call_put=OptionType.CALL,
                num_steps=10,
                averaging_start=avg_start,
                averaging=AsianAveraging.GEOMETRIC,
            ),
            PricingMethod.BSM,
        ).present_value()

        put_pv = OptionValuation(
            und,
            _asian_spec(
                strike=100,
                maturity=maturity,
                call_put=OptionType.PUT,
                num_steps=10,
                averaging_start=avg_start,
                averaging=AsianAveraging.GEOMETRIC,
            ),
            PricingMethod.BSM,
        ).present_value()

        # Compute E[G] (N intervals → M=N+1 prices)
        T = 365.0 / 365.0
        N = 10
        M = N + 1
        r, q, sigma = 0.05, 0.0, 0.2
        t_s = 60.0 / 365.0
        delta = (T - t_s) / N
        t_bar = t_s + N * delta / 2.0
        M1 = np.log(100) + (r - q - 0.5 * sigma**2) * t_bar
        M2 = sigma**2 * (t_s + delta * N * (2 * N + 1) / (6.0 * M))
        F_G = np.exp(M1 + 0.5 * M2)
        df = np.exp(-r * T)

        assert np.isclose(call_pv - put_pv, df * (F_G - 100), atol=1e-10)


# ---------------------------------------------------------------------------
# 4-method comparison: Analytical vs Binomial-MC vs Binomial-Hull vs MC
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "spot,strike,vol,r,q,days,call_put",
    [
        (100, 100, 0.20, 0.05, 0.00, 365, OptionType.CALL),
        (100, 100, 0.20, 0.05, 0.00, 365, OptionType.PUT),
        (50, 50, 0.40, 0.10, 0.00, 365, OptionType.CALL),  # Hull Example 26.3
        (110, 100, 0.25, 0.03, 0.02, 270, OptionType.CALL),
        (95, 90, 0.30, 0.05, 0.01, 540, OptionType.PUT),
        (105, 110, 0.18, 0.02, 0.00, 180, OptionType.PUT),
    ],
)
def test_geometric_asian_four_method_comparison(
    spot,
    strike,
    vol,
    r,
    q,
    days,
    call_put,
):
    """Compare geometric Asian European prices across 4 methods.

    1. Analytical (BSM) — Kemna-Vorst closed-form
    2. Binomial MC — binomial tree with MC path sampling for Asian payoff
    3. Binomial Hull — Hull's tree-average interpolation method
    4. Monte Carlo — direct GBM path simulation

    This is a diagnostic/comparison test; it logs all four prices
    and asserts they are broadly consistent (within MC noise).
    """
    maturity = PRICING_DATE + dt.timedelta(days=days)
    # q_curve = (
    #     None if q == 0.0
    #     else flat_curve(PRICING_DATE, maturity, q)
    # )

    # --- 1. Analytical (BSM) ---
    und_determ = _underlying(
        spot=spot,
        vol=vol,
        short_rate=r,
        maturity=maturity,
        dividend_curve=_flat_dividend_curve(q, maturity),
    )
    analytical_pv = OptionValuation(
        und_determ,
        _asian_spec(
            strike=strike,
            maturity=maturity,
            call_put=call_put,
            num_steps=NUM_STEPS,
            averaging=AsianAveraging.GEOMETRIC,
        ),
        PricingMethod.BSM,
    ).present_value()

    # --- 2. Binomial MC ---
    binom_mc_pv = OptionValuation(
        und_determ,
        _asian_spec(
            strike=strike,
            maturity=maturity,
            call_put=call_put,
            averaging=AsianAveraging.GEOMETRIC,
        ),
        PricingMethod.BINOMIAL,
        params=BinomialParams(
            num_steps=NUM_STEPS * 2,
            mc_paths=MC_PATHS,
            random_seed=MC_SEED,
        ),
    ).present_value()

    # --- 3. Binomial Hull tree averages ---
    hull_pv = OptionValuation(
        und_determ,
        _asian_spec(
            strike=strike,
            maturity=maturity,
            call_put=call_put,
            averaging=AsianAveraging.GEOMETRIC,
        ),
        PricingMethod.BINOMIAL,
        params=BinomialParams(
            num_steps=BINOM_STEPS,
            asian_tree_averages=ASIAN_TREE_AVERAGES,
        ),
    ).present_value()

    # --- 4. Monte Carlo ---
    mc_und = _gbm_underlying(
        spot=spot,
        vol=vol,
        short_rate=r,
        maturity=maturity,
        paths=MC_PATHS,
        num_steps=NUM_STEPS,
        dividend_curve=_flat_dividend_curve(q, maturity),
    )
    mc_pv = OptionValuation(
        mc_und,
        _asian_spec(
            strike=strike,
            maturity=maturity,
            call_put=call_put,
            averaging=AsianAveraging.GEOMETRIC,
        ),
        PricingMethod.MONTE_CARLO,
        params=MonteCarloParams(random_seed=MC_SEED),
    ).present_value()

    logger.info(
        "Geometric Asian %s S=%.0f K=%.0f vol=%.2f r=%.2f q=%.2f days=%d\n"
        "  Analytical=%.6f  BinomMC=%.6f  Hull=%.6f  MC=%.6f",
        call_put.value,
        spot,
        strike,
        vol,
        r,
        q,
        days,
        analytical_pv,
        binom_mc_pv,
        hull_pv,
        mc_pv,
    )

    # Broad consistency: all four should be within 5% of each other
    prices = [analytical_pv, binom_mc_pv, hull_pv, mc_pv]
    mid = np.mean(prices)
    for label, pv in zip(["Analytical", "BinomMC", "Hull", "MC"], prices):
        assert np.isclose(pv, mid, rtol=0.02), f"{label}={pv:.6f} deviates from mean={mid:.6f}"


# ===========================================================================
# ARITHMETIC AVERAGE — Turnbull-Wakeman / Hull §26.13
# ===========================================================================


# ---------------------------------------------------------------------------
# Hull Example 26.3 verification
# ---------------------------------------------------------------------------


class TestHullExample26_3:
    """Verify against Hull Example 26.3 (pp. 626-627).

    Hull's example uses observations at T/m, 2T/m, ..., T (i.e. does NOT
    include S₀ at time 0).  Our convention includes S₀, so our "num_steps=m"
    gives m+1 observations.  We test both the raw moments against Hull's
    numbers (using his convention) and the pipeline with our S₀ convention.
    """

    def test_hull_continuous_moments(self):
        """Verify M₁, M₂ formulas against Hull's continuous-average closed forms.

        Hull continuous formulas (r=0.1, q=0, σ=0.4, T=1, S₀=50):
            M₁ = [exp((r-q)T) - 1] / [(r-q)T] · S₀ = 52.59
            M₂ = 2922.76 (from equation 26.4)
        """
        S0, r, q, sigma, T = 50.0, 0.1, 0.0, 0.4, 1.0

        # Hull's continuous M₁
        hull_M1 = (np.exp((r - q) * T) - 1) / ((r - q) * T) * S0

        # Hull's continuous M₂ (equation 26.4)
        hull_M2 = 2 * np.exp((2 * (r - q) + sigma**2) * T) * S0**2 / (
            (r - q + sigma**2) * (2 * (r - q) + sigma**2) * T**2
        ) + 2 * S0**2 / ((r - q) * T**2) * (
            1 / (2 * (r - q) + sigma**2) - np.exp((r - q) * T) / (r - q + sigma**2)
        )

        assert np.isclose(hull_M1, 52.59, atol=0.01), f"M1={hull_M1}"
        assert np.isclose(hull_M2, 2922.76, rtol=5e-4), f"M2={hull_M2}"

        # Verify our discrete formula approaches the continuous limit
        N_large = 10_000
        M = N_large + 1
        delta = T / N_large
        t = np.arange(M, dtype=float) * delta
        F = S0 * np.exp((r - q) * t)
        our_M1 = np.mean(F)
        F_cumrev = np.cumsum(F[::-1])[::-1]
        our_M2 = np.sum(F * np.exp(sigma**2 * t) * (2 * F_cumrev - F)) / M**2

        assert np.isclose(our_M1, hull_M1, rtol=1e-4), f"our M1={our_M1} vs hull {hull_M1}"
        assert np.isclose(our_M2, hull_M2, rtol=1e-3), f"our M2={our_M2} vs hull {hull_M2}"

    def test_hull_continuous_price(self):
        """Hull continuous average price: 5.62 for a call."""
        S0, K, r, q, sigma, T = 50.0, 50.0, 0.1, 0.0, 0.4, 1.0

        # Large N approximates continuous average
        price = _asian_arithmetic_analytical(
            spot=S0,
            strike=K,
            time_to_maturity=T,
            volatility=sigma,
            risk_free_rate=r,
            dividend_yield=q,
            option_type=OptionType.CALL,
            num_steps=10_000,
        )
        assert np.isclose(price, 5.62, atol=0.02), f"price={price:.4f} expected ~5.62"

    def test_hull_discrete_observations(self):
        """Hull's discrete prices: 12 obs → 6.00, 52 obs → 5.70, 250 obs → 5.63.

        Hull uses m observations at T/m, 2T/m, ..., T (no S₀).
        Our pricer with ``averaging_start = T/m`` and ``num_steps = m - 1``
        places M = m observations at exactly those times, matching Hull.
        """
        S0, K, r, q, sigma, T = 50.0, 50.0, 0.1, 0.0, 0.4, 1.0

        for m, hull_price in [(12, 6.00), (52, 5.70), (250, 5.63)]:
            # averaging_start = T/m skips S₀, giving m obs at T/m, ..., T
            price = _asian_arithmetic_analytical(
                spot=S0,
                strike=K,
                time_to_maturity=T,
                volatility=sigma,
                risk_free_rate=r,
                dividend_yield=q,
                option_type=OptionType.CALL,
                num_steps=m - 1,
                averaging_start=T / m,
            )

            logger.info(
                "Hull 26.3: m=%d price=%.4f (expected %.2f)",
                m,
                price,
                hull_price,
            )
            assert np.isclose(price, hull_price, atol=0.02), (
                f"m={m}: price={price:.4f} expected {hull_price:.2f}"
            )


# ---------------------------------------------------------------------------
# Arithmetic: put-call parity
# ---------------------------------------------------------------------------


class TestArithmeticPutCallParity:
    """For European arithmetic Asians: C - P = e^{-rT}(M₁ - K)."""

    @pytest.mark.parametrize(
        "spot,strike,vol,r,q,days,num_steps",
        [
            (100, 100, 0.20, 0.05, 0.00, 365, 12),
            (100, 100, 0.25, 0.05, 0.02, 365, 52),
            (50, 50, 0.40, 0.10, 0.00, 365, 30),
            (110, 90, 0.30, 0.03, 0.00, 180, 6),
        ],
    )
    def test_put_call_parity(self, spot, strike, vol, r, q, days, num_steps):
        maturity = PRICING_DATE + dt.timedelta(days=days)
        und = _underlying(
            spot=spot,
            vol=vol,
            short_rate=r,
            maturity=maturity,
            dividend_curve=_flat_dividend_curve(q, maturity),
        )

        call_pv = OptionValuation(
            und,
            _asian_spec(
                strike=strike,
                maturity=maturity,
                call_put=OptionType.CALL,
                num_steps=num_steps,
                averaging=AsianAveraging.ARITHMETIC,
            ),
            PricingMethod.BSM,
        ).present_value()

        put_pv = OptionValuation(
            und,
            _asian_spec(
                strike=strike,
                maturity=maturity,
                call_put=OptionType.PUT,
                num_steps=num_steps,
                averaging=AsianAveraging.ARITHMETIC,
            ),
            PricingMethod.BSM,
        ).present_value()

        # E[S_avg] = M₁ via forward prices
        T = days / 365.0
        N = num_steps
        M = N + 1
        delta = T / N
        t = np.arange(M, dtype=float) * delta
        F = spot * np.exp((r - q) * t)
        M1 = np.mean(F)
        df = np.exp(-r * T)

        parity_rhs = df * (M1 - strike)
        assert np.isclose(call_pv - put_pv, parity_rhs, atol=1e-10), (
            f"C-P={call_pv - put_pv:.10f} vs df*(M1-K)={parity_rhs:.10f}"
        )


# ---------------------------------------------------------------------------
# Arithmetic vs MC convergence
# ---------------------------------------------------------------------------


class TestArithmeticVsMC:
    """Turnbull-Wakeman approximation should be close to MC arithmetic average."""

    @pytest.mark.parametrize(
        "spot,strike,vol,r,q,days,call_put",
        [
            (100, 100, 0.20, 0.05, 0.00, 365, OptionType.CALL),
            (100, 100, 0.20, 0.05, 0.00, 365, OptionType.PUT),
            (50, 50, 0.40, 0.10, 0.00, 365, OptionType.CALL),
            (110, 100, 0.25, 0.03, 0.02, 270, OptionType.CALL),
            (95, 90, 0.30, 0.05, 0.01, 540, OptionType.PUT),
        ],
    )
    def test_analytical_close_to_mc(self, spot, strike, vol, r, q, days, call_put):
        maturity = PRICING_DATE + dt.timedelta(days=days)
        num_steps = 60

        # Turnbull-Wakeman analytical
        und = _underlying(
            spot=spot,
            vol=vol,
            short_rate=r,
            maturity=maturity,
            dividend_curve=_flat_dividend_curve(q, maturity),
        )
        analytical_pv = OptionValuation(
            und,
            _asian_spec(
                strike=strike,
                maturity=maturity,
                call_put=call_put,
                num_steps=num_steps,
                averaging=AsianAveraging.ARITHMETIC,
            ),
            PricingMethod.BSM,
        ).present_value()

        # MC arithmetic
        mc_und = _gbm_underlying(
            spot=spot,
            vol=vol,
            short_rate=r,
            maturity=maturity,
            paths=300_000,
            num_steps=num_steps,
            dividend_curve=_flat_dividend_curve(q, maturity),
        )
        mc_pv = OptionValuation(
            mc_und,
            _asian_spec(
                strike=strike,
                maturity=maturity,
                call_put=call_put,
                averaging=AsianAveraging.ARITHMETIC,
            ),
            PricingMethod.MONTE_CARLO,
            params=MonteCarloParams(random_seed=42),
        ).present_value()

        logger.info(
            "Arith Asian %s S=%.0f K=%.0f analytical=%.6f MC=%.6f",
            call_put.value,
            spot,
            strike,
            analytical_pv,
            mc_pv,
        )
        # Turnbull-Wakeman is an approximation; allow slightly wider tolerance
        assert np.isclose(analytical_pv, mc_pv, rtol=0.03), (
            f"analytical={analytical_pv:.6f} MC={mc_pv:.6f}"
        )


# ---------------------------------------------------------------------------
# Arithmetic properties
# ---------------------------------------------------------------------------


class TestArithmeticProperties:
    """Sanity checks on arithmetic analytical prices."""

    def test_arithmetic_geq_geometric_call(self):
        """Arithmetic call ≥ geometric call (AM-GM: E[arith avg] ≥ E[geom avg])."""
        maturity = PRICING_DATE + dt.timedelta(days=365)
        und = _underlying(spot=100, vol=0.25, short_rate=0.05, maturity=maturity)

        arith_pv = OptionValuation(
            und,
            _asian_spec(
                strike=100,
                maturity=maturity,
                call_put=OptionType.CALL,
                num_steps=52,
                averaging=AsianAveraging.ARITHMETIC,
            ),
            PricingMethod.BSM,
        ).present_value()

        geom_pv = OptionValuation(
            und,
            _asian_spec(
                strike=100,
                maturity=maturity,
                call_put=OptionType.CALL,
                num_steps=52,
                averaging=AsianAveraging.GEOMETRIC,
            ),
            PricingMethod.BSM,
        ).present_value()

        assert arith_pv >= geom_pv - 1e-10, f"arith call={arith_pv:.6f} < geom call={geom_pv:.6f}"

    def test_arithmetic_leq_geometric_put(self):
        """Arithmetic put ≤ geometric put (AM-GM: higher avg lowers put value)."""
        maturity = PRICING_DATE + dt.timedelta(days=365)
        und = _underlying(spot=100, vol=0.25, short_rate=0.05, maturity=maturity)

        arith_pv = OptionValuation(
            und,
            _asian_spec(
                strike=100,
                maturity=maturity,
                call_put=OptionType.PUT,
                num_steps=52,
                averaging=AsianAveraging.ARITHMETIC,
            ),
            PricingMethod.BSM,
        ).present_value()

        geom_pv = OptionValuation(
            und,
            _asian_spec(
                strike=100,
                maturity=maturity,
                call_put=OptionType.PUT,
                num_steps=52,
                averaging=AsianAveraging.GEOMETRIC,
            ),
            PricingMethod.BSM,
        ).present_value()

        assert arith_pv <= geom_pv + 1e-10, f"arith put={arith_pv:.6f} > geom put={geom_pv:.6f}"

    def test_arithmetic_less_than_vanilla(self):
        """Arithmetic Asian call < vanilla European call (averaging effect)."""
        maturity = PRICING_DATE + dt.timedelta(days=365)
        und = _underlying(spot=100, vol=0.25, short_rate=0.05, maturity=maturity)

        arith_pv = OptionValuation(
            und,
            _asian_spec(
                strike=100,
                maturity=maturity,
                call_put=OptionType.CALL,
                num_steps=52,
                averaging=AsianAveraging.ARITHMETIC,
            ),
            PricingMethod.BSM,
        ).present_value()

        vanilla_pv = OptionValuation(
            und,
            OptionSpec(
                option_type=OptionType.CALL,
                exercise_type=ExerciseType.EUROPEAN,
                strike=100,
                maturity=maturity,
                currency=CURRENCY,
            ),
            PricingMethod.BSM,
        ).present_value()

        assert arith_pv < vanilla_pv

    def test_positive_price(self):
        maturity = PRICING_DATE + dt.timedelta(days=365)
        und = _underlying(spot=100, vol=0.2, short_rate=0.05, maturity=maturity)
        for call_put in (OptionType.CALL, OptionType.PUT):
            pv = OptionValuation(
                und,
                _asian_spec(
                    strike=100,
                    maturity=maturity,
                    call_put=call_put,
                    num_steps=12,
                    averaging=AsianAveraging.ARITHMETIC,
                ),
                PricingMethod.BSM,
            ).present_value()
            assert pv > 0.0

    def test_dividend_reduces_call(self):
        maturity = PRICING_DATE + dt.timedelta(days=365)
        pv_no_q = OptionValuation(
            _underlying(spot=100, vol=0.2, short_rate=0.05, maturity=maturity),
            _asian_spec(
                strike=100,
                maturity=maturity,
                call_put=OptionType.CALL,
                num_steps=12,
                averaging=AsianAveraging.ARITHMETIC,
            ),
            PricingMethod.BSM,
        ).present_value()

        pv_with_q = OptionValuation(
            _underlying(
                spot=100,
                vol=0.2,
                short_rate=0.05,
                maturity=maturity,
                dividend_curve=_flat_dividend_curve(0.03, maturity),
            ),
            _asian_spec(
                strike=100,
                maturity=maturity,
                call_put=OptionType.CALL,
                num_steps=12,
                averaging=AsianAveraging.ARITHMETIC,
            ),
            PricingMethod.BSM,
        ).present_value()

        assert pv_with_q < pv_no_q
