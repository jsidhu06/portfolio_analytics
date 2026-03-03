"""Tests for Asian option pricing consistency across methods."""

import datetime as dt
import logging
from typing import Sequence

import numpy as np
import pytest

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
from portfolio_analytics.utils import pv_discrete_dividends
from portfolio_analytics.valuation import OptionValuation, UnderlyingPricingData
from portfolio_analytics.valuation.core import AsianOptionSpec
from portfolio_analytics.valuation.params import BinomialParams, MonteCarloParams


logger = logging.getLogger(__name__)


PRICING_DATE = dt.datetime(2025, 1, 1)
CURRENCY = "USD"
MC_PATHS = 200_000
MC_SEED = 42
NUM_STEPS = 60
ASIAN_TREE_AVERAGES = 100


def _market_data(short_rate: float, maturity: dt.datetime) -> MarketData:
    curve = flat_curve(PRICING_DATE, maturity, short_rate)
    return MarketData(PRICING_DATE, curve, currency=CURRENCY)


def _gbm_underlying(
    *,
    spot: float,
    vol: float,
    short_rate: float,
    dividend_curve: DiscountCurve | None = None,
    discrete_dividends: Sequence[tuple[dt.datetime, float]] | None = None,
    maturity: dt.datetime,
    paths: int,
    num_steps: int,
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


def _binomial_underlying(
    *,
    spot: float,
    vol: float,
    short_rate: float,
    dividend_curve: DiscountCurve | None = None,
    discrete_dividends: Sequence[tuple[dt.datetime, float]] | None = None,
    maturity: dt.datetime,
) -> UnderlyingPricingData:
    return UnderlyingPricingData(
        initial_value=spot,
        volatility=vol,
        market_data=_market_data(short_rate, maturity),
        dividend_curve=dividend_curve,
        discrete_dividends=discrete_dividends,
    )


def _asian_spec(
    *,
    strike: float,
    maturity: dt.datetime,
    call_put: OptionType,
    exercise_type: ExerciseType = ExerciseType.EUROPEAN,
    averaging: AsianAveraging = AsianAveraging.ARITHMETIC,
) -> AsianOptionSpec:
    return AsianOptionSpec(
        averaging=averaging,
        call_put=call_put,
        strike=strike,
        maturity=maturity,
        currency=CURRENCY,
        exercise_type=exercise_type,
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
    q_curve = None if dividend_yield == 0.0 else flat_curve(PRICING_DATE, maturity, dividend_yield)

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

    binom_underlying = _binomial_underlying(
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

    binom_underlying = _binomial_underlying(
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

    binom_underlying = _binomial_underlying(
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

    binom_underlying = _binomial_underlying(
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
        binom_underlying = _binomial_underlying(
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
        binom_underlying = _binomial_underlying(
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
