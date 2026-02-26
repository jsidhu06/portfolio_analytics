"""Tests for analytical Asian option pricing.

- Geometric: Kemna-Vorst (1990) exact closed-form
- Arithmetic: Turnbull-Wakeman (1991) moment-matching (Hull §26.13)
"""

import datetime as dt
import logging

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
from portfolio_analytics.market_environment import MarketData
from portfolio_analytics.stochastic_processes import (
    GBMParams,
    GBMProcess,
    SimulationConfig,
)
from portfolio_analytics.tests.helpers import flat_curve
from portfolio_analytics.valuation import OptionValuation, OptionSpec, UnderlyingPricingData
from portfolio_analytics.valuation.core import AsianOptionSpec
from portfolio_analytics.valuation.asian_analytical import (
    _asian_geometric_analytical,
    _asian_arithmetic_analytical,
)
from portfolio_analytics.valuation.params import BinomialParams, MonteCarloParams

logger = logging.getLogger(__name__)

PRICING_DATE = dt.datetime(2025, 1, 1)
CURRENCY = "USD"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _market_data(short_rate: float, maturity: dt.datetime) -> MarketData:
    curve = flat_curve(PRICING_DATE, maturity, short_rate)
    return MarketData(PRICING_DATE, curve, currency=CURRENCY)


def _underlying(
    *,
    spot: float,
    vol: float,
    short_rate: float,
    maturity: dt.datetime,
    dividend_yield: float = 0.0,
) -> UnderlyingPricingData:
    q_curve = None if dividend_yield == 0.0 else flat_curve(PRICING_DATE, maturity, dividend_yield)
    return UnderlyingPricingData(
        initial_value=spot,
        volatility=vol,
        market_data=_market_data(short_rate, maturity),
        dividend_curve=q_curve,
    )


def _asian_spec(
    *,
    strike: float,
    maturity: dt.datetime,
    call_put: OptionType,
    num_steps: int | None = None,
    averaging: AsianAveraging = AsianAveraging.GEOMETRIC,
    averaging_start: dt.datetime | None = None,
) -> AsianOptionSpec:
    return AsianOptionSpec(
        averaging=averaging,
        call_put=call_put,
        strike=strike,
        maturity=maturity,
        currency=CURRENCY,
        num_steps=num_steps,
        averaging_start=averaging_start,
    )


def _gbm_underlying(
    *,
    spot: float,
    vol: float,
    short_rate: float,
    maturity: dt.datetime,
    paths: int,
    num_steps: int,
    dividend_yield: float = 0.0,
) -> GBMProcess:
    q_curve = None if dividend_yield == 0.0 else flat_curve(PRICING_DATE, maturity, dividend_yield)
    sim = SimulationConfig(
        paths=paths,
        day_count_convention=DayCountConvention.ACT_365F,
        num_steps=num_steps,
        end_date=maturity,
    )
    params = GBMParams(initial_value=spot, volatility=vol, dividend_curve=q_curve)
    return GBMProcess(
        _market_data(short_rate, maturity),
        params,
        sim,
    )


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
        und = _underlying(spot=spot, vol=vol, short_rate=r, maturity=maturity, dividend_yield=q)

        asian_pv = OptionValuation(
            und,
            _asian_spec(strike=strike, maturity=maturity, call_put=call_put, num_steps=1),
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
        und = _underlying(spot=spot, vol=vol, short_rate=r, maturity=maturity, dividend_yield=q)

        call_pv = OptionValuation(
            und,
            _asian_spec(
                strike=strike, maturity=maturity, call_put=OptionType.CALL, num_steps=num_obs
            ),
            PricingMethod.BSM,
        ).present_value()

        put_pv = OptionValuation(
            und,
            _asian_spec(
                strike=strike, maturity=maturity, call_put=OptionType.PUT, num_steps=num_obs
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
        und = _underlying(spot=spot, vol=vol, short_rate=r, maturity=maturity, dividend_yield=q)
        analytical_pv = OptionValuation(
            und,
            _asian_spec(strike=strike, maturity=maturity, call_put=call_put, num_steps=num_steps),
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
            dividend_yield=q,
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
            _asian_spec(strike=100, maturity=maturity, call_put=OptionType.CALL, num_steps=12),
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
                _asian_spec(strike=100, maturity=maturity, call_put=OptionType.CALL, num_steps=12),
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
                _asian_spec(strike=100, maturity=maturity, call_put=OptionType.PUT, num_steps=12),
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
            _asian_spec(strike=100, maturity=maturity, call_put=OptionType.CALL, num_steps=4),
            PricingMethod.BSM,
        ).present_value()
        pv_252 = OptionValuation(
            und,
            _asian_spec(strike=100, maturity=maturity, call_put=OptionType.CALL, num_steps=252),
            PricingMethod.BSM,
        ).present_value()
        assert pv_252 > pv_4

    def test_geometric_call_leq_vanilla_bsm(self):
        """Geometric average call ≤ vanilla European call (averaging reduces variance)."""
        maturity = PRICING_DATE + dt.timedelta(days=365)
        und = _underlying(spot=100, vol=0.25, short_rate=0.05, maturity=maturity)

        geom_pv = OptionValuation(
            und,
            _asian_spec(strike=100, maturity=maturity, call_put=OptionType.CALL, num_steps=52),
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
            _asian_spec(strike=100, maturity=maturity, call_put=OptionType.PUT, num_steps=52),
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
            _underlying(spot=100, vol=0.2, short_rate=0.05, maturity=maturity, dividend_yield=0.0),
            _asian_spec(strike=100, maturity=maturity, call_put=OptionType.CALL, num_steps=12),
            PricingMethod.BSM,
        ).present_value()

        pv_with_q = OptionValuation(
            _underlying(spot=100, vol=0.2, short_rate=0.05, maturity=maturity, dividend_yield=0.03),
            _asian_spec(strike=100, maturity=maturity, call_put=OptionType.CALL, num_steps=12),
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
            _underlying(spot=100, vol=0.2, short_rate=0.05, maturity=maturity, dividend_yield=0.0),
            _asian_spec(strike=100, maturity=maturity, call_put=OptionType.PUT, num_steps=12),
            PricingMethod.BSM,
        ).present_value()

        pv_with_q = OptionValuation(
            _underlying(spot=100, vol=0.2, short_rate=0.05, maturity=maturity, dividend_yield=0.03),
            _asian_spec(strike=100, maturity=maturity, call_put=OptionType.PUT, num_steps=12),
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
                _asian_spec(strike=100, maturity=maturity, call_put=OptionType.CALL),
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
            _asian_spec(strike=100, maturity=maturity, call_put=OptionType.CALL, num_steps=12),
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


MC_PATHS = 200_000
MC_SEED = 42
NUM_STEPS = 60
BINOM_STEPS = 100
TREE_AVERAGES = 100


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
        dividend_yield=q,
    )
    analytical_pv = OptionValuation(
        und_determ,
        _asian_spec(
            strike=strike,
            maturity=maturity,
            call_put=call_put,
            num_steps=NUM_STEPS,
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
            asian_tree_averages=TREE_AVERAGES,
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
        dividend_yield=q,
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

        Hull uses observations at T/m, ..., T (no S₀).  We compute using
        the raw moment formulas directly to match his convention.
        """
        S0, K, r, q, sigma, T = 50.0, 50.0, 0.1, 0.0, 0.4, 1.0

        for m, hull_price in [(12, 6.00), (52, 5.70), (250, 5.63)]:
            # Hull convention: m observations at T/m, 2T/m, ..., T
            delta = T / m
            t = np.arange(1, m + 1, dtype=float) * delta  # t[0]=T/m, t[-1]=T
            F = S0 * np.exp((r - q) * t)
            M1 = np.mean(F)
            F_cumrev = np.cumsum(F[::-1])[::-1]
            M2 = np.sum(F * np.exp(sigma**2 * t) * (2 * F_cumrev - F)) / m**2

            sigma_a_sq = np.log(M2 / M1**2) / T
            sigma_a = np.sqrt(sigma_a_sq)
            d1 = (np.log(M1 / K) + 0.5 * sigma_a_sq * T) / (sigma_a * np.sqrt(T))
            d2 = d1 - sigma_a * np.sqrt(T)
            df = np.exp(-r * T)
            price = df * (M1 * norm.cdf(d1) - K * norm.cdf(d2))

            logger.info(
                "Hull 26.3: m=%d M1=%.2f M2=%.2f sigma_a=%.4f price=%.4f (expected %.2f)",
                m,
                M1,
                M2,
                sigma_a,
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
        und = _underlying(spot=spot, vol=vol, short_rate=r, maturity=maturity, dividend_yield=q)

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
        und = _underlying(spot=spot, vol=vol, short_rate=r, maturity=maturity, dividend_yield=q)
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
            dividend_yield=q,
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
            _underlying(spot=100, vol=0.2, short_rate=0.05, maturity=maturity, dividend_yield=0.0),
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
            _underlying(spot=100, vol=0.2, short_rate=0.05, maturity=maturity, dividend_yield=0.03),
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
