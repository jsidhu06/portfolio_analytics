"""Tests for Binomial tree option valuation."""

import datetime as dt

import numpy as np

from portfolio_analytics.enums import DayCountConvention, ExerciseType, OptionType, PricingMethod
from portfolio_analytics.market_environment import MarketData
from portfolio_analytics.tests.conftest import (
    BINOM_PARAMS,
    PRICING_DATE,
    MATURITY,
    CURRENCY,
    SPOT,
    STRIKE,
    RATE,
    VOL,
)
from portfolio_analytics.tests.helpers import flat_curve, pv
from portfolio_analytics.utils import calculate_year_fraction, expected_binomial_payoff
from portfolio_analytics.valuation import (
    BinomialParams,
    VanillaSpec,
    UnderlyingData,
)

# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

_CURVE = flat_curve(PRICING_DATE, MATURITY, RATE)
_MD = MarketData(PRICING_DATE, _CURVE, currency=CURRENCY)


def _underlying(
    spot: float = SPOT,
    vol: float = VOL,
    dividend_curve=None,
    discrete_dividends=None,
) -> UnderlyingData:
    return UnderlyingData(
        initial_value=spot,
        volatility=vol,
        market_data=_MD,
        dividend_curve=dividend_curve,
        discrete_dividends=discrete_dividends,
    )


def _spec(
    option_type: OptionType = OptionType.CALL,
    exercise: ExerciseType = ExerciseType.EUROPEAN,
    strike: float = STRIKE,
) -> VanillaSpec:
    return VanillaSpec(
        option_type=option_type,
        exercise_type=exercise,
        strike=strike,
        maturity=MATURITY,
        currency=CURRENCY,
    )


def _binom(ud: UnderlyingData, spec: VanillaSpec, params: BinomialParams = BINOM_PARAMS) -> float:
    return pv(ud, spec, PricingMethod.BINOMIAL, params=params)


class TestBinomialValuation:
    """Tests for Binomial tree option valuation."""

    def test_binomial_european_call_atm(self):
        """Test binomial European call option."""
        result = _binom(_underlying(), _spec())
        assert result > 0
        assert np.isclose(result, 10.45, rtol=0.01)

    def test_binomial_american_call_no_div_equal_to_european(self):
        """Test that American call >= European call (same parameters)."""
        eu_price = _binom(_underlying(), _spec())
        am_price = _binom(_underlying(), _spec(exercise=ExerciseType.AMERICAN))
        assert np.isclose(am_price, eu_price, rtol=0.005)

    def test_binomial_european_call_discrete_dividends_reduce_price(self):
        """Discrete dividends should reduce European call price in binomial tree."""
        spec = _spec()
        pv_no_div = _binom(_underlying(), spec)
        pv_div = _binom(
            _underlying(discrete_dividends=[(PRICING_DATE + dt.timedelta(days=180), 1.0)]),
            spec,
        )
        assert pv_div < pv_no_div

    def test_binomial_american_put_early_exercise(self):
        """Test American put has early exercise premium."""
        eu_price = _binom(
            _underlying(), _spec(OptionType.PUT), params=BinomialParams(num_steps=100)
        )
        am_price = _binom(
            _underlying(),
            _spec(OptionType.PUT, exercise=ExerciseType.AMERICAN),
            params=BinomialParams(num_steps=100),
        )
        assert am_price > eu_price

    def test_binomial_convergence(self):
        """Test that binomial prices converge with more steps."""
        spec = _spec()
        price_100 = _binom(_underlying(), spec, params=BinomialParams(num_steps=100))
        price_200 = _binom(_underlying(), spec, params=BinomialParams(num_steps=200))
        assert abs(price_200 - price_100) < 1.0

    def test_binomial_pv_matches_expected_binomial_payoff(self):
        n_steps = 250
        pv_binom = _binom(_underlying(), _spec(), params=BinomialParams(num_steps=n_steps))

        T = calculate_year_fraction(
            PRICING_DATE, MATURITY, day_count_convention=DayCountConvention.ACT_365F
        )
        dt_step = T / n_steps
        u = np.exp(VOL * np.sqrt(dt_step))

        expected_payoff = expected_binomial_payoff(
            S0=SPOT,
            n=n_steps,
            T=T,
            option_type=OptionType.CALL,
            K=STRIKE,
            r=RATE,
            q=0,
            u=u,
        )
        pv_expected = np.exp(-RATE * T) * expected_payoff
        assert np.isclose(pv_binom, pv_expected, rtol=1.0e-4)
