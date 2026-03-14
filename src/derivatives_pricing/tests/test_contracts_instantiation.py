import datetime as dt

from derivatives_pricing.enums import AsianAveraging, ExerciseType, OptionType
from derivatives_pricing.valuation import AsianSpec, PayoffSpec, VanillaSpec
from derivatives_pricing.valuation import contracts


def test_contract_types_reexport_from_valuation_module():
    """Public re-exports should point to the canonical contracts module classes."""
    assert VanillaSpec is contracts.VanillaSpec
    assert PayoffSpec is contracts.PayoffSpec
    assert AsianSpec is contracts.AsianSpec


def test_vanilla_spec_instantiation_smoke():
    spec = VanillaSpec(
        option_type=OptionType.CALL,
        exercise_type=ExerciseType.EUROPEAN,
        strike=100.0,
        maturity=dt.datetime(2026, 1, 1),
        currency="USD",
    )
    assert spec.option_type is OptionType.CALL
    assert spec.exercise_type is ExerciseType.EUROPEAN
    assert spec.strike == 100.0


def test_payoff_spec_instantiation_smoke():
    spec = PayoffSpec(
        exercise_type=ExerciseType.AMERICAN,
        maturity=dt.datetime(2026, 1, 1),
        payoff_fn=lambda s: s,
        currency="USD",
    )
    assert spec.exercise_type is ExerciseType.AMERICAN
    assert spec.strike is None


def test_asian_spec_instantiation_smoke():
    spec = AsianSpec(
        averaging=AsianAveraging.ARITHMETIC,
        option_type=OptionType.PUT,
        exercise_type=ExerciseType.EUROPEAN,
        strike=100.0,
        maturity=dt.datetime(2026, 1, 1),
        currency="USD",
        num_observations=12,
    )
    assert spec.averaging is AsianAveraging.ARITHMETIC
    assert spec.option_type is OptionType.PUT
    assert spec.exercise_type is ExerciseType.EUROPEAN
