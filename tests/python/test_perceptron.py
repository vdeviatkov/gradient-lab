import pytest

from ml_scratch.pure.datasets import logic_gate_dataset
from ml_scratch.pure.perceptron import Perceptron


def test_prediction_uses_weighted_score_and_step_activation() -> None:
    model = Perceptron(feature_count=2)
    model.weights = [1.0, 1.0]
    model.bias = -1.5

    assert model.predict((0.0, 1.0)) == 0
    assert model.predict((1.0, 1.0)) == 1


@pytest.mark.parametrize("gate", ["and", "or"])
def test_linearly_separable_gate_is_learned(gate: str) -> None:
    dataset = logic_gate_dataset(gate)  # type: ignore[arg-type]
    model = Perceptron(feature_count=2, seed=7)

    result = model.fit(dataset, max_epochs=100)

    assert result.converged
    assert result.mistakes_per_epoch[-1] == 0
    assert model.accuracy(dataset) == 1.0


def test_xor_is_not_reported_as_converged() -> None:
    dataset = logic_gate_dataset("xor")
    model = Perceptron(feature_count=2, seed=7)

    result = model.fit(dataset, max_epochs=100)

    assert not result.converged
    assert result.epochs == 100
    assert model.accuracy(dataset) < 1.0


def test_same_seed_reproduces_training_exactly() -> None:
    dataset = logic_gate_dataset("or")
    first = Perceptron(feature_count=2, seed=23)
    second = Perceptron(feature_count=2, seed=23)

    first_result = first.fit(dataset)
    second_result = second.fit(dataset)

    assert first_result == second_result
    assert first.weights == second.weights
    assert first.bias == second.bias
    assert first.predict_many(features for features, _ in dataset) == second.predict_many(
        features for features, _ in dataset
    )


@pytest.mark.parametrize(
    ("constructor_arguments", "message"),
    [
        ({"feature_count": 0}, "feature_count"),
        ({"feature_count": 2, "learning_rate": 0.0}, "learning_rate"),
    ],
)
def test_invalid_model_configuration_is_rejected(
    constructor_arguments: dict[str, float], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        Perceptron(**constructor_arguments)  # type: ignore[arg-type]


def test_invalid_training_inputs_are_rejected() -> None:
    model = Perceptron(feature_count=2)

    with pytest.raises(ValueError, match="empty"):
        model.fit(())
    with pytest.raises(ValueError, match="expected 2 features"):
        model.predict((1.0,))
    with pytest.raises(ValueError, match="binary"):
        model.fit((((0.0, 0.0), 2),))
