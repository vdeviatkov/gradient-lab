import math

import pytest

from ml_scratch.pure.datasets import logic_gate_dataset
from ml_scratch.pure.mlp import MlpParameters, XorMlp


def _hand_built_model() -> XorMlp:
    """A two-unit network whose every parameter is known, so a forward pass can be checked."""
    model = XorMlp(hidden_units=2, seed=0)
    model.set_parameters(
        MlpParameters(
            input_weights=((1.0, -2.0), (0.5, 0.25)),
            hidden_biases=(0.5, -1.0),
            output_weights=(2.0, -3.0),
            output_bias=0.25,
        )
    )
    return model


def test_forward_pass_matches_the_hand_computation() -> None:
    model = _hand_built_model()
    features = (1.0, 2.0)

    first = math.tanh(1.0 * 1.0 + -2.0 * 2.0 + 0.5)
    second = math.tanh(0.5 * 1.0 + 0.25 * 2.0 + -1.0)
    expected_logit = 2.0 * first + -3.0 * second + 0.25

    assert model.hidden_activations(features) == pytest.approx((first, second))
    assert model.logit(features) == pytest.approx(expected_logit)
    assert model.predict_probability(features) == pytest.approx(
        1.0 / (1.0 + math.exp(-expected_logit))
    )
    assert model.predict(features) == int(expected_logit >= 0.0)


def test_loss_is_the_mean_cross_entropy() -> None:
    model = _hand_built_model()
    dataset = logic_gate_dataset("xor")

    expected = 0.0
    for features, target in dataset:
        probability = model.predict_probability(features)
        expected -= target * math.log(probability) + (1 - target) * math.log(1.0 - probability)
    expected /= len(dataset)

    assert model.binary_cross_entropy(dataset) == pytest.approx(expected)


def test_loss_does_not_overflow_on_a_confident_wrong_answer() -> None:
    """The logit form has to survive a score large enough to overflow ``exp``."""
    model = XorMlp(hidden_units=1, seed=0)
    model.set_parameters(
        MlpParameters(((0.0, 0.0),), (0.0,), (0.0,), output_bias=-800.0)
    )
    loss = model.binary_cross_entropy((((0.0, 0.0), 1),))

    assert math.isfinite(loss)
    assert loss == pytest.approx(800.0)


def test_output_delta_collapses_to_probability_minus_target() -> None:
    """With the sigmoid inside the loss, the output bias's gradient is the mean of ``p - y``."""
    model = _hand_built_model()
    dataset = logic_gate_dataset("xor")

    expected = sum(
        model.predict_probability(features) - target for features, target in dataset
    ) / len(dataset)

    assert model.gradient(dataset).output_bias == pytest.approx(expected)


@pytest.mark.parametrize("hidden_units", [1, 2, 4])
def test_hand_derived_gradient_matches_central_differences(hidden_units: int) -> None:
    model = XorMlp(hidden_units=hidden_units, seed=3)
    dataset = logic_gate_dataset("xor")

    analytic = model.gradient(dataset)
    numerical = model.numerical_gradient(dataset)

    for unit in range(hidden_units):
        assert analytic.input_weights[unit] == pytest.approx(
            numerical.input_weights[unit], abs=1e-9
        )
    assert analytic.hidden_biases == pytest.approx(numerical.hidden_biases, abs=1e-9)
    assert analytic.output_weights == pytest.approx(numerical.output_weights, abs=1e-9)
    assert analytic.output_bias == pytest.approx(numerical.output_bias, abs=1e-9)


def test_numerical_gradient_leaves_the_parameters_untouched() -> None:
    model = XorMlp(hidden_units=3, seed=5)
    before = model.parameters

    model.numerical_gradient(logic_gate_dataset("xor"))

    assert model.parameters == before


def test_training_learns_xor() -> None:
    dataset = logic_gate_dataset("xor")
    model = XorMlp(hidden_units=4, learning_rate=0.5, seed=7)

    result = model.fit(dataset, max_epochs=10_000, target_loss=0.02)

    assert result.converged
    assert result.loss_per_epoch[-1] <= 0.02
    assert model.accuracy(dataset) == 1.0
    assert model.predict_many(features for features, _ in dataset) == (0, 1, 1, 0)


def test_training_loss_falls_monotonically_on_this_problem() -> None:
    """Full-batch descent at this step size should not overshoot; if it did, the claim is wrong."""
    model = XorMlp(hidden_units=4, learning_rate=0.5, seed=7)

    result = model.fit(logic_gate_dataset("xor"), max_epochs=300, target_loss=1e-9)

    losses = result.loss_per_epoch
    assert all(
        later <= earlier + 1e-12 for earlier, later in zip(losses, losses[1:], strict=False)
    )


def test_one_hidden_unit_cannot_represent_xor() -> None:
    """A single tanh unit is a monotone function of one linear score, so it is still one boundary.

    The perceptron milestone showed XOR needs more than a single straight boundary; this shows
    that adding a hidden layer does not help unless it is wide enough to draw more than one.
    """
    dataset = logic_gate_dataset("xor")
    accuracies = []
    for seed in range(8):
        model = XorMlp(hidden_units=1, learning_rate=0.5, seed=seed)
        result = model.fit(dataset, max_epochs=2_000, target_loss=0.02)
        assert not result.converged
        accuracies.append(model.accuracy(dataset))

    assert max(accuracies) < 1.0
    # The best a single boundary can do on XOR is three of the four rows, exactly as for the
    # perceptron.
    assert max(accuracies) <= 0.75


def test_four_hidden_units_solve_xor_from_every_seed_tried() -> None:
    dataset = logic_gate_dataset("xor")

    for seed in range(8):
        model = XorMlp(hidden_units=4, learning_rate=0.5, seed=seed)
        result = model.fit(dataset, max_epochs=10_000, target_loss=0.02)

        assert result.converged
        assert model.accuracy(dataset) == 1.0


def test_same_seed_reproduces_training_exactly() -> None:
    dataset = logic_gate_dataset("xor")
    first = XorMlp(hidden_units=4, seed=23)
    second = XorMlp(hidden_units=4, seed=23)

    assert first.parameters == second.parameters

    first_result = first.fit(dataset, max_epochs=200, target_loss=1e-9)
    second_result = second.fit(dataset, max_epochs=200, target_loss=1e-9)

    assert first_result == second_result
    assert first.parameters == second.parameters
    assert XorMlp(hidden_units=4, seed=24).parameters != first.parameters


def test_biases_start_at_zero_and_weights_do_not() -> None:
    model = XorMlp(hidden_units=4, seed=11)

    assert model.hidden_biases == [0.0, 0.0, 0.0, 0.0]
    assert model.output_bias == 0.0
    assert all(weight != 0.0 for row in model.input_weights for weight in row)
    assert all(weight != 0.0 for weight in model.output_weights)
    # Glorot limits: every initial weight lies inside sqrt(6 / (fan_in + fan_out)).
    hidden_limit = math.sqrt(6.0 / (2 + 4))
    output_limit = math.sqrt(6.0 / (4 + 1))
    assert all(abs(weight) < hidden_limit for row in model.input_weights for weight in row)
    assert all(abs(weight) < output_limit for weight in model.output_weights)


def test_parameters_round_trip() -> None:
    model = XorMlp(hidden_units=2, seed=1)
    snapshot = model.parameters

    model.fit(logic_gate_dataset("xor"), max_epochs=10, target_loss=1e-9)
    assert model.parameters != snapshot

    model.set_parameters(snapshot)
    assert model.parameters == snapshot


@pytest.mark.parametrize(
    ("constructor_arguments", "message"),
    [
        ({"hidden_units": 0}, "hidden_units"),
        ({"hidden_units": 2, "learning_rate": 0.0}, "learning_rate"),
        ({"hidden_units": 2, "feature_count": 0}, "feature_count"),
    ],
)
def test_invalid_construction_is_rejected(
    constructor_arguments: dict[str, float], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        XorMlp(**constructor_arguments)  # type: ignore[arg-type]


def test_invalid_input_is_rejected() -> None:
    model = XorMlp(hidden_units=2, seed=0)

    with pytest.raises(ValueError, match="expected 2 features"):
        model.predict((1.0,))
    with pytest.raises(ValueError, match="must not be empty"):
        model.binary_cross_entropy(())
    with pytest.raises(ValueError, match="binary values"):
        model.binary_cross_entropy((((0.0, 0.0), 2),))
    with pytest.raises(ValueError, match="max_epochs"):
        model.fit(logic_gate_dataset("xor"), max_epochs=0)
    with pytest.raises(ValueError, match="target_loss"):
        model.fit(logic_gate_dataset("xor"), target_loss=0.0)
    with pytest.raises(ValueError, match="one row per hidden unit"):
        model.set_parameters(MlpParameters(((1.0, 1.0),), (0.0, 0.0), (1.0, 1.0), 0.0))
    with pytest.raises(ValueError, match="one entry per feature"):
        model.set_parameters(MlpParameters(((1.0,), (1.0,)), (0.0, 0.0), (1.0, 1.0), 0.0))
    with pytest.raises(ValueError, match="one entry per hidden unit"):
        model.set_parameters(MlpParameters(((1.0, 1.0), (1.0, 1.0)), (0.0,), (1.0, 1.0), 0.0))
