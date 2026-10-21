import numpy as np
import pytest
from ml_scratch.numpy.logistic_regression import (
    BinaryLogisticRegression,
    LogisticTrainingConfig,
    classification_metrics,
    select_threshold_for_f1,
)
from ml_scratch.numpy.logistic_regression.experiment import run_experiment
from ml_scratch.pure.logistic_regression import BinaryLogisticRegression as ScalarRegression
from ml_scratch.pure.logistic_regression.experiment import run_experiment as scalar_experiment

DATA = (((-2, 1), 0), ((0.2, -1), 1), ((1, 3), 1))


@pytest.mark.parametrize("weight", [1, 3.5])
def test_weighted_gradient_scalar_and_central_differences(weight):
    model, scalar = BinaryLogisticRegression(2), ScalarRegression(2)
    model.weights[:], model.bias = [0.3, -0.1], 0.2
    scalar.weights, scalar.bias = list(model.weights), model.bias
    assert model.binary_cross_entropy(DATA, weight) == pytest.approx(
        scalar.binary_cross_entropy(DATA, weight)
    )
    w, b = model.gradient(DATA, weight)
    rw, rb = scalar.gradient(DATA, weight)
    np.testing.assert_allclose(w, rw, atol=1e-14)
    assert b == pytest.approx(rb)
    eps = 1e-6
    for index in range(3):
        original = model.weights[index] if index < 2 else model.bias
        losses = []
        for sign in (1, -1):
            if index < 2:
                model.weights[index] = original + sign * eps
            else:
                model.bias = original + sign * eps
            losses.append(model.binary_cross_entropy(DATA, weight))
        if index < 2:
            model.weights[index] = original
        else:
            model.bias = original
        assert (losses[0] - losses[1]) / (2 * eps) == pytest.approx(
            w[index] if index < 2 else b, abs=1e-8
        )


@pytest.mark.parametrize("batch_size", [0, 1, 2])
def test_training_reset_seed_and_scalar_equivalence(batch_size):
    config = LogisticTrainingConfig(
        batch_size=batch_size, seed=19, max_epochs=20, positive_class_weight=2
    )
    model, scalar = BinaryLogisticRegression(2), ScalarRegression(2)
    result, reference = model.fit(DATA, config), scalar.fit(DATA, config)
    np.testing.assert_allclose(result.loss_per_epoch, reference.loss_per_epoch, atol=1e-14)
    np.testing.assert_allclose(model.weights, scalar.weights, atol=1e-14)
    assert model.fit(DATA, config) == result  # Every fit resets to zero.


def test_extreme_logits_without_cancellation_or_overflow():
    model = BinaryLogisticRegression(1)
    for score in (-1000, 1000):
        model.bias = score
        assert np.isfinite(model.binary_cross_entropy((((0,), 0), ((0,), 1))))
        assert 0 <= model.predict_probability([0]) <= 1
    model.bias = 50
    assert model.binary_cross_entropy((((0,), 1),)) > 0  # Preserve the tiny positive loss.


def test_metrics_threshold_ties_and_invalid_values():
    result = classification_metrics(np.array([0, 0, 1, 1]), np.array([0, 1, 0, 1]))
    assert result.f1 == 0.5 and result.accuracy == 0.5
    assert classification_metrics([0], [0]).f1 == 0
    selection = select_threshold_for_f1([0, 1], [0.1, 0.9], [0.6, 0.4, 0.5])
    assert selection.threshold == 0.5
    assert select_threshold_for_f1([0, 1], [0.1, 0.9], [0.75, 0.25]).threshold == 0.25
    for args in (([], [], []), ([1], [np.nan], [0.5]), ([1], [0.2], [-1])):
        with pytest.raises(ValueError):
            select_threshold_for_f1(*args)


@pytest.mark.parametrize("data", [(), (((1,), 0),), (((1, np.inf), 0),), (((1, 2), 2),)])
def test_invalid_data(data):
    with pytest.raises(ValueError):
        BinaryLogisticRegression(2).fit(data)


def test_invalid_configuration():
    for config in (
        LogisticTrainingConfig(learning_rate=np.nan),
        LogisticTrainingConfig(max_epochs=0),
        LogisticTrainingConfig(batch_size=-1),
        LogisticTrainingConfig(positive_class_weight=0),
        LogisticTrainingConfig(target_loss=np.inf),
    ):
        with pytest.raises(ValueError):
            BinaryLogisticRegression(2).fit(DATA, config)


def test_experiment_matches_scalar():
    result, scalar = run_experiment(), scalar_experiment()
    assert result.training.epochs == scalar.training.epochs
    np.testing.assert_allclose(result.weights, scalar.weights, atol=1e-13)
    assert result.selection == scalar.selection and result.selected == scalar.selected
