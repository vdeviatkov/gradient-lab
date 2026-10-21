import pytest
import torch
from ml_scratch.pure.logistic_regression import BinaryLogisticRegression as ScalarRegression
from ml_scratch.pure.logistic_regression.experiment import run_experiment as scalar_experiment
from ml_scratch.pytorch.logistic_regression import (
    BinaryLogisticRegression,
    LogisticTrainingConfig,
    classification_metrics,
    select_threshold_for_f1,
)
from ml_scratch.pytorch.logistic_regression.experiment import run_experiment

DATA = (((-2, 1), 0), ((0.2, -1), 1), ((1, 3), 1))


@pytest.mark.parametrize("weight", [1, 3.7])
def test_weighted_loss_manual_autograd_scalar_and_central_differences(weight):
    model, scalar = BinaryLogisticRegression(2), ScalarRegression(2)
    model.weights[:] = torch.tensor([0.3, -0.1], dtype=torch.float64)
    model.bias = 0.2
    scalar.weights, scalar.bias = model.weights.tolist(), model.bias
    assert model.binary_cross_entropy(DATA, weight) == pytest.approx(
        scalar.binary_cross_entropy(DATA, weight), abs=1e-13
    )
    w, b = model.gradient(DATA, weight)
    aw, ab = model.autograd_gradient(DATA, weight)
    rw, rb = scalar.gradient(DATA, weight)
    torch.testing.assert_close(w, aw, atol=1e-13, rtol=1e-13)
    torch.testing.assert_close(w, torch.tensor(rw, dtype=torch.float64), atol=1e-13, rtol=1e-13)
    assert b == pytest.approx(ab, abs=1e-13) == pytest.approx(rb, abs=1e-13)
    eps = 1e-6
    for index in range(3):
        original = float(model.weights[index]) if index < 2 else model.bias
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
            float(w[index]) if index < 2 else b, abs=1e-8
        )
    assert model.weights.grad is None


@pytest.mark.parametrize("batch_size", [0, 1, 2])
def test_training_reset_and_scalar_agreement(batch_size):
    config = LogisticTrainingConfig(
        batch_size=batch_size, seed=19, max_epochs=20, positive_class_weight=2
    )
    model, scalar = BinaryLogisticRegression(2), ScalarRegression(2)
    result, reference = model.fit(DATA, config), scalar.fit(DATA, config)
    torch.testing.assert_close(
        torch.tensor(result.loss_per_epoch, dtype=torch.float64),
        torch.tensor(reference.loss_per_epoch, dtype=torch.float64),
        atol=1e-13,
        rtol=1e-13,
    )
    torch.testing.assert_close(model.weights, torch.tensor(scalar.weights, dtype=torch.float64))
    assert model.fit(DATA, config) == result


def test_extreme_logits_metrics_and_threshold_ties():
    model = BinaryLogisticRegression(1)
    for score in (-1000, 1000):
        model.bias = score
        assert 0 <= model.predict_probability([0]) <= 1
        assert model.binary_cross_entropy((((0,), 0), ((0,), 1))) == 500
    model.bias = 50
    assert model.binary_cross_entropy((((0,), 1),)) > 0
    assert classification_metrics([0, 0, 1, 1], [0, 1, 0, 1]).f1 == 0.5
    assert classification_metrics([0], [0]).f1 == 0
    assert select_threshold_for_f1([0, 1], [0.1, 0.9], [0.6, 0.5, 0.4]).threshold == 0.5
    assert select_threshold_for_f1([0, 1], [0.1, 0.9], [0.75, 0.25]).threshold == 0.25


@pytest.mark.parametrize("data", [(), (((1,), 0),), (((1, float("inf")), 0),), (((1, 2), 2),)])
def test_invalid_data(data):
    with pytest.raises(ValueError):
        BinaryLogisticRegression(2).fit(data)


def test_invalid_controls_and_experiment():
    for config in (
        LogisticTrainingConfig(learning_rate=float("nan")),
        LogisticTrainingConfig(max_epochs=0),
        LogisticTrainingConfig(batch_size=-1),
        LogisticTrainingConfig(positive_class_weight=0),
        LogisticTrainingConfig(target_loss=float("inf")),
    ):
        with pytest.raises(ValueError):
            BinaryLogisticRegression(2).fit(DATA, config)
    for args in (([], [], []), ([1], [float("nan")], [0.5]), ([1], [0.2], [-1])):
        with pytest.raises(ValueError):
            select_threshold_for_f1(*args)
    result, reference = run_experiment(), scalar_experiment()
    assert result.training.epochs == reference.training.epochs
    assert result.selection == reference.selection and result.selected == reference.selected
    torch.testing.assert_close(
        torch.tensor(result.weights, dtype=torch.float64),
        torch.tensor(reference.weights, dtype=torch.float64),
    )
