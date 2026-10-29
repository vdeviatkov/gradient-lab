import math
import random

import pytest
from ml_scratch.pure.softmax_mnist import (
    SoftmaxRegression,
    SoftmaxTrainingConfig,
    multiclass_metrics,
    softmax,
    sparse_dataset,
)


def blobs(seed=3, per_class=20, classes=3, width=6):
    """Learnable data: class k raises features 2k and 2k + 1; about half of all values are zero."""
    rng = random.Random(seed)
    examples = []
    for label in range(classes):
        for _ in range(per_class):
            row = [rng.choice((0.0, 0.0, rng.random() * 0.3)) for _ in range(width)]
            row[2 * label] += 0.6 + 0.4 * rng.random()
            row[2 * label + 1] += 0.6 + 0.4 * rng.random()
            examples.append((row, label))
    rng.shuffle(examples)
    return examples


def test_zero_model_loss_gradient_and_probabilities_by_hand():
    model = SoftmaxRegression(2, 3)
    data = [([1.0, 2.0], 0), ([0.0, -1.0], 2)]
    assert model.cross_entropy(data) == pytest.approx(math.log(3))
    assert model.predict_probabilities([5.0, 1.0]) == pytest.approx([1 / 3] * 3)
    assert model.predict([5.0, 1.0]) == 0  # Ties resolve to the lowest class.
    # dL/dw_k = mean (p_k - y_k) x with p = 1/3 everywhere.
    expected_rows = [
        [(1 / 3 - 1) * 1.0 / 2 + (1 / 3) * 0.0 / 2, (1 / 3 - 1) * 2.0 / 2 + (1 / 3) * -1.0 / 2],
        [(1 / 3) * 1.0 / 2 + (1 / 3) * 0.0 / 2, (1 / 3) * 2.0 / 2 + (1 / 3) * -1.0 / 2],
        [(1 / 3) * 1.0 / 2 + (1 / 3 - 1) * 0.0 / 2, (1 / 3) * 2.0 / 2 + (1 / 3 - 1) * -1.0 / 2],
    ]
    expected_biases = [(1 / 3 - 1 + 1 / 3) / 2, (2 / 3) / 2, (1 / 3 + 1 / 3 - 1) / 2]
    expected = [g for row in expected_rows for g in row] + expected_biases
    assert model.gradient(data) == pytest.approx(expected)


def test_parameters_layout_round_trip_and_scores():
    model = SoftmaxRegression(2, 2)
    model.set_parameters([1.0, 2.0, 3.0, 4.0, 0.5, -0.5])
    assert model.weights == [[1.0, 2.0], [3.0, 4.0]] and model.biases == [0.5, -0.5]
    assert model.scores([1.0, 1.0]) == pytest.approx([3.5, 6.5])
    assert model.parameters() == [1.0, 2.0, 3.0, 4.0, 0.5, -0.5]
    assert model.parameter_count == 6


@pytest.mark.parametrize("l2", [0.0, 0.05])
def test_gradient_matches_central_differences(l2):
    data = blobs(per_class=4)
    model = SoftmaxRegression(6, 3)
    rng = random.Random(1)
    model.set_parameters([rng.uniform(-1, 1) for _ in range(model.parameter_count)])
    analytic = model.gradient(data, l2)
    base = model.parameters()
    for index in range(model.parameter_count):
        values = list(base)
        values[index] += 1e-6
        model.set_parameters(values)
        above = model.cross_entropy(data, l2)
        values[index] -= 2e-6
        model.set_parameters(values)
        below = model.cross_entropy(data, l2)
        assert analytic[index] == pytest.approx((above - below) / 2e-6, abs=1e-8)


def test_sparse_and_dense_inputs_agree_exactly():
    examples = blobs()
    model = SoftmaxRegression(6, 3)
    model.fit(examples, SoftmaxTrainingConfig(learning_rate=0.5, max_epochs=3, batch_size=8))
    sparse = sparse_dataset(examples)
    assert sum(map(len, sparse.indices)) < 6 * len(examples)  # Zeros really were dropped.
    for (row, _), indices in zip(examples, sparse.indices, strict=True):
        dense_scores = [
            b + sum(w * x for w, x in zip(weights, row, strict=True))
            for weights, b in zip(model.weights, model.biases, strict=True)
        ]
        assert model.scores(row) == pytest.approx(dense_scores, rel=1e-15)
        assert all(row[i] != 0 for i in indices)
    assert model.cross_entropy(examples) == model.cross_entropy(sparse)


def test_training_learns_and_is_reproducible():
    examples = blobs(per_class=30)
    config = SoftmaxTrainingConfig(learning_rate=0.5, max_epochs=30, batch_size=8, seed=4)
    first, second = SoftmaxRegression(6, 3), SoftmaxRegression(6, 3)
    result = first.fit(examples, config)
    assert result == second.fit(examples, config) and first.parameters() == second.parameters()
    assert result.loss_per_epoch[-1] < result.loss_per_epoch[0] / 3
    assert first.accuracy(examples) == 1
    assert sum(first.predict_probabilities(examples[0][0])) == pytest.approx(1)
    target = SoftmaxRegression(6, 3).fit(
        examples, SoftmaxTrainingConfig(learning_rate=0.5, max_epochs=500, target_loss=0.2)
    )
    assert target.converged and target.loss_per_epoch[-1] <= 0.2 and target.epochs < 500


def test_weight_decay_shrinks_untouched_weights_but_not_biases():
    # Feature 1 is always zero, so its weights receive no data gradient: only the penalty moves
    # them. Biases are never penalized.
    model = SoftmaxRegression(2, 2)
    model.set_parameters([0.0, 1.0, 0.0, -1.0, 0.0, 0.0])
    data = [([1.0, 0.0], 0), ([1.0, 0.0], 1)]
    model.fit(
        data,
        SoftmaxTrainingConfig(learning_rate=0.1, max_epochs=1, batch_size=0, l2_regularization=0.5),
    )
    assert model.weights[0][1] == pytest.approx(1.0 - 0.1 * 0.5 * 1.0)
    assert model.weights[1][1] == pytest.approx(-1.0 + 0.1 * 0.5 * 1.0)
    no_decay = SoftmaxRegression(2, 2)
    no_decay.set_parameters([0.0, 1.0, 0.0, -1.0, 0.0, 0.0])
    no_decay.fit(data, SoftmaxTrainingConfig(learning_rate=0.1, max_epochs=1, batch_size=0))
    assert no_decay.weights[0][1] == 1.0 and no_decay.biases == [0.0, 0.0]


def test_saturated_logits_stay_finite():
    model = SoftmaxRegression(1, 3)
    model.set_parameters([1000.0, -1000.0, 0.0, 0.0, 0.0, 0.0])
    assert model.predict_probabilities([1.0]) == pytest.approx([1.0, 0.0, 0.0])
    assert model.cross_entropy([([1.0], 1)]) == pytest.approx(2000.0)
    assert softmax([1e308, 1e308]) == [0.5, 0.5]


def test_confusion_matrix_and_metrics():
    model = SoftmaxRegression(1, 3)
    model.set_parameters([1.0, 0.0, -1.0, 0.0, 0.0, 0.0])  # Predicts 0 for x > 0, 2 for x < 0.
    data = [([1.0], 0), ([2.0], 1), ([-1.0], 2), ([-3.0], 2), ([0.5], 0)]
    confusion = model.confusion_matrix(data)
    assert confusion == [[2, 0, 0], [1, 0, 0], [0, 0, 2]]
    metrics = multiclass_metrics(confusion)
    assert metrics.accuracy == pytest.approx(4 / 5)
    assert metrics.per_class_precision == pytest.approx((2 / 3, 0.0, 1.0))
    assert metrics.per_class_recall == pytest.approx((1.0, 0.0, 1.0))
    assert metrics.per_class_f1 == pytest.approx((0.8, 0.0, 1.0))  # Never predicted: defined as 0.
    assert metrics.macro_f1 == pytest.approx(0.6)
    for bad in ([], [[1, 0]], [[0, 0], [0, 0]], [[1, -1], [0, 1]], [[1.5, 0], [0, 1]]):
        with pytest.raises(ValueError):
            multiclass_metrics(bad)


def test_divergence_and_invalid_inputs():
    with pytest.raises(ArithmeticError):
        SoftmaxRegression(1, 2).fit(
            [([1e200], 0), ([-1e200], 1)], SoftmaxTrainingConfig(learning_rate=1e200, max_epochs=5)
        )
    for build in (lambda: SoftmaxRegression(0, 2), lambda: SoftmaxRegression(2, 1)):
        with pytest.raises(ValueError):
            build()
    for kwargs in (
        {"learning_rate": 0},
        {"max_epochs": 0},
        {"batch_size": -1},
        {"seed": -1},
        {"l2_regularization": math.nan},
        {"target_loss": -1},
    ):
        with pytest.raises(ValueError):
            SoftmaxTrainingConfig(**kwargs)
    model = SoftmaxRegression(2, 2)
    for data in ([], [([1.0], 0)], [([1.0, math.inf], 0)], [([1.0, 0.0], 2)], [([1.0, 0.0], 0.5)]):
        with pytest.raises(ValueError):
            model.cross_entropy(data)
    with pytest.raises(ValueError):
        model.cross_entropy([([1.0, 0.0], 0)], l2_regularization=-1)
    with pytest.raises(ValueError):
        model.set_parameters([0.0] * 5)
    with pytest.raises(ValueError):
        model.predict([1.0])
