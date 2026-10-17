import math
from dataclasses import replace

import pytest
from ml_scratch.pure.logistic_regression import (
    BinaryLogisticRegression,
    ConfusionMatrix,
    LogisticTrainingConfig,
    classification_metrics,
    select_threshold_for_f1,
)

DATA = (((1.0, 2.0), 1), ((-1.0, 0.5), 0), ((0.2, -0.4), 1))


@pytest.mark.parametrize("alpha", [1.0, 3.0])
def test_weighted_gradient_matches_central_differences(alpha):
    model = BinaryLogisticRegression(2)
    model.weights, model.bias = [0.3, -0.7], 0.2
    weights, bias = model.gradient(DATA, alpha)
    epsilon = 1e-6
    for j, expected in enumerate([*weights, bias]):
        original = model.weights[j] if j < 2 else model.bias
        losses = []
        for delta in (epsilon, -epsilon):
            if j < 2:
                model.weights[j] = original + delta
            else:
                model.bias = original + delta
            losses.append(model.binary_cross_entropy(DATA, alpha))
        if j < 2:
            model.weights[j] = original
        else:
            model.bias = original
        assert expected == pytest.approx((losses[0] - losses[1]) / (2 * epsilon), abs=1e-9)


def test_stable_loss_and_probabilities_at_extreme_scores():
    model = BinaryLogisticRegression(1)
    model.weights = [1.0]
    assert model.predict_probability((1000.0,)) == 1
    assert model.predict_probability((-1000.0,)) == 0
    assert model.binary_cross_entropy((((1000.0,), 0), ((-1000.0,), 1)), 3) == 2000
    # This positive tail would cancel to zero in softplus(z) - z.
    assert model.binary_cross_entropy((((40.0,), 1),)) == pytest.approx(math.exp(-40), rel=1e-12)
    assert model.predict((0.0,)) == 1


def test_metrics_match_hand_counted_confusion():
    result = classification_metrics([0, 0, 0, 1, 1], [0, 1, 0, 1, 0])
    assert result.confusion == ConfusionMatrix(2, 1, 1, 1)
    assert result.accuracy == 0.6
    assert result.precision == result.recall == result.f1 == 0.5
    assert classification_metrics([0, 1], [0, 0]).f1 == 0
    assert classification_metrics([0], [0]).recall == 0


def test_threshold_ties_and_boundaries():
    for candidates in ([0.75, 0.25], [0.25, 0.75]):
        assert select_threshold_for_f1([1], [1.0], candidates).threshold == 0.25
    assert select_threshold_for_f1([1], [1.0], [0.25, 0.5]).threshold == 0.5
    assert select_threshold_for_f1([1], [0.5], [0.5]).metrics.f1 == 1


def test_full_batch_step_matches_hand_derivation_and_refit_resets():
    model = BinaryLogisticRegression(2)
    gradient, bias = model.gradient(DATA, 2.0)
    config = LogisticTrainingConfig(max_epochs=1, positive_class_weight=2.0)
    first = model.fit(DATA, config)
    assert model.weights == pytest.approx([-0.1 * g for g in gradient])
    assert model.bias == pytest.approx(-0.1 * bias)
    assert not first.converged
    assert model.fit(DATA, config) == first


def test_partial_batches_use_their_actual_size():
    data = (((1.0,), 1), ((2.0,), 0), ((3.0,), 1))
    expected = BinaryLogisticRegression(1)
    for batch in (data[:2], data[2:]):
        weights, bias = expected.gradient(batch)
        expected.weights[0] -= 0.1 * weights[0]
        expected.bias -= 0.1 * bias
    model = BinaryLogisticRegression(1)
    model.fit(data, LogisticTrainingConfig(max_epochs=1, batch_size=2, shuffle=False))
    assert model.weights == expected.weights
    assert model.bias == expected.bias


def test_separable_multivariate_data_converges():
    model = BinaryLogisticRegression(2)
    result = model.fit(DATA, LogisticTrainingConfig(max_epochs=2000, target_loss=0.05))
    assert result.converged
    assert result.loss_per_epoch[-1] <= 0.05
    assert model.evaluate(DATA).accuracy == 1


def test_seeded_minibatches_are_reproducible():
    config = LogisticTrainingConfig(max_epochs=20, batch_size=1, seed=7)
    a, b = BinaryLogisticRegression(2), BinaryLogisticRegression(2)
    assert a.fit(DATA, config) == b.fit(DATA, config)
    assert a.weights == b.weights
    assert a.fit(DATA, replace(config, seed=8)) != b.fit(DATA, config)


@pytest.mark.parametrize(
    "field,value",
    [
        ("learning_rate", 0),
        ("learning_rate", math.inf),
        ("max_epochs", 0),
        ("max_epochs", 1.5),
        ("batch_size", -1),
        ("batch_size", 1.5),
        ("positive_class_weight", 0),
        ("positive_class_weight", math.nan),
        ("target_loss", -1),
        ("target_loss", math.nan),
    ],
)
def test_invalid_config_is_rejected(field, value):
    with pytest.raises(ValueError, match=field):
        BinaryLogisticRegression(2).fit(DATA, replace(LogisticTrainingConfig(), **{field: value}))


@pytest.mark.parametrize("data", [(), (((1.0,), 0),), (((1.0, 2.0), 2),), (((math.inf, 2.0), 0),)])
def test_invalid_dataset_is_rejected(data):
    with pytest.raises(ValueError):
        BinaryLogisticRegression(2).fit(data)


@pytest.mark.parametrize("value", [-0.1, 1.1, math.nan, math.inf])
def test_invalid_thresholds_and_probabilities(value):
    with pytest.raises(ValueError):
        BinaryLogisticRegression(1).predict((0.0,), value)
    with pytest.raises(ValueError):
        select_threshold_for_f1([1], [value], [0.5])
    with pytest.raises(ValueError):
        select_threshold_for_f1([1], [0.5], [value])


@pytest.mark.parametrize("targets,predictions", [([], []), ([0], []), ([2], [0]), ([0], [2])])
def test_invalid_metric_inputs(targets, predictions):
    with pytest.raises(ValueError):
        classification_metrics(targets, predictions)


def test_invalid_threshold_selection_inputs():
    for targets, probabilities, candidates in (([], [], [0.5]), ([1], [], [0.5]), ([1], [0.5], [])):
        with pytest.raises(ValueError):
            select_threshold_for_f1(targets, probabilities, candidates)
    with pytest.raises(ValueError):
        BinaryLogisticRegression(0)


def test_positive_weight_changes_the_learned_class_frequency():
    # Same intercept-only fixture as the C++ test: weighting positives by three
    # balances a 3:1 negative-to-positive dataset exactly.
    data = (((0.0,), 0),) * 3 + (((0.0,), 1),)
    ordinary = BinaryLogisticRegression(1)
    weighted = BinaryLogisticRegression(1)
    ordinary.fit(data)
    weighted.fit(data, LogisticTrainingConfig(positive_class_weight=3.0))
    assert ordinary.predict_probability((0.0,)) == pytest.approx(0.25, abs=1e-6)
    assert weighted.predict_probability((0.0,)) == 0.5
