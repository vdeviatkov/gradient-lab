"""Weighted binary logistic regression with explicit array gradients."""

import random

from ml_scratch.pure.logistic_regression.model import (
    ClassificationMetrics,
    ConfusionMatrix,
    LogisticTrainingConfig,
    LogisticTrainingResult,
    ThresholdSelectionResult,
)

import numpy as np
from ml_scratch.numpy.validation import (
    binary_loss,
    dataset,
    matrix,
    positive,
    positive_integer,
    sigmoid,
    vector,
)


def _probabilities(values):
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError("probabilities and thresholds must be finite and in [0, 1]")
    return values


def classification_metrics(targets, predictions):
    y, p = np.asarray(targets), np.asarray(predictions)
    if y.ndim != 1 or not y.size or y.shape != p.shape:
        raise ValueError("targets and predictions must be nonempty vectors of equal length")
    if not np.isin(y, [0, 1]).all() or not np.isin(p, [0, 1]).all():
        raise ValueError("labels must be binary")
    tn, fp, fn, tp = (int(np.sum((y == a) & (p == b))) for a, b in ((0, 0), (0, 1), (1, 0), (1, 1)))
    return ClassificationMetrics(
        ConfusionMatrix(tn, fp, fn, tp),
        (tn + tp) / len(y),
        tp / (tp + fp) if tp + fp else 0.0,
        tp / (tp + fn) if tp + fn else 0.0,
        2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
    )


def select_threshold_for_f1(targets, probabilities, candidate_thresholds):
    p, thresholds = _probabilities(probabilities), _probabilities(candidate_thresholds)
    if p.ndim != 1 or thresholds.ndim != 1 or not thresholds.size:
        raise ValueError(
            "probabilities and candidate thresholds must be vectors; candidates nonempty"
        )
    results = [
        ThresholdSelectionResult(float(t), classification_metrics(targets, p >= t))
        for t in thresholds
    ]
    return min(results, key=lambda r: (-r.metrics.f1, abs(r.threshold - 0.5), r.threshold))


class BinaryLogisticRegression:
    def __init__(self, feature_count):
        positive_integer(feature_count, "feature_count")
        self.feature_count = feature_count
        self.weights, self.bias = np.zeros(feature_count), 0.0

    def decision_score(self, features):
        return float(vector(features, self.feature_count) @ self.weights + self.bias)

    def predict_probability(self, features):
        return float(sigmoid(self.decision_score(features)))

    def predict_probabilities(self, features):
        return sigmoid(matrix(features, self.feature_count) @ self.weights + self.bias)

    def predict(self, features, threshold=0.5):
        _probabilities(threshold)
        return int(self.predict_probability(features) >= threshold)

    def binary_cross_entropy(self, data, positive_class_weight=1.0):
        positive(positive_class_weight, "positive_class_weight")
        x, y = dataset(data, self.feature_count, binary=True)
        return float(binary_loss(x @ self.weights + self.bias, y, positive_class_weight).mean())

    def _gradient(self, x, y, weight):
        delta = (sigmoid(x @ self.weights + self.bias) - y) * np.where(y == 1, weight, 1)
        return x.T @ delta / len(x), float(delta.mean())

    def gradient(self, data, positive_class_weight=1.0):
        positive(positive_class_weight, "positive_class_weight")
        x, y = dataset(data, self.feature_count, binary=True)
        return self._gradient(x, y, positive_class_weight)

    def evaluate(self, data, threshold=0.5):
        _probabilities(threshold)
        x, y = dataset(data, self.feature_count, binary=True)
        return classification_metrics(y, self.predict_probabilities(x) >= threshold)

    def fit(self, data, config=LogisticTrainingConfig()):
        x, y = dataset(data, self.feature_count, binary=True)
        positive(config.positive_class_weight, "positive_class_weight")
        positive(config.learning_rate, "learning_rate")
        positive_integer(config.max_epochs, "max_epochs")
        if type(config.batch_size) is not int or config.batch_size < 0:
            raise ValueError("batch_size must be a non-negative integer")
        if not np.isfinite(config.target_loss) or config.target_loss < 0:
            raise ValueError("target_loss must be finite and non-negative")
        self.weights, self.bias = np.zeros(self.feature_count), 0.0
        order, rng, history = list(range(len(x))), random.Random(config.seed), []
        size = min(config.batch_size or len(x), len(x))
        for epoch in range(1, config.max_epochs + 1):
            if config.shuffle and size < len(x):
                rng.shuffle(order)
            for start in range(0, len(x), size):
                indices = order[start : start + size]
                w, b = self._gradient(x[indices], y[indices], config.positive_class_weight)
                self.weights -= config.learning_rate * w
                self.bias -= config.learning_rate * b
            loss = float(
                binary_loss(x @ self.weights + self.bias, y, config.positive_class_weight).mean()
            )
            if not np.isfinite(loss):
                raise RuntimeError("training diverged to a non-finite loss")
            history.append(loss)
            if config.target_loss > 0 and loss <= config.target_loss:
                return LogisticTrainingResult(epoch, True, tuple(history))
        return LogisticTrainingResult(config.max_epochs, False, tuple(history))
