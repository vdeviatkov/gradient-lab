"""Binary logistic regression with explicit gradients and held-out threshold selection."""

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass, field

from ml_scratch.pure.datasets import Dataset
from ml_scratch.pure.numerics import sigmoid as _sigmoid
from ml_scratch.pure.numerics import softplus as _softplus


def _positive(value: float, name: str) -> None:
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")


def _probability(value: float) -> None:
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("probability or threshold must be finite and between zero and one")


@dataclass(frozen=True)
class ConfusionMatrix:
    true_negatives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    true_positives: int = 0


@dataclass(frozen=True)
class ClassificationMetrics:
    confusion: ConfusionMatrix
    accuracy: float
    precision: float
    recall: float
    f1: float


def classification_metrics(
    targets: Sequence[int], predictions: Sequence[int]
) -> ClassificationMetrics:
    """Compute binary metrics; a zero denominator gives zero."""
    if not targets or len(targets) != len(predictions):
        raise ValueError("targets must not be empty and sizes must be equal")
    if any(value not in (0, 1) for value in (*targets, *predictions)):
        raise ValueError("labels must be binary values: 0 or 1")
    tn, fp, fn, tp = (
        sum(y == target and p == prediction for y, p in zip(targets, predictions, strict=True))
        for target, prediction in ((0, 0), (0, 1), (1, 0), (1, 1))
    )
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return ClassificationMetrics(
        ConfusionMatrix(tn, fp, fn, tp),
        (tn + tp) / len(targets),
        precision,
        recall,
        2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
    )


@dataclass(frozen=True)
class ThresholdSelectionResult:
    threshold: float
    metrics: ClassificationMetrics


def select_threshold_for_f1(
    targets: Sequence[int], probabilities: Sequence[float], candidate_thresholds: Sequence[float]
) -> ThresholdSelectionResult:
    """Maximize validation F1, then prefer closest to 0.5, then lowest threshold."""
    if not candidate_thresholds or len(targets) != len(probabilities):
        raise ValueError("candidates must not be empty and sizes must be equal")
    for value in (*probabilities, *candidate_thresholds):
        _probability(value)
    results = [
        ThresholdSelectionResult(
            t, classification_metrics(targets, [int(p >= t) for p in probabilities])
        )
        for t in candidate_thresholds
    ]
    return min(results, key=lambda r: (-r.metrics.f1, abs(r.threshold - 0.5), r.threshold))


@dataclass(frozen=True)
class LogisticTrainingConfig:
    learning_rate: float = 0.1
    max_epochs: int = 1_000
    batch_size: int = 0  # Zero selects full-batch descent, as in C++.
    shuffle: bool = True
    seed: int = 0
    positive_class_weight: float = 1.0
    target_loss: float = 0.0  # Zero disables early stopping.


@dataclass(frozen=True)
class LogisticTrainingResult:
    epochs: int
    converged: bool
    loss_per_epoch: tuple[float, ...]


@dataclass
class BinaryLogisticRegression:
    """An affine logit and sigmoid, initialized to zero on every fit.

    Positive-class weighting multiplies positive examples' losses, but normalization
    remains by sample count (not the sum of class weights), matching C++.
    """

    feature_count: int
    weights: list[float] = field(init=False)
    bias: float = field(init=False, default=0.0)

    def __post_init__(self) -> None:
        if not isinstance(self.feature_count, int) or self.feature_count <= 0:
            raise ValueError("feature_count must be a positive integer")
        self.weights = [0.0] * self.feature_count

    def decision_score(self, features: Sequence[float]) -> float:
        if len(features) != self.feature_count:
            raise ValueError(f"expected {self.feature_count} features")
        if any(not math.isfinite(x) for x in features):
            raise ValueError("features must be finite")
        return self.bias + sum(w * x for w, x in zip(self.weights, features, strict=True))

    def predict_probability(self, features: Sequence[float]) -> float:
        return _sigmoid(self.decision_score(features))

    def predict(self, features: Sequence[float], threshold: float = 0.5) -> int:
        _probability(threshold)
        return int(self.predict_probability(features) >= threshold)

    def _validate(self, dataset: Dataset, positive_class_weight: float = 1.0) -> None:
        _positive(positive_class_weight, "positive_class_weight")
        if not dataset:
            raise ValueError("dataset must not be empty")
        for features, target in dataset:
            self.decision_score(features)
            if target not in (0, 1):
                raise ValueError("targets must be binary values: 0 or 1")

    def binary_cross_entropy(self, dataset: Dataset, positive_class_weight: float = 1.0) -> float:
        """Use softplus(-z) for positives to avoid cancellation at large positive z."""
        self._validate(dataset, positive_class_weight)
        return sum(
            positive_class_weight * _softplus(-self.decision_score(x))
            if y
            else _softplus(self.decision_score(x))
            for x, y in dataset
        ) / len(dataset)

    def gradient(
        self, dataset: Dataset, positive_class_weight: float = 1.0
    ) -> tuple[list[float], float]:
        """Differentiate weighted BCE explicitly; no automatic differentiation."""
        self._validate(dataset, positive_class_weight)
        weights, bias = [0.0] * self.feature_count, 0.0
        for features, target in dataset:
            delta = self.predict_probability(features) - target
            if target:
                delta *= positive_class_weight
            bias += delta
            for j, value in enumerate(features):
                weights[j] += delta * value
        return [g / len(dataset) for g in weights], bias / len(dataset)

    def evaluate(self, dataset: Dataset, threshold: float = 0.5) -> ClassificationMetrics:
        self._validate(dataset)
        return classification_metrics(
            [y for _, y in dataset], [self.predict(x, threshold) for x, _ in dataset]
        )

    def fit(
        self, dataset: Dataset, config: LogisticTrainingConfig = LogisticTrainingConfig()
    ) -> LogisticTrainingResult:
        self._validate(dataset, config.positive_class_weight)
        _positive(config.learning_rate, "learning_rate")
        if not isinstance(config.max_epochs, int) or config.max_epochs <= 0:
            raise ValueError("max_epochs must be a positive integer")
        if not isinstance(config.batch_size, int) or config.batch_size < 0:
            raise ValueError("batch_size must be a non-negative integer")
        if not math.isfinite(config.target_loss) or config.target_loss < 0:
            raise ValueError("target_loss must be finite and non-negative")
        self.weights, self.bias = [0.0] * self.feature_count, 0.0
        order = list(range(len(dataset)))
        rng = random.Random(config.seed)
        batch_size = min(config.batch_size or len(dataset), len(dataset))
        history = []
        for epoch in range(1, config.max_epochs + 1):
            if config.shuffle and batch_size < len(dataset):
                rng.shuffle(order)
            for start in range(0, len(dataset), batch_size):
                batch = tuple(dataset[i] for i in order[start : start + batch_size])
                weights, bias = self.gradient(batch, config.positive_class_weight)
                for j, gradient in enumerate(weights):
                    self.weights[j] -= config.learning_rate * gradient
                self.bias -= config.learning_rate * bias
            loss = self.binary_cross_entropy(dataset, config.positive_class_weight)
            if not math.isfinite(loss):
                raise RuntimeError("training diverged to a non-finite loss")
            history.append(loss)
            if config.target_loss > 0 and loss <= config.target_loss:
                return LogisticTrainingResult(epoch, True, tuple(history))
        return LogisticTrainingResult(config.max_epochs, False, tuple(history))
