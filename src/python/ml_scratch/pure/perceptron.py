"""A dependency-free binary perceptron with deterministic initialization."""

from __future__ import annotations

import random
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from ml_scratch.pure.datasets import Dataset, Example, Features


@dataclass(frozen=True)
class TrainingResult:
    """Describe a completed training attempt without hiding non-convergence."""

    epochs: int
    converged: bool
    mistakes_per_epoch: tuple[int, ...]


@dataclass
class Perceptron:
    """Binary linear classifier trained with the classic perceptron update."""

    feature_count: int
    learning_rate: float = 0.1
    seed: int = 0
    weights: list[float] = field(init=False)
    bias: float = field(init=False)
    _random: random.Random = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.feature_count <= 0:
            raise ValueError("feature_count must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")

        self._random = random.Random(self.seed)
        self.weights = [self._random.uniform(-0.5, 0.5) for _ in range(self.feature_count)]
        self.bias = self._random.uniform(-0.5, 0.5)

    def decision_score(self, features: Sequence[float]) -> float:
        """Return the signed score used by the step activation."""
        self._validate_features(features)
        return sum(weight * value for weight, value in zip(self.weights, features, strict=True)) + (
            self.bias
        )

    def predict(self, features: Sequence[float]) -> int:
        """Predict class 0 or 1 using a zero-threshold step activation."""
        return int(self.decision_score(features) >= 0.0)

    def predict_many(self, examples: Iterable[Features]) -> tuple[int, ...]:
        """Predict several feature vectors in their given order."""
        return tuple(self.predict(features) for features in examples)

    def fit(
        self, dataset: Dataset, *, max_epochs: int = 100, shuffle: bool = True
    ) -> TrainingResult:
        """Train until a mistake-free epoch or the epoch budget is exhausted."""
        examples = list(dataset)
        self._validate_training_data(examples, max_epochs)
        history: list[int] = []

        for epoch in range(1, max_epochs + 1):
            if shuffle:
                self._random.shuffle(examples)

            mistakes = 0
            for features, target in examples:
                error = target - self.predict(features)
                if error == 0:
                    continue
                adjustment = self.learning_rate * error
                self.weights = [
                    weight + adjustment * value
                    for weight, value in zip(self.weights, features, strict=True)
                ]
                self.bias += adjustment
                mistakes += 1

            history.append(mistakes)
            if mistakes == 0:
                return TrainingResult(epoch, True, tuple(history))

        return TrainingResult(max_epochs, False, tuple(history))

    def accuracy(self, dataset: Dataset) -> float:
        """Return the fraction of correctly classified examples."""
        examples = list(dataset)
        self._validate_training_data(examples, max_epochs=1)
        correct = sum(self.predict(features) == target for features, target in examples)
        return correct / len(examples)

    def _validate_features(self, features: Sequence[float]) -> None:
        if len(features) != self.feature_count:
            raise ValueError(f"expected {self.feature_count} features, received {len(features)}")

    def _validate_training_data(self, examples: list[Example], max_epochs: int) -> None:
        if max_epochs <= 0:
            raise ValueError("max_epochs must be positive")
        if not examples:
            raise ValueError("dataset must not be empty")
        for features, target in examples:
            self._validate_features(features)
            if target not in (0, 1):
                raise ValueError("targets must be binary values: 0 or 1")
