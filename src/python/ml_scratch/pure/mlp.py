"""A dependency-free two-layer network, the smallest model that can represent XOR."""

from __future__ import annotations

import math
import random
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from ml_scratch.pure.datasets import Dataset, Example, Features


@dataclass(frozen=True)
class MlpTrainingResult:
    """Describe a completed training attempt without hiding non-convergence."""

    epochs: int
    converged: bool
    loss_per_epoch: tuple[float, ...]


@dataclass(frozen=True)
class MlpParameters:
    """Every learnable value, in a shape that can be read, compared, and set by hand."""

    input_weights: tuple[tuple[float, ...], ...]
    hidden_biases: tuple[float, ...]
    output_weights: tuple[float, ...]
    output_bias: float


def _sigmoid(value: float) -> float:
    """Logistic function, written so neither branch can overflow."""
    if value >= 0.0:
        return 1.0 / (1.0 + math.exp(-value))
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


def _softplus(value: float) -> float:
    """``log(1 + e**v)`` without overflow, used to write the loss in terms of the logit."""
    return max(value, 0.0) + math.log1p(math.exp(-abs(value)))


@dataclass
class XorMlp:
    """Two-layer network: tanh hidden units, one logit output, binary cross-entropy.

    "Two-layer" counts the two trainable affine maps, input-to-hidden and hidden-to-output. The
    hidden layer is what the perceptron lacked: with a nonlinearity between two linear maps the
    model can carve the input space with several boundaries instead of one, which is exactly what
    XOR needs. How many boundaries it gets is ``hidden_units``, and the experiment measures what
    happens when there are too few.

    Gradients are derived by hand from the chain rule. The output layer keeps its raw score and
    the sigmoid is applied inside the loss, so the output delta collapses to ``probability -
    target`` with no separate activation slope, and the loss is computed in a form that cannot
    overflow.
    """

    hidden_units: int = 4
    learning_rate: float = 0.5
    seed: int = 0
    input_weights: list[list[float]] = field(init=False)
    hidden_biases: list[float] = field(init=False)
    output_weights: list[float] = field(init=False)
    output_bias: float = field(init=False)
    _random: random.Random = field(init=False, repr=False)

    feature_count: int = 2

    def __post_init__(self) -> None:
        if self.hidden_units <= 0:
            raise ValueError("hidden_units must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.feature_count <= 0:
            raise ValueError("feature_count must be positive")

        self._random = random.Random(self.seed)
        # Glorot scaling: a limit of sqrt(6 / (fan_in + fan_out)) keeps the variance of the
        # activations and of the gradients roughly equal through the layer, which is what stops a
        # tanh unit from starting saturated.
        hidden_limit = math.sqrt(6.0 / (self.feature_count + self.hidden_units))
        output_limit = math.sqrt(6.0 / (self.hidden_units + 1))
        self.input_weights = [
            [self._random.uniform(-hidden_limit, hidden_limit) for _ in range(self.feature_count)]
            for _ in range(self.hidden_units)
        ]
        # Biases start at zero: the weights already break the symmetry between units, and a
        # zero bias is the one choice that adds no opinion of its own.
        self.hidden_biases = [0.0] * self.hidden_units
        self.output_weights = [
            self._random.uniform(-output_limit, output_limit) for _ in range(self.hidden_units)
        ]
        self.output_bias = 0.0

    # -- forward ---------------------------------------------------------------------------

    def hidden_activations(self, features: Sequence[float]) -> tuple[float, ...]:
        """Return the tanh activation of every hidden unit."""
        self._validate_features(features)
        return tuple(
            math.tanh(
                sum(weight * value for weight, value in zip(weights, features, strict=True)) + bias
            )
            for weights, bias in zip(self.input_weights, self.hidden_biases, strict=True)
        )

    def logit(self, features: Sequence[float]) -> float:
        """Return the raw output score, before the sigmoid the loss applies."""
        activations = self.hidden_activations(features)
        return (
            sum(
                weight * activation
                for weight, activation in zip(self.output_weights, activations, strict=True)
            )
            + self.output_bias
        )

    def predict_probability(self, features: Sequence[float]) -> float:
        """Return the probability the model assigns to class 1."""
        return _sigmoid(self.logit(features))

    def predict(self, features: Sequence[float]) -> int:
        """Predict class 0 or 1 by thresholding the probability at one half."""
        return int(self.predict_probability(features) >= 0.5)

    def predict_many(self, examples: Iterable[Features]) -> tuple[int, ...]:
        """Predict several feature vectors in their given order."""
        return tuple(self.predict(features) for features in examples)

    # -- objective -------------------------------------------------------------------------

    def binary_cross_entropy(self, dataset: Dataset) -> float:
        """Return the mean cross-entropy, written through the logit so it cannot overflow."""
        examples = self._validated(dataset)
        total = 0.0
        for features, target in examples:
            score = self.logit(features)
            # softplus(z) - y * z is -log p written so that neither branch overflows.
            total += _softplus(score) - target * score
        return total / len(examples)

    def accuracy(self, dataset: Dataset) -> float:
        """Return the fraction of correctly classified examples."""
        examples = self._validated(dataset)
        return sum(self.predict(features) == target for features, target in examples) / len(
            examples
        )

    # -- gradients -------------------------------------------------------------------------

    def gradient(self, dataset: Dataset) -> MlpParameters:
        """Return the mean gradient of the loss, derived by hand from the chain rule.

        For one example, with ``h`` the hidden activations and ``p`` the output probability:

        * ``dL/dz = p - y`` for the output score ``z``, because composing the sigmoid with the
          cross-entropy cancels the activation's slope;
        * ``dL/dW_out[j] = (p - y) * h[j]`` and ``dL/db_out = p - y``;
        * ``dL/da[j] = (p - y) * W_out[j] * (1 - h[j]**2)`` through the tanh, whose derivative is
          ``1 - tanh**2``;
        * ``dL/dW_in[j][k] = dL/da[j] * x[k]`` and ``dL/db_in[j] = dL/da[j]``.
        """
        examples = self._validated(dataset)
        scale = 1.0 / len(examples)

        input_weights = [[0.0] * self.feature_count for _ in range(self.hidden_units)]
        hidden_biases = [0.0] * self.hidden_units
        output_weights = [0.0] * self.hidden_units
        output_bias = 0.0

        for features, target in examples:
            activations = self.hidden_activations(features)
            score = (
                sum(
                    weight * activation
                    for weight, activation in zip(self.output_weights, activations, strict=True)
                )
                + self.output_bias
            )
            output_delta = _sigmoid(score) - target

            output_bias += scale * output_delta
            for unit, activation in enumerate(activations):
                output_weights[unit] += scale * output_delta * activation
                hidden_delta = (
                    output_delta * self.output_weights[unit] * (1.0 - activation * activation)
                )
                hidden_biases[unit] += scale * hidden_delta
                for index, value in enumerate(features):
                    input_weights[unit][index] += scale * hidden_delta * value

        return MlpParameters(
            tuple(tuple(row) for row in input_weights),
            tuple(hidden_biases),
            tuple(output_weights),
            output_bias,
        )

    def numerical_gradient(self, dataset: Dataset, epsilon: float = 1e-5) -> MlpParameters:
        """Return the same gradient estimated by central differences, to verify ``gradient``.

        Nothing in training uses this: it exists so a test can check the hand-derived gradient
        against a method that knows nothing about the derivation. The backpropagation milestone
        turns the idea into a general facility; here it is the smallest useful version of it.
        """
        def perturbed(read, write) -> float:
            original = read()
            write(original + epsilon)
            above = self.binary_cross_entropy(dataset)
            write(original - epsilon)
            below = self.binary_cross_entropy(dataset)
            write(original)
            return (above - below) / (2.0 * epsilon)

        input_weights = tuple(
            tuple(
                perturbed(
                    lambda unit=unit, index=index: self.input_weights[unit][index],
                    lambda value, unit=unit, index=index: self.input_weights[unit].__setitem__(
                        index, value
                    ),
                )
                for index in range(self.feature_count)
            )
            for unit in range(self.hidden_units)
        )
        hidden_biases = tuple(
            perturbed(
                lambda unit=unit: self.hidden_biases[unit],
                lambda value, unit=unit: self.hidden_biases.__setitem__(unit, value),
            )
            for unit in range(self.hidden_units)
        )
        output_weights = tuple(
            perturbed(
                lambda unit=unit: self.output_weights[unit],
                lambda value, unit=unit: self.output_weights.__setitem__(unit, value),
            )
            for unit in range(self.hidden_units)
        )
        output_bias = perturbed(
            lambda: self.output_bias,
            lambda value: setattr(self, "output_bias", value),
        )
        return MlpParameters(input_weights, hidden_biases, output_weights, output_bias)

    # -- training --------------------------------------------------------------------------

    def fit(
        self, dataset: Dataset, *, max_epochs: int = 10_000, target_loss: float = 0.02
    ) -> MlpTrainingResult:
        """Train by full-batch gradient descent until the loss target or the epoch budget.

        Full batch, not stochastic: the dataset is the complete four-row truth table, so there is
        nothing to sample and every epoch sees the same gradient. That makes the run a
        deterministic function of the seed alone.
        """
        self._validated(dataset)
        if max_epochs <= 0:
            raise ValueError("max_epochs must be positive")
        if target_loss <= 0:
            raise ValueError("target_loss must be positive")

        history: list[float] = []
        for epoch in range(1, max_epochs + 1):
            gradient = self.gradient(dataset)
            for unit in range(self.hidden_units):
                for index in range(self.feature_count):
                    self.input_weights[unit][index] -= (
                        self.learning_rate * gradient.input_weights[unit][index]
                    )
                self.hidden_biases[unit] -= self.learning_rate * gradient.hidden_biases[unit]
                self.output_weights[unit] -= self.learning_rate * gradient.output_weights[unit]
            self.output_bias -= self.learning_rate * gradient.output_bias

            loss = self.binary_cross_entropy(dataset)
            history.append(loss)
            if loss <= target_loss:
                return MlpTrainingResult(epoch, True, tuple(history))

        return MlpTrainingResult(max_epochs, False, tuple(history))

    # -- parameters ------------------------------------------------------------------------

    @property
    def parameters(self) -> MlpParameters:
        """Return an immutable snapshot of every learnable value."""
        return MlpParameters(
            tuple(tuple(row) for row in self.input_weights),
            tuple(self.hidden_biases),
            tuple(self.output_weights),
            self.output_bias,
        )

    def set_parameters(self, parameters: MlpParameters) -> None:
        """Replace every learnable value, so a test can start from a known position."""
        if len(parameters.input_weights) != self.hidden_units:
            raise ValueError("input_weights must have one row per hidden unit")
        if any(len(row) != self.feature_count for row in parameters.input_weights):
            raise ValueError("every input_weights row must have one entry per feature")
        if (
            len(parameters.hidden_biases) != self.hidden_units
            or len(parameters.output_weights) != self.hidden_units
        ):
            raise ValueError("biases and output weights must have one entry per hidden unit")

        self.input_weights = [list(row) for row in parameters.input_weights]
        self.hidden_biases = list(parameters.hidden_biases)
        self.output_weights = list(parameters.output_weights)
        self.output_bias = parameters.output_bias

    # -- validation ------------------------------------------------------------------------

    def _validate_features(self, features: Sequence[float]) -> None:
        if len(features) != self.feature_count:
            raise ValueError(f"expected {self.feature_count} features, received {len(features)}")

    def _validated(self, dataset: Dataset) -> list[Example]:
        examples = list(dataset)
        if not examples:
            raise ValueError("dataset must not be empty")
        for features, target in examples:
            self._validate_features(features)
            if target not in (0, 1):
                raise ValueError("targets must be binary values: 0 or 1")
        return examples
