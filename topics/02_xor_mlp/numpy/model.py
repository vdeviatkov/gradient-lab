"""Batch matrix backpropagation for a tanh hidden layer and binary logit."""

import math
import random

from ml_scratch.pure.mlp.model import MlpParameters, MlpTrainingResult

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


class XorMlp:
    def __init__(self, hidden_units=4, learning_rate=0.5, seed=0, feature_count=2):
        positive_integer(hidden_units, "hidden_units")
        positive_integer(feature_count, "feature_count")
        positive(learning_rate, "learning_rate")
        self.hidden_units, self.feature_count = hidden_units, feature_count
        self.learning_rate, self.seed = learning_rate, seed
        rng = random.Random(seed)
        limit = math.sqrt(6 / (feature_count + hidden_units))
        self.input_weights = np.array(
            [
                [rng.uniform(-limit, limit) for _ in range(feature_count)]
                for _ in range(hidden_units)
            ]
        )
        self.hidden_biases = np.zeros(hidden_units)
        limit = math.sqrt(6 / (hidden_units + 1))
        self.output_weights = np.array([rng.uniform(-limit, limit) for _ in range(hidden_units)])
        self.output_bias = 0.0

    def _forward(self, x):
        h = np.tanh(x @ self.input_weights.T + self.hidden_biases)
        return h, h @ self.output_weights + self.output_bias

    def hidden_activations(self, features):
        return self._forward(vector(features, self.feature_count))[0]

    def logit(self, features):
        return float(self._forward(vector(features, self.feature_count))[1])

    def predict_probability(self, features):
        return float(sigmoid(self.logit(features)))

    def predict(self, features):
        return int(self.logit(features) >= 0)

    def predict_many(self, examples):
        return (self._forward(matrix(list(examples), self.feature_count))[1] >= 0).astype(int)

    def binary_cross_entropy(self, data):
        x, y = dataset(data, self.feature_count, binary=True)
        return float(binary_loss(self._forward(x)[1], y).mean())

    def accuracy(self, data):
        x, y = dataset(data, self.feature_count, binary=True)
        return float(np.mean((self._forward(x)[1] >= 0) == y))

    def _gradient(self, x, y):
        h, scores = self._forward(x)
        delta = (sigmoid(scores) - y) / len(x)
        hidden_delta = delta[:, None] * self.output_weights * (1 - h * h)
        return MlpParameters(
            hidden_delta.T @ x, hidden_delta.sum(axis=0), h.T @ delta, float(delta.sum())
        )

    def gradient(self, data):
        return self._gradient(*dataset(data, self.feature_count, binary=True))

    def numerical_gradient(self, data, epsilon=1e-5):
        positive(epsilon, "epsilon")
        snapshot = self.parameters
        gradients = []
        try:
            for name in ("input_weights", "hidden_biases", "output_weights"):
                values = getattr(self, name)
                gradient = np.empty_like(values)
                for index in np.ndindex(values.shape):
                    original = values[index]
                    values[index] = original + epsilon
                    above = self.binary_cross_entropy(data)
                    values[index] = original - epsilon
                    below = self.binary_cross_entropy(data)
                    values[index] = original
                    gradient[index] = (above - below) / (2 * epsilon)
                gradients.append(gradient)
            self.output_bias = snapshot.output_bias + epsilon
            above = self.binary_cross_entropy(data)
            self.output_bias = snapshot.output_bias - epsilon
            below = self.binary_cross_entropy(data)
            return MlpParameters(*gradients, (above - below) / (2 * epsilon))
        finally:
            self.set_parameters(snapshot)

    def fit(self, data, *, max_epochs=10_000, target_loss=0.02):
        x, y = dataset(data, self.feature_count, binary=True)
        positive_integer(max_epochs, "max_epochs")
        positive(target_loss, "target_loss")
        history = []
        for epoch in range(1, max_epochs + 1):
            g = self._gradient(x, y)
            self.input_weights -= self.learning_rate * g.input_weights
            self.hidden_biases -= self.learning_rate * g.hidden_biases
            self.output_weights -= self.learning_rate * g.output_weights
            self.output_bias -= self.learning_rate * g.output_bias
            loss = float(binary_loss(self._forward(x)[1], y).mean())
            if not np.isfinite(loss):
                raise RuntimeError("training diverged to a non-finite loss")
            history.append(loss)
            if loss <= target_loss:
                return MlpTrainingResult(epoch, True, tuple(history))
        return MlpTrainingResult(max_epochs, False, tuple(history))

    @property
    def parameters(self):
        # Immutable snapshots use the same public value type as the scalar reference.
        return MlpParameters(
            tuple(map(tuple, self.input_weights)),
            tuple(self.hidden_biases),
            tuple(self.output_weights),
            float(self.output_bias),
        )

    def set_parameters(self, parameters):
        w = matrix(parameters.input_weights, self.feature_count)
        if len(w) != self.hidden_units:
            raise ValueError("input_weights must have one row per hidden unit")
        b = vector(parameters.hidden_biases, self.hidden_units)
        v = vector(parameters.output_weights, self.hidden_units)
        if not np.isfinite(parameters.output_bias):
            raise ValueError("output bias must be finite")
        self.input_weights, self.hidden_biases, self.output_weights = w.copy(), b.copy(), v.copy()
        self.output_bias = float(parameters.output_bias)
