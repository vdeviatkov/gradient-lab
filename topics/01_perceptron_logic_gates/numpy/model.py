"""Online perceptron with vectorized feature updates and batch prediction."""

import random

from ml_scratch.pure.perceptron.model import TrainingResult

import numpy as np
from ml_scratch.numpy.validation import dataset, matrix, positive, positive_integer, vector


class Perceptron:
    def __init__(self, feature_count, learning_rate=0.1, seed=0):
        positive_integer(feature_count, "feature_count")
        positive(learning_rate, "learning_rate")
        self.feature_count, self.learning_rate, self.seed = feature_count, learning_rate, seed
        self._random = random.Random(seed)
        self.weights = np.array([self._random.uniform(-0.5, 0.5) for _ in range(feature_count)])
        self.bias = self._random.uniform(-0.5, 0.5)

    def decision_score(self, features):
        return float(vector(features, self.feature_count) @ self.weights + self.bias)

    def predict(self, features):
        return int(self.decision_score(features) >= 0)

    def predict_many(self, examples):
        return (matrix(list(examples), self.feature_count) @ self.weights + self.bias >= 0).astype(
            int
        )

    def fit(self, data, *, max_epochs=100, shuffle=True):
        x, y = dataset(data, self.feature_count, binary=True)
        positive_integer(max_epochs, "max_epochs")
        order = list(range(len(x)))
        history = []
        for epoch in range(1, max_epochs + 1):
            if shuffle:
                self._random.shuffle(order)
            mistakes = 0
            # Online updates must remain sequential: later predictions see earlier updates.
            for i in order:
                error = y[i] - int(x[i] @ self.weights + self.bias >= 0)
                if error:
                    self.weights += self.learning_rate * error * x[i]
                    self.bias += self.learning_rate * error
                    mistakes += 1
            history.append(mistakes)
            if mistakes == 0:
                return TrainingResult(epoch, True, tuple(history))
        return TrainingResult(max_epochs, False, tuple(history))

    def accuracy(self, data):
        x, y = dataset(data, self.feature_count, binary=True)
        return float(np.mean((x @ self.weights + self.bias >= 0) == y))
