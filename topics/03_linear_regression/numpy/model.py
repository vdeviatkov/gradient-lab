"""Vectorized least squares with explicit elimination and batch/SGD/mini-batch descent."""

import random

from ml_scratch.pure.linear_regression.model import (
    Optimizer,
    TrainingConfig,
    TrainingResult,
)

import numpy as np
from ml_scratch.numpy.validation import dataset, matrix, positive, positive_integer, vector


def solve_linear_system(matrix_values, vector_values):
    """Gaussian elimination with partial pivoting; no call to a linear-system solver."""
    a = matrix(matrix_values).copy()
    size = len(a)
    if a.shape != (size, size):
        raise ValueError("matrix must be square")
    b = vector(vector_values, size).copy()
    for pivot in range(size):
        best = pivot + int(np.argmax(np.abs(a[pivot:, pivot])))
        if abs(a[best, pivot]) <= 1e-12:
            raise ValueError("the system is singular to working precision")
        a[[pivot, best]] = a[[best, pivot]]
        b[[pivot, best]] = b[[best, pivot]]
        factors = a[pivot + 1 :, pivot] / a[pivot, pivot]
        a[pivot + 1 :, pivot:] -= factors[:, None] * a[pivot, pivot:]
        b[pivot + 1 :] -= factors * b[pivot]
    solution = np.zeros(size)
    for row in range(size - 1, -1, -1):
        solution[row] = (b[row] - a[row, row + 1 :] @ solution[row + 1 :]) / a[row, row]
    return solution


def _design(data):
    examples = list(data)
    if not examples:
        raise ValueError("dataset must not be empty")
    x, y = dataset(examples, len(examples[0][0]))
    return np.column_stack((np.ones(len(x)), x)), y


def hessian(data):
    x, _ = _design(data)
    return 2 / len(x) * (x.T @ x)


def _power_iteration(a, iterations, tolerance):
    positive_integer(iterations, "iterations")
    positive(tolerance, "tolerance")
    v, eigenvalue = np.ones(len(a)), 0.0
    for _ in range(iterations):
        product = a @ v
        norm = float(np.sqrt(product @ product))
        if norm == 0:
            return 0.0
        v = product / norm
        if abs(norm - eigenvalue) <= tolerance * max(norm, 1):
            return norm
        eigenvalue = norm
    return eigenvalue


def hessian_eigenvalue_range(data, *, iterations=10_000, tolerance=1e-15):
    a = hessian(data)
    largest = _power_iteration(a, iterations, tolerance)
    gap = _power_iteration(largest * np.eye(len(a)) - a, iterations, tolerance)
    return max(0.0, largest - gap), largest


def largest_hessian_eigenvalue(data, *, iterations=10_000, tolerance=1e-15):
    return _power_iteration(hessian(data), iterations, tolerance)


def condition_number(data):
    smallest, largest = hessian_eigenvalue_range(data)
    return largest / smallest if smallest > 0 else float("inf")


def standardize(data):
    design, y = _design(data)
    x = design[:, 1:]
    deviation = x.std(axis=0)
    scaled = (x - x.mean(axis=0)) / np.where(deviation > 0, deviation, 1)
    return tuple((tuple(row), float(target)) for row, target in zip(scaled, y, strict=True))


class LinearRegression:
    def __init__(self, feature_count):
        positive_integer(feature_count, "feature_count")
        self.feature_count = feature_count
        self.weights, self.bias = np.zeros(feature_count), 0.0

    def predict(self, features):
        return float(vector(features, self.feature_count) @ self.weights + self.bias)

    def predict_many(self, examples):
        return matrix(list(examples), self.feature_count) @ self.weights + self.bias

    def mean_squared_error(self, data):
        x, y = dataset(data, self.feature_count)
        with np.errstate(over="ignore", invalid="ignore"):
            return float(np.mean((x @ self.weights + self.bias - y) ** 2))

    def r_squared(self, data):
        x, y = dataset(data, self.feature_count)
        total = float(np.sum((y - y.mean()) ** 2))
        if total == 0:
            raise ValueError("r_squared is undefined when every target is identical")
        return 1 - float(np.sum((x @ self.weights + self.bias - y) ** 2)) / total

    def _gradient(self, x, y):
        residual = x @ self.weights + self.bias - y
        return 2 / len(x) * (x.T @ residual), float(2 * residual.mean())

    def gradient(self, data):
        return self._gradient(*dataset(data, self.feature_count))

    def fit_closed_form(self, data):
        x, y = dataset(data, self.feature_count)
        design = np.column_stack((np.ones(len(x)), x))
        theta = solve_linear_system(design.T @ design, design.T @ y)
        self.bias, self.weights = float(theta[0]), theta[1:].copy()

    def fit(self, data, config=None):
        config = config or TrainingConfig()
        x, y = dataset(data, self.feature_count)
        positive(config.learning_rate, "learning_rate")
        positive_integer(config.max_epochs, "max_epochs")
        positive_integer(config.batch_size, "batch_size")
        if not isinstance(config.optimizer, Optimizer):
            raise ValueError("unknown optimizer")
        if not np.isfinite(config.target_mse) or config.target_mse < 0:
            raise ValueError("target_mse must be finite and non-negative")
        size = (
            len(x)
            if config.optimizer is Optimizer.BATCH
            else 1
            if config.optimizer is Optimizer.STOCHASTIC
            else min(config.batch_size, len(x))
        )
        rng, order, history = random.Random(config.seed), list(range(len(x))), []
        with np.errstate(over="ignore", invalid="ignore"):
            for epoch in range(1, config.max_epochs + 1):
                if config.shuffle and size < len(x):
                    rng.shuffle(order)
                for start in range(0, len(x), size):
                    indices = order[start : start + size]
                    w, b = self._gradient(x[indices], y[indices])
                    self.weights -= config.learning_rate * w
                    self.bias -= config.learning_rate * b
                loss = float(np.mean((x @ self.weights + self.bias - y) ** 2))
                history.append(loss)
                if not np.isfinite(loss):
                    return TrainingResult(epoch, False, tuple(history), diverged=True)
                if loss <= config.target_mse:
                    return TrainingResult(epoch, True, tuple(history))
        return TrainingResult(config.max_epochs, False, tuple(history))
