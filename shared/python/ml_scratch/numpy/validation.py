"""Array boundaries and stable elementwise functions shared by NumPy models."""

import numpy as np


def positive_integer(value, name):
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or value <= 0
    ):
        raise ValueError(f"{name} must be a positive integer")


def positive(value, name):
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")


def matrix(data, width=None):
    rows = np.asarray(data, dtype=np.float64)
    if rows.ndim != 2 or min(rows.shape) == 0:
        raise ValueError("data must be a nonempty feature matrix")
    if width is not None and rows.shape[1] != width:
        raise ValueError(f"expected {width} features")
    if not np.isfinite(rows).all():
        raise ValueError("features must be finite")
    return rows


def vector(data, width):
    values = np.asarray(data, dtype=np.float64)
    if values.shape != (width,) or not np.isfinite(values).all():
        raise ValueError(f"expected {width} finite features")
    return values


def dataset(data, width, *, binary=False):
    examples = list(data)
    if not examples:
        raise ValueError("dataset must not be empty")
    x = matrix([row for row, _ in examples], width)
    y = np.asarray([label for _, label in examples], dtype=np.float64)
    if y.shape != (len(x),) or not np.isfinite(y).all():
        raise ValueError("targets must be finite scalars")
    if binary and not np.isin(y, [0, 1]).all():
        raise ValueError("targets must be binary")
    return x, y


def sigmoid(scores):
    """exp(-logaddexp(0, -z)) avoids evaluating an overflowing exponential branch."""
    return np.exp(-np.logaddexp(0.0, -np.asarray(scores, dtype=np.float64)))


def binary_loss(scores, targets, positive_class_weight=1.0):
    # Select the sign before softplus, avoiding softplus(z) - z cancellation.
    return np.where(targets == 1, positive_class_weight, 1.0) * np.logaddexp(
        0.0, np.where(targets == 1, -scores, scores)
    )
