"""Input contracts shared by this topic's two unsupervised algorithms."""

import math
from collections.abc import Sequence

Matrix = Sequence[Sequence[float]]


def positive_integer(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def feature_matrix(data: Matrix) -> tuple[tuple[float, ...], ...]:
    if not data or not data[0]:
        raise ValueError("data must not be empty and must have features")
    width = len(data[0])
    if any(len(row) != width for row in data):
        raise ValueError("data must be rectangular")
    rows = tuple(tuple(float(value) for value in row) for row in data)
    if any(not math.isfinite(value) for row in rows for value in row):
        raise ValueError("features must be finite")
    return rows


def vector(values: Sequence[float], size: int) -> tuple[float, ...]:
    if len(values) != size:
        raise ValueError(f"expected {size} values")
    result = tuple(float(value) for value in values)
    if any(not math.isfinite(value) for value in result):
        raise ValueError("values must be finite")
    return result


def squared_distance(left: Sequence[float], right: Sequence[float]) -> float:
    total = sum((x - y) * (x - y) for x, y in zip(left, right, strict=True))
    if not math.isfinite(total):
        raise ArithmeticError("squared distance overflowed; rescale the features")
    return total
