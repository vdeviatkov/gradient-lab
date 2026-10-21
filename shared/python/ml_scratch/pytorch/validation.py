"""CPU float64 input boundaries and stable differentiable scalar primitives."""

import math
from numbers import Integral

import torch


def positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def positive(value, name):
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")


def tensor(data):
    """Normalize external data without retaining an input autograd graph."""
    try:
        if isinstance(data, (list, tuple)) and data and isinstance(data[0], torch.Tensor):
            return torch.stack([tensor(value) for value in data])
        return torch.as_tensor(data, dtype=torch.float64, device="cpu").detach()
    except (TypeError, RuntimeError) as error:
        raise ValueError("expected rectangular numeric data") from error


def zeros(*shape):
    return torch.zeros(*shape, dtype=torch.float64, device="cpu")


def ones(*shape):
    return torch.ones(*shape, dtype=torch.float64, device="cpu")


def matrix(data, width=None):
    values = tensor(data)
    if values.ndim != 2 or min(values.shape) == 0:
        raise ValueError("data must be a nonempty feature matrix")
    if width is not None and values.shape[1] != width:
        raise ValueError(f"expected {width} features")
    if not torch.isfinite(values).all():
        raise ValueError("features must be finite")
    return values


def vector(data, width):
    values = tensor(data)
    if values.shape != (width,) or not torch.isfinite(values).all():
        raise ValueError(f"expected {width} finite features")
    return values


def dataset(data, width, *, binary=False):
    examples = list(data)
    if not examples:
        raise ValueError("dataset must not be empty")
    x = matrix([row for row, _ in examples], width)
    y = tensor([target for _, target in examples])
    if y.shape != (len(x),) or not torch.isfinite(y).all():
        raise ValueError("targets must be finite scalars")
    if binary and not ((y == 0) | (y == 1)).all():
        raise ValueError("targets must be binary")
    return x, y


def sigmoid(scores):
    # Unlike tensor(), this primitive must preserve a graph for autograd checks.
    scores = torch.as_tensor(scores, dtype=torch.float64, device="cpu")
    return torch.exp(-torch.logaddexp(torch.zeros_like(scores), -scores))


def binary_loss(scores, targets, positive_class_weight=1.0):
    signed = torch.where(targets == 1, -scores, scores)
    weights = torch.where(
        targets == 1, torch.full_like(scores, positive_class_weight), torch.ones_like(scores)
    )
    return weights * torch.logaddexp(torch.zeros_like(scores), signed)
