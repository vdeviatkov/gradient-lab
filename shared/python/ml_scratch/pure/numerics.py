"""Stable scalar functions shared by pure-Python models."""

import math


def sigmoid(value: float) -> float:
    """Logistic function, written so neither branch can overflow."""
    if value >= 0.0:
        return 1.0 / (1.0 + math.exp(-value))
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


def softplus(value: float) -> float:
    """``log(1 + e**v)`` without overflow, used to write the loss in terms of the logit."""
    return max(value, 0.0) + math.log1p(math.exp(-abs(value)))
