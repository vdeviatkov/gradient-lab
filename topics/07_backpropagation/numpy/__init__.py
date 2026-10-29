"""NumPy batched backpropagation for dense networks, verified by numerical gradient checks."""

from ml_scratch.pure.backpropagation.model import (
    Activation,
    DenseLayer,
    Loss,
    NetworkTrainingConfig,
    NetworkTrainingResult,
)

from .gradient_check import (
    GradientCheckResult,
    check_gradient,
    check_network_gradient,
    forward_difference_gradient,
    numerical_gradient,
)
from .model import FeedForwardNetwork

__all__ = [
    "Activation",
    "DenseLayer",
    "FeedForwardNetwork",
    "GradientCheckResult",
    "Loss",
    "NetworkTrainingConfig",
    "NetworkTrainingResult",
    "check_gradient",
    "check_network_gradient",
    "forward_difference_gradient",
    "numerical_gradient",
]
