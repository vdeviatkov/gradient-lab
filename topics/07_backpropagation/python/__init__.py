"""From-scratch backpropagation for dense networks, verified by numerical gradient checks."""

from .gradient_check import (
    GradientCheckResult,
    check_gradient,
    check_network_gradient,
    forward_difference_gradient,
    numerical_gradient,
)
from .model import (
    Activation,
    DenseLayer,
    FeedForwardNetwork,
    Loss,
    NetworkTrainingConfig,
    NetworkTrainingResult,
    initial_parameters,
    mt19937,
)

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
    "initial_parameters",
    "mt19937",
    "numerical_gradient",
]
