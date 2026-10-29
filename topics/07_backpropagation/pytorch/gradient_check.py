"""Central differences over the flat parameter tensor and the norm-ratio comparison."""

import copy
import math

import torch
from ml_scratch.pure.backpropagation.gradient_check import GradientCheckResult

from ml_scratch.pytorch.validation import tensor


def _positive(value, name):
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")


@torch.no_grad()
def numerical_gradient(network, dataset, epsilon=1e-5):
    """Perturb one coordinate of a copy's flat tensor at a time; each loss is a batched pass."""
    _positive(epsilon, "epsilon")
    x, y = network._validate(dataset)
    probe = copy.deepcopy(network)
    estimate = torch.empty(network.parameter_count, dtype=torch.float64)
    for index in range(network.parameter_count):
        value = float(network._flat[index])
        probe._flat[index] = value + epsilon
        above = probe._loss(x, y)
        probe._flat[index] = value - epsilon
        below = probe._loss(x, y)
        probe._flat[index] = value
        estimate[index] = (above - below) / (2.0 * epsilon)
    return estimate


@torch.no_grad()
def forward_difference_gradient(network, dataset, epsilon=1e-5):
    """The first-order one-sided estimate, kept only to show why the central form is used."""
    _positive(epsilon, "epsilon")
    x, y = network._validate(dataset)
    base = network._loss(x, y)
    probe = copy.deepcopy(network)
    estimate = torch.empty(network.parameter_count, dtype=torch.float64)
    for index in range(network.parameter_count):
        value = float(network._flat[index])
        probe._flat[index] = value + epsilon
        estimate[index] = (probe._loss(x, y) - base) / epsilon
        probe._flat[index] = value
    return estimate


def check_gradient(analytic, numerical, tolerance=1e-7):
    a, n = tensor(analytic), tensor(numerical)
    if a.ndim != 1 or not a.numel():
        raise ValueError("gradients must be nonempty vectors")
    if a.shape != n.shape:
        raise ValueError("gradients must have equal sizes")
    _positive(tolerance, "tolerance")
    difference = (a - n).abs()
    # argmax keeps the first maximum, matching the scalar strict-improvement scan.
    worst = int(torch.argmax(difference))
    # The floor keeps two essentially-zero entries from producing a meaningless large ratio.
    elementwise = difference / torch.clamp(a.abs() + n.abs(), min=1e-8)
    scale = math.sqrt(float(a @ a)) + math.sqrt(float(n @ n))
    ratio = 0.0 if scale == 0 else math.sqrt(float(difference @ difference)) / scale
    return GradientCheckResult(
        ratio,
        float(difference[worst]),
        float(elementwise.max()),
        worst,
        float(a[worst]),
        float(n[worst]),
        ratio <= tolerance,
    )


def check_network_gradient(network, dataset, epsilon=1e-5, tolerance=1e-7):
    """Compare the network's batched backpropagated gradient with central differences."""
    return check_gradient(
        network.gradient(dataset), numerical_gradient(network, dataset, epsilon), tolerance
    )
