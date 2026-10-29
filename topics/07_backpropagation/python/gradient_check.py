"""Central-difference gradients and the norm-ratio comparison that verifies backpropagation."""

import copy
import math
from dataclasses import dataclass


@dataclass(frozen=True)
class GradientCheckResult:
    # ||a - n|| / (||a|| + ||n||): the verdict. Normalizing by the whole gradient keeps a
    # near-zero entry from reporting a large error, but the ratio degenerates when the entire
    # gradient vanishes, as at a minimum; read max_absolute_error there.
    relative_error: float
    # Diagnostics that locate a bug once the norm ratio says there is one.
    max_absolute_error: float
    max_elementwise_relative_error: float
    worst_parameter: int
    analytic_value: float
    numerical_value: float
    passed: bool


def _positive(value, name):
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")


def numerical_gradient(network, dataset, epsilon=1e-5):
    """(L(theta + eps e_i) - L(theta - eps e_i)) / (2 eps) for every i, on a perturbed copy."""
    _positive(epsilon, "epsilon")
    network.loss(dataset)  # Validate before the loop rather than once per perturbation.
    probe = copy.deepcopy(network)
    original = network.parameters()
    values = list(original)
    estimate = []
    for index, value in enumerate(original):
        values[index] = value + epsilon
        probe.set_parameters(values)
        above = probe.loss(dataset)
        values[index] = value - epsilon
        probe.set_parameters(values)
        below = probe.loss(dataset)
        values[index] = value
        estimate.append((above - below) / (2.0 * epsilon))
    return estimate


def forward_difference_gradient(network, dataset, epsilon=1e-5):
    """The first-order one-sided estimate, kept only to show why the central form is used."""
    _positive(epsilon, "epsilon")
    base = network.loss(dataset)
    probe = copy.deepcopy(network)
    original = network.parameters()
    values = list(original)
    estimate = []
    for index, value in enumerate(original):
        values[index] = value + epsilon
        probe.set_parameters(values)
        estimate.append((probe.loss(dataset) - base) / epsilon)
        values[index] = value
    return estimate


def check_gradient(analytic, numerical, tolerance=1e-7):
    analytic, numerical = [float(v) for v in analytic], [float(v) for v in numerical]
    if not analytic:
        raise ValueError("gradients must not be empty")
    if len(analytic) != len(numerical):
        raise ValueError("gradients must have equal sizes")
    _positive(tolerance, "tolerance")
    difference_norm = analytic_norm = numerical_norm = 0.0
    worst, worst_difference, elementwise = 0, 0.0, 0.0
    for index, (a, n) in enumerate(zip(analytic, numerical, strict=True)):
        difference = abs(a - n)
        difference_norm += difference * difference
        analytic_norm += a * a
        numerical_norm += n * n
        if difference > worst_difference:
            worst, worst_difference = index, difference
        # The floor keeps two essentially-zero entries from dividing a tiny difference by an
        # equally tiny denominator.
        elementwise = max(elementwise, difference / max(abs(a) + abs(n), 1e-8))
    scale = math.sqrt(analytic_norm) + math.sqrt(numerical_norm)
    # An all-zero pair is a perfect match, not a division by zero.
    ratio = 0.0 if scale == 0 else math.sqrt(difference_norm) / scale
    return GradientCheckResult(
        ratio,
        worst_difference,
        elementwise,
        worst,
        analytic[worst],
        numerical[worst],
        ratio <= tolerance,
    )


def check_network_gradient(network, dataset, epsilon=1e-5, tolerance=1e-7):
    """Compare the network's backpropagated gradient with central differences."""
    return check_gradient(
        network.gradient(dataset), numerical_gradient(network, dataset, epsilon), tolerance
    )
