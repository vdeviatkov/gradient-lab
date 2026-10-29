"""Gradient-check every architecture, sweep the step size, then train on interleaved spirals."""

import argparse
import math
from dataclasses import dataclass, replace

from . import gradient_check
from .model import Activation, DenseLayer, FeedForwardNetwork, Loss, NetworkTrainingConfig

SEED = 20260910
NETWORK_SEED = 20260910
CLASS_COUNT = 3
SAMPLES_PER_CLASS = 150
EPOCHS = 4000
EPSILONS = (1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10)
CORRUPTED_PARAMETER = 7

REGRESSION_DATA = (
    ((0.3, -0.7), (0.4, -0.2)),
    ((-0.5, 0.9), (-0.1, 0.8)),
    ((0.8, 0.2), (0.6, 0.1)),
)
BINARY_DATA = (
    ((0.0, 0.0), (0.0,)),
    ((0.0, 1.0), (1.0,)),
    ((1.0, 0.0), (1.0,)),
    ((1.0, 1.0), (0.0,)),
)

_I, _S, _T, _R = Activation.IDENTITY, Activation.SIGMOID, Activation.TANH, Activation.RELU
CONFIGURATIONS = (
    ("linear, squared error", (DenseLayer(2, 2, _I),), Loss.MEAN_SQUARED_ERROR),
    (
        "tanh hidden, squared error",
        (DenseLayer(2, 5, _T), DenseLayer(5, 2, _I)),
        Loss.MEAN_SQUARED_ERROR,
    ),
    (
        "sigmoid output, squared error",
        (DenseLayer(2, 4, _T), DenseLayer(4, 2, _S)),
        Loss.MEAN_SQUARED_ERROR,
    ),
    (
        "binary cross-entropy",
        (DenseLayer(2, 4, _T), DenseLayer(4, 1, _I)),
        Loss.BINARY_CROSS_ENTROPY,
    ),
    (
        "softmax cross-entropy",
        (DenseLayer(2, 8, _T), DenseLayer(8, 3, _I)),
        Loss.SOFTMAX_CROSS_ENTROPY,
    ),
    (
        "three hidden layers, mixed",
        (DenseLayer(2, 6, _T), DenseLayer(6, 6, _S), DenseLayer(6, 4, _T), DenseLayer(4, 3, _I)),
        Loss.SOFTMAX_CROSS_ENTROPY,
    ),
    (
        "ReLU hidden layers",
        (DenseLayer(2, 6, _R), DenseLayer(6, 6, _R), DenseLayer(6, 3, _I)),
        Loss.SOFTMAX_CROSS_ENTROPY,
    ),
)
PROBE_LAYERS = (DenseLayer(2, 8, _T), DenseLayer(8, 3, _I))
LINEAR_LAYERS = (DenseLayer(2, 3, _I),)
MLP_LAYERS = (DenseLayer(2, 16, _T), DenseLayer(16, 16, _T), DenseLayer(16, 3, _I))


class _SplitMix64:
    """Dataset-only stream matching ml_scratch::DeterministicRandom, including Box-Muller."""

    def __init__(self, seed):
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**64:
            raise ValueError("seed must be a uint64 integer")
        self.state = seed

    def uniform(self):
        mask = 2**64 - 1
        self.state = (self.state + 0x9E3779B97F4A7C15) & mask
        value = self.state
        value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & mask
        value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & mask
        value ^= value >> 31
        return (value >> 11) / 2**53

    def gaussian(self):
        first = max(self.uniform(), 1e-12)
        second = self.uniform()
        return math.sqrt(-2 * math.log(first)) * math.cos(2 * math.pi * second)


@dataclass(frozen=True)
class Splits:
    train: tuple
    validation: tuple
    test: tuple


def build_spiral(seed=SEED):
    """Three interleaved arms; three of every five samples train, then one validates, one tests.

    The index sweeps position along an arm, so a range split would give each split a disjoint
    piece of the spiral and measure extrapolation instead of generalization.
    """
    rng = _SplitMix64(seed)
    splits = ([], [], [])
    for label in range(CLASS_COUNT):
        for index in range(SAMPLES_PER_CLASS):
            position = index / (SAMPLES_PER_CLASS - 1)
            radius = 0.2 + 0.8 * position
            angle = 4.0 * position + 2.0 * math.pi * label / CLASS_COUNT + 0.14 * rng.gaussian()
            targets = tuple(1.0 if k == label else 0.0 for k in range(CLASS_COUNT))
            phase = index % 5
            split = splits[0] if phase < 3 else splits[1] if phase == 3 else splits[2]
            split.append(((radius * math.sin(angle), radius * math.cos(angle)), targets))
    return Splits(*(tuple(split) for split in splits))


def majority_baseline(splits):
    counts = [sum(targets[k] == 1.0 for _, targets in splits.train) for k in range(CLASS_COUNT)]
    majority = max(range(CLASS_COUNT), key=counts.__getitem__)
    return sum(targets[majority] == 1.0 for _, targets in splits.test) / len(splits.test)


@dataclass(frozen=True)
class ConfigurationCheck:
    name: str
    parameters: int
    result: gradient_check.GradientCheckResult


@dataclass(frozen=True)
class StepSize:
    epsilon: float
    central: float
    forward: float


@dataclass(frozen=True)
class TrainingReport:
    parameters: int
    final_loss: float
    epochs: int
    train: float
    validation: float
    test: float


@dataclass(frozen=True)
class TrajectoryCheck:
    name: str
    result: object
    largest_gradient: float


@dataclass(frozen=True)
class ExperimentResult:
    configurations: tuple[ConfigurationCheck, ...]
    step_sizes: tuple[StepSize, ...]
    detection: object
    baseline: float
    linear: TrainingReport
    mlp: TrainingReport
    trajectory: tuple[TrajectoryCheck, ...]


def run_with(network_type, checks, seed=SEED, epochs=EPOCHS):
    """Run the protocol against any backend exposing this package's network and check API."""
    splits = build_spiral(seed)
    data_for = {
        Loss.MEAN_SQUARED_ERROR: REGRESSION_DATA,
        Loss.BINARY_CROSS_ENTROPY: BINARY_DATA,
        Loss.SOFTMAX_CROSS_ENTROPY: splits.train,
    }
    configurations = []
    for name, layers, loss in CONFIGURATIONS:
        network = network_type(layers, loss, NETWORK_SEED)
        result = checks.check_network_gradient(network, data_for[loss])
        configurations.append(ConfigurationCheck(name, network.parameter_count, result))

    # Truncation falls as epsilon^2 for central differences and epsilon for one-sided ones, while
    # cancellation in the difference of two nearly equal losses grows as 1/epsilon.
    probe = network_type(PROBE_LAYERS, Loss.SOFTMAX_CROSS_ENTROPY, NETWORK_SEED)
    analytic = [float(v) for v in probe.gradient(splits.train)]
    step_sizes = tuple(
        StepSize(
            epsilon,
            checks.check_gradient(
                analytic, checks.numerical_gradient(probe, splits.train, epsilon)
            ).relative_error,
            checks.check_gradient(
                analytic, checks.forward_difference_gradient(probe, splits.train, epsilon)
            ).relative_error,
        )
        for epsilon in EPSILONS
    )
    # A checker that never fails proves nothing, so corrupt a known-good gradient.
    corrupted = list(analytic)
    corrupted[CORRUPTED_PARAMETER] *= 1.001
    detection = checks.check_gradient(
        corrupted, checks.numerical_gradient(probe, splits.train, 1e-5)
    )

    config = NetworkTrainingConfig(
        learning_rate=0.5, max_epochs=epochs, batch_size=32, seed=NETWORK_SEED
    )

    def report(network, trained):
        return TrainingReport(
            network.parameter_count,
            trained.loss_per_epoch[-1],
            trained.epochs,
            *(network.accuracy(split) for split in (splits.train, splits.validation, splits.test)),
        )

    linear = network_type(LINEAR_LAYERS, Loss.SOFTMAX_CROSS_ENTROPY, NETWORK_SEED)
    linear_report = report(linear, linear.fit(splits.train, config))

    def trajectory(name, network):
        gradient = [float(v) for v in network.gradient(splits.train)]
        result = checks.check_gradient(
            gradient, checks.numerical_gradient(network, splits.train, 1e-5)
        )
        return TrajectoryCheck(name, result, max(abs(v) for v in gradient))

    mlp = network_type(MLP_LAYERS, Loss.SOFTMAX_CROSS_ENTROPY, NETWORK_SEED)
    initial = trajectory("initialization", mlp)
    # A short run first, so the gradient is also checked away from both ends of the trajectory.
    mlp.fit(splits.train, replace(config, max_epochs=20))
    partial = trajectory("after 20 epochs", mlp)
    mlp_report = report(mlp, mlp.fit(splits.train, config))
    converged = trajectory("at convergence", mlp)

    result = ExperimentResult(
        tuple(configurations),
        step_sizes,
        detection,
        majority_baseline(splits),
        linear_report,
        mlp_report,
        (initial, partial, converged),
    )
    at_1e5 = step_sizes[EPSILONS.index(1e-5)]
    if not (
        all(c.result.passed for c in result.configurations)
        and at_1e5.central < at_1e5.forward
        and not detection.passed
        and mlp_report.test > linear_report.test > result.baseline
        and initial.result.passed
        and partial.result.passed
        # At convergence the norm ratio is expected to degrade; absolute agreement still counts.
        and converged.result.max_absolute_error < 1e-9
    ):
        raise RuntimeError("backpropagation experiment failed its criteria")
    return result


def run_experiment(seed=SEED, epochs=EPOCHS):
    return run_with(FeedForwardNetwork, gradient_check, seed, epochs)


def print_result(title, seed, epochs, result):
    print(f"{title} backpropagation and gradient checking experiment (seed={seed})")
    print("gradient check at initialization (central differences, epsilon=1e-5, tolerance=1e-7)")
    for check in result.configurations:
        verdict = "pass" if check.result.passed else "FAIL"
        print(
            f"  {check.name:30} parameters={check.parameters:4}, "
            f"gradient-check ratio={check.result.relative_error:.3e}  {verdict}"
        )
    print("step-size sweep on the softmax network\n  epsilon    central ratio   forward ratio")
    for step in result.step_sizes:
        print(f"  {step.epsilon:8.0e}{step.central:16.3e}{step.forward:16.3e}")
    detection = result.detection
    print(
        f"detection of a deliberate 0.1% error in parameter {CORRUPTED_PARAMETER}: "
        f"{'MISSED' if detection.passed else 'caught'}, ratio={detection.relative_error:.3e} "
        f"at parameter {detection.worst_parameter}"
    )
    print(f"training on the spiral ({epochs} epochs, batch=32, learning rate=0.5)")
    print(f"  majority baseline: test accuracy={result.baseline:.4f}")
    for name, report in (("linear softmax", result.linear), ("2x16 tanh MLP", result.mlp)):
        print(
            f"  {name} ({report.parameters} params): final loss={report.final_loss:.4f}, "
            f"train={report.train:.4f}, validation={report.validation:.4f}, "
            f"test={report.test:.4f}"
        )
    print("gradient check of the MLP on its trajectory")
    print("  point            norm ratio   max elementwise   max absolute   largest |gradient|")
    for point in result.trajectory:
        check = point.result
        print(
            f"  {point.name:17}{check.relative_error:11.3e}"
            f"{check.max_elementwise_relative_error:18.3e}{check.max_absolute_error:15.3e}"
            f"{point.largest_gradient:21.3e}  {'pass' if check.passed else 'FAIL'}"
        )


def main(run=run_experiment, title="Python"):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    args = parser.parse_args()
    print_result(title, args.seed, args.epochs, run(args.seed, args.epochs))


if __name__ == "__main__":
    main()
