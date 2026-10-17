"""Run the reproducible pure-Python linear-regression experiment.

Four fitting methods against an exact reference, then the two things that decide whether gradient
descent works at all on a least-squares problem: the largest step it tolerates, and how many
steps it needs. Both are properties of the loss Hessian, and both are predicted here and then
measured.
"""

from __future__ import annotations

import argparse
import math
import random
from collections.abc import Sequence
from dataclasses import dataclass

from ml_scratch.pure.linear_regression import (
    LinearRegression,
    Optimizer,
    RegressionDataset,
    TrainingConfig,
    condition_number,
    hessian_eigenvalue_range,
    standardize,
)

# y = 1.5 + 2x exactly, the same relationship, splits and seed as the C++ milestone.
TRAIN: RegressionDataset = tuple(
    ((x,), 1.5 + 2.0 * x) for x in (-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0)
)
TEST: RegressionDataset = (((-2.5,), -3.5), ((0.25,), 2.0), ((3.0,), 7.5))
SEED = 20260903

# A two-feature problem whose second feature is multiplied by a growing scale, which is the
# cheapest way to make a well-posed problem badly conditioned without changing what it means.
_SCALING_BASE = (
    (-2.0, -1.0),
    (-1.5, 0.5),
    (-1.0, 1.0),
    (-0.5, -0.5),
    (0.0, 2.0),
    (0.5, 0.0),
    (1.0, -2.0),
    (1.5, 1.5),
    (2.0, 0.5),
)


def scaled_dataset(scale: float) -> RegressionDataset:
    """Return ``y = 1.5 + 2 a - 0.5 b`` with the second feature measured in units of ``scale``."""
    return tuple(
        ((first, second * scale), 1.5 + 2.0 * first - 0.5 * (second * scale))
        for first, second in _SCALING_BASE
    )


def collinear_dataset(separation: float, *, seed: int = 0, count: int = 12) -> RegressionDataset:
    """Return a dataset whose two features differ by ``separation``, with a little target noise."""
    generator = random.Random(seed)
    rows = []
    for _ in range(count):
        first = generator.uniform(-1.0, 1.0)
        second = first + separation * generator.uniform(-1.0, 1.0)
        target = 1.0 + 2.0 * first + 3.0 * second + 1e-3 * generator.uniform(-1.0, 1.0)
        rows.append(((first, second), target))
    return tuple(rows)


@dataclass(frozen=True)
class MethodSummary:
    """One fitting method's measured outcome."""

    name: str
    epochs: int | None
    train_mse: float
    test_mse: float
    weight: float
    bias: float


@dataclass(frozen=True)
class StabilitySummary:
    """The predicted and measured largest usable step for full-batch gradient descent."""

    largest_eigenvalue: float
    predicted_threshold: float
    measured_threshold: float

    @property
    def relative_error(self) -> float:
        return abs(self.measured_threshold - self.predicted_threshold) / self.predicted_threshold


@dataclass(frozen=True)
class ConditioningSummary:
    """How many epochs one scaling of the data costs, raw and standardized."""

    scale: float
    condition: float
    learning_rate: float
    epochs: int
    converged: bool
    standardized_condition: float
    standardized_epochs: int
    standardized_converged: bool


@dataclass(frozen=True)
class CollinearitySummary:
    """What a near-collinear design does to the coefficients, and to the fit."""

    separation: float
    singular: bool
    weights: tuple[float, ...]
    bias: float
    train_mse: float

    @property
    def weight_sum(self) -> float:
        return sum(self.weights)


def fit_methods(
    train: RegressionDataset = TRAIN, test: RegressionDataset = TEST, *, seed: int = SEED
) -> tuple[MethodSummary, ...]:
    """Fit the closed form and three gradient methods, and measure each the same way."""
    reference = LinearRegression(1)
    reference.fit_closed_form(train)
    summaries = [
        MethodSummary(
            "closed form",
            None,
            reference.mean_squared_error(train),
            reference.mean_squared_error(test),
            reference.weights[0],
            reference.bias,
        )
    ]

    settings = (
        ("batch GD", Optimizer.BATCH, 0.10, 9),
        ("SGD", Optimizer.STOCHASTIC, 0.03, 1),
        ("mini-batch GD", Optimizer.MINI_BATCH, 0.05, 3),
    )
    for name, optimizer, learning_rate, batch_size in settings:
        model = LinearRegression(1)
        result = model.fit(
            train,
            TrainingConfig(
                optimizer=optimizer,
                learning_rate=learning_rate,
                max_epochs=2_000,
                batch_size=batch_size,
                seed=seed,
                target_mse=1e-12,
            ),
        )
        summaries.append(
            MethodSummary(
                name,
                result.epochs,
                model.mean_squared_error(train),
                model.mean_squared_error(test),
                model.weights[0],
                model.bias,
            )
        )
    return tuple(summaries)


# The C++ milestone deliberately minimizes half the mean squared error, a convention its
# mathematics documents: halving does not move the optimum and it cancels the factor of two that
# differentiation produces, so its update accumulates `residual` where this one accumulates
# `2 * residual`. Both are self-consistent; what they are not is interchangeable, because the same
# nominal learning rate then takes a step twice as large here as there.
CPP_EPOCHS = {"batch GD": 135, "SGD": 51, "mini-batch GD": 92}


def match_cpp_convention(*, seed: int = SEED) -> tuple[tuple[str, int, int], ...]:
    """Rerun each gradient method at half its learning rate and return ``(name, ours, theirs)``.

    Halving the step is exactly what the difference in convention amounts to, so if that is the
    whole difference these counts should reproduce the C++ run's epochs.
    """
    settings = (
        ("batch GD", Optimizer.BATCH, 0.10, 9),
        ("SGD", Optimizer.STOCHASTIC, 0.03, 1),
        ("mini-batch GD", Optimizer.MINI_BATCH, 0.05, 3),
    )
    results = []
    for name, optimizer, learning_rate, batch_size in settings:
        model = LinearRegression(1)
        result = model.fit(
            TRAIN,
            TrainingConfig(
                optimizer=optimizer,
                learning_rate=0.5 * learning_rate,
                max_epochs=5_000,
                batch_size=batch_size,
                seed=seed,
                target_mse=1e-12,
            ),
        )
        results.append((name, result.epochs, CPP_EPOCHS[name]))
    return tuple(results)


def measure_stability(
    dataset: RegressionDataset = TRAIN, *, max_epochs: int = 2_000, steps: int = 60
) -> StabilitySummary:
    """Bisect for the largest learning rate at which full-batch descent still makes progress.

    Theory says a quadratic's gradient descent converges exactly when the step is below
    ``2 / lambda_max``; above it, every step multiplies the error along the steepest direction by
    more than one. "Still makes progress" is the operational test used here: a run is counted as
    unstable when it ends with a larger error than it started with.
    """
    _, largest = hessian_eigenvalue_range(dataset)
    predicted = 2.0 / largest

    def unstable(learning_rate: float) -> bool:
        model = LinearRegression(len(dataset[0][0]))
        result = model.fit(
            dataset,
            TrainingConfig(learning_rate=learning_rate, max_epochs=max_epochs, target_mse=0.0),
        )
        if result.diverged or not math.isfinite(result.mse_per_epoch[-1]):
            return True
        return result.mse_per_epoch[-1] > result.mse_per_epoch[0]

    low, high = 1e-6, 4.0 * predicted
    for _ in range(steps):
        middle = 0.5 * (low + high)
        if unstable(middle):
            high = middle
        else:
            low = middle
    return StabilitySummary(largest, predicted, low)


def measure_conditioning(
    scale: float, *, target_mse: float = 1e-10, max_epochs: int = 200_000
) -> ConditioningSummary:
    """Train at 90% of each dataset's own stability limit, raw and standardized."""

    def train_at_limit(dataset: RegressionDataset) -> tuple[float, int, bool]:
        _, largest = hessian_eigenvalue_range(dataset)
        learning_rate = 0.9 * 2.0 / largest
        model = LinearRegression(len(dataset[0][0]))
        result = model.fit(
            dataset,
            TrainingConfig(
                learning_rate=learning_rate, max_epochs=max_epochs, target_mse=target_mse
            ),
        )
        return learning_rate, result.epochs, result.converged

    dataset = scaled_dataset(scale)
    standardized = standardize(dataset)
    learning_rate, epochs, converged = train_at_limit(dataset)
    _, standardized_epochs, standardized_converged = train_at_limit(standardized)
    return ConditioningSummary(
        scale,
        condition_number(dataset),
        learning_rate,
        epochs,
        converged,
        condition_number(standardized),
        standardized_epochs,
        standardized_converged,
    )


def measure_collinearity(separation: float, *, seed: int = 0) -> CollinearitySummary:
    """Fit a near-collinear design in closed form and report what came out."""
    dataset = collinear_dataset(separation, seed=seed)
    model = LinearRegression(2)
    try:
        model.fit_closed_form(dataset)
    except ValueError:
        return CollinearitySummary(separation, True, (), 0.0, math.nan)
    return CollinearitySummary(
        separation,
        False,
        tuple(model.weights),
        model.bias,
        model.mean_squared_error(dataset),
    )


def run_experiment(
    *,
    seed: int = SEED,
    scales: Sequence[float] = (1.0, 10.0, 100.0),
    separations: Sequence[float] = (1e-2, 1e-4, 1e-6, 1e-8),
) -> tuple[
    tuple[MethodSummary, ...],
    StabilitySummary,
    tuple[ConditioningSummary, ...],
    tuple[CollinearitySummary, ...],
]:
    """Run every part and return the measured summaries."""
    methods = fit_methods(seed=seed)
    stability = measure_stability()
    conditioning = tuple(measure_conditioning(scale) for scale in scales)
    collinearity = tuple(measure_collinearity(separation) for separation in separations)
    return methods, stability, conditioning, collinearity


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command-line experiment."""
    parser = argparse.ArgumentParser(description="pure-Python linear regression experiment")
    parser.add_argument("--seed", type=int, default=SEED, help="seed for the stochastic methods")
    arguments = parser.parse_args(argv)

    print(f"Pure-Python linear regression experiment (seed={arguments.seed})")

    print("\npart one: four ways to fit y = 1.5 + 2x, measured the same way")
    print("  method           epochs        weight          bias     train MSE      test MSE")
    methods = fit_methods(seed=arguments.seed)
    for summary in methods:
        epochs = "-" if summary.epochs is None else str(summary.epochs)
        print(
            f"  {summary.name:<15}{epochs:>7}{summary.weight:>14.9f}{summary.bias:>14.9f}"
            f"{summary.train_mse:>14.3e}{summary.test_mse:>14.3e}"
        )
    reference = methods[0]
    largest_gap = max(
        max(abs(summary.weight - reference.weight), abs(summary.bias - reference.bias))
        for summary in methods[1:]
    )
    print(f"  the gradient methods land within {largest_gap:.2e} of the exact coefficients")
    print(
        "\n  the C++ milestone descends on half the mean squared error, so its learning rate means"
        " something else;\n  halving ours should reproduce its epoch counts exactly:"
    )
    matched = match_cpp_convention(seed=arguments.seed)
    for name, ours, theirs in matched:
        mark = "matches" if ours == theirs else "DIFFERS"
        print(f"    {name:<15}{ours:>6} against the C++ run's {theirs:<6} {mark}")

    print("\npart two: the largest step full-batch gradient descent tolerates")
    stability = measure_stability()
    print(
        f"  the loss Hessian's largest eigenvalue is {stability.largest_eigenvalue:.9f}, so theory"
        f" predicts a threshold of 2/lambda = {stability.predicted_threshold:.9f}"
    )
    print(
        f"  bisecting for the largest rate that still reduces the error measures "
        f"{stability.measured_threshold:.9f}, off by {stability.relative_error:.1e}"
    )

    print("\npart three: what conditioning costs, at 90% of each problem's own stability limit")
    print("  scale   condition   learning rate   epochs   standardized condition   epochs")
    conditioning = tuple(measure_conditioning(scale) for scale in (1.0, 10.0, 100.0))
    for summary in conditioning:
        print(
            f"  {summary.scale:>5.0f}{summary.condition:>12.1f}{summary.learning_rate:>16.2e}"
            f"{summary.epochs:>9}{summary.standardized_condition:>25.1f}"
            f"{summary.standardized_epochs:>9}"
        )

    print("\npart four: two features that are nearly the same, fitted in closed form")
    print("  separation   weights                    sum of weights     train MSE")
    collinearity = tuple(measure_collinearity(value) for value in (1e-2, 1e-4, 1e-6, 1e-8))
    for summary in collinearity:
        if summary.singular:
            print(f"  {summary.separation:>10.0e}    refused: singular to working precision")
            continue
        weights = ", ".join(f"{weight:+.4f}" for weight in summary.weights)
        print(
            f"  {summary.separation:>10.0e}   {'(' + weights + ')':<26}"
            f"{summary.weight_sum:>14.4f}{summary.train_mse:>14.3e}"
        )

    if largest_gap > 1e-5:
        raise RuntimeError("a gradient method did not reach the closed-form coefficients")
    if any(ours != theirs for _, ours, theirs in matched):
        raise RuntimeError(
            "halving the learning rate no longer reproduces the C++ epoch counts, so the "
            "factor of two is not the whole difference between the two implementations"
        )
    if stability.relative_error > 1e-4:
        raise RuntimeError("the measured stability threshold does not match 2 / lambda_max")
    if not all(summary.converged for summary in conditioning):
        raise RuntimeError("a conditioning run failed to reach its target")
    if conditioning[-1].epochs <= 100 * conditioning[0].epochs:
        raise RuntimeError("worse conditioning should cost far more epochs")
    if conditioning[-1].standardized_epochs >= conditioning[-1].epochs // 100:
        raise RuntimeError("standardizing should remove most of that cost")
    if not collinearity[-1].singular:
        raise RuntimeError("an exactly collinear design should be refused, not answered")
    fitted = [summary for summary in collinearity if not summary.singular]
    if max(abs(summary.weights[1]) for summary in fitted) < 10.0:
        raise RuntimeError("near-collinearity should blow the individual coefficients up")
    if max(abs(summary.weight_sum - 5.0) for summary in fitted) > 0.01:
        raise RuntimeError("the sum of the coefficients should stay determined")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
