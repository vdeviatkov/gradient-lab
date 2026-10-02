"""Run the reproducible pure-Python XOR experiment: one hidden layer, and how wide it must be."""

from __future__ import annotations

import argparse
import math
from collections.abc import Sequence
from dataclasses import dataclass
from statistics import median

from ml_scratch.pure.datasets import logic_gate_dataset
from ml_scratch.pure.mlp import MlpParameters, XorMlp

CHANCE_LOSS = math.log(2.0)


@dataclass(frozen=True)
class RunSummary:
    """Record one training run without hiding its outcome."""

    hidden_units: int
    seed: int
    converged: bool
    epochs: int
    final_loss: float
    accuracy: float


@dataclass(frozen=True)
class WidthSummary:
    """Aggregate every seed tried at one hidden-layer width."""

    hidden_units: int
    attempts: int
    solved: int
    median_epochs: int | None
    failure_accuracies: tuple[float, ...]
    failure_losses: tuple[float, ...]

    @property
    def solved_fraction(self) -> float:
        return self.solved / self.attempts


def train_once(
    hidden_units: int, *, seed: int, learning_rate: float, max_epochs: int, target_loss: float
) -> tuple[XorMlp, RunSummary]:
    """Train one network on the complete XOR truth table and report what happened."""
    dataset = logic_gate_dataset("xor")
    model = XorMlp(hidden_units=hidden_units, learning_rate=learning_rate, seed=seed)
    result = model.fit(dataset, max_epochs=max_epochs, target_loss=target_loss)
    summary = RunSummary(
        hidden_units,
        seed,
        result.converged and model.accuracy(dataset) == 1.0,
        result.epochs,
        result.loss_per_epoch[-1],
        model.accuracy(dataset),
    )
    return model, summary


def sweep_width(
    hidden_units: int, *, seeds: int, learning_rate: float, max_epochs: int, target_loss: float
) -> WidthSummary:
    """Train one width from many seeds, so initialization sensitivity is measured not assumed."""
    runs = [
        train_once(
            hidden_units,
            seed=seed,
            learning_rate=learning_rate,
            max_epochs=max_epochs,
            target_loss=target_loss,
        )[1]
        for seed in range(seeds)
    ]
    solved = [run for run in runs if run.converged]
    failures = [run for run in runs if not run.converged]
    return WidthSummary(
        hidden_units,
        len(runs),
        len(solved),
        int(median(run.epochs for run in solved)) if solved else None,
        tuple(sorted({run.accuracy for run in failures})),
        tuple(sorted({round(run.final_loss, 4) for run in failures})),
    )


def gradient_check(hidden_units: int = 4, *, seed: int = 7, epsilon: float = 1e-5) -> float:
    """Return the largest disagreement between the hand-derived and numerical gradients.

    The gradients in :mod:`ml_scratch.pure.mlp` are written out from the chain rule, so something
    has to check them against a method that knows nothing about the derivation. Central
    differences is that method, and the number it produces here is the evidence.
    """
    dataset = logic_gate_dataset("xor")
    model = XorMlp(hidden_units=hidden_units, seed=seed)
    analytic = model.gradient(dataset)
    numerical = model.numerical_gradient(dataset, epsilon=epsilon)

    def flatten(parameters: MlpParameters) -> list[float]:
        values = [value for row in parameters.input_weights for value in row]
        values.extend(parameters.hidden_biases)
        values.extend(parameters.output_weights)
        values.append(parameters.output_bias)
        return values

    return max(
        abs(left - right)
        for left, right in zip(flatten(analytic), flatten(numerical), strict=True)
    )


def run_experiment(
    *,
    seed: int = 7,
    hidden_units: int = 4,
    learning_rate: float = 0.5,
    max_epochs: int = 10_000,
    target_loss: float = 0.02,
    seeds: int = 50,
    widths: Sequence[int] = (1, 2, 3, 4, 8),
) -> tuple[RunSummary, tuple[WidthSummary, ...], float]:
    """Run every part and return the measured summaries."""
    _, headline = train_once(
        hidden_units,
        seed=seed,
        learning_rate=learning_rate,
        max_epochs=max_epochs,
        target_loss=target_loss,
    )
    if not headline.converged:
        raise RuntimeError(
            f"the {hidden_units}-unit network failed to learn XOR at seed {seed}, "
            "which the experiment is built to demonstrate"
        )
    sweeps = tuple(
        sweep_width(
            width,
            seeds=seeds,
            learning_rate=learning_rate,
            max_epochs=max_epochs,
            target_loss=target_loss,
        )
        for width in widths
    )
    return headline, sweeps, gradient_check(hidden_units, seed=seed)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command-line experiment."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=7, help="seed for the headline network")
    parser.add_argument("--hidden-units", type=int, default=4, help="width of the hidden layer")
    parser.add_argument("--learning-rate", type=float, default=0.5, help="gradient-descent step")
    parser.add_argument("--max-epochs", type=int, default=10_000, help="training epoch budget")
    parser.add_argument("--target-loss", type=float, default=0.02, help="stopping loss")
    parser.add_argument("--seeds", type=int, default=50, help="seeds per width in the sweep")
    arguments = parser.parse_args(argv)

    dataset = logic_gate_dataset("xor")
    model, headline = train_once(
        arguments.hidden_units,
        seed=arguments.seed,
        learning_rate=arguments.learning_rate,
        max_epochs=arguments.max_epochs,
        target_loss=arguments.target_loss,
    )

    print(f"Pure-Python two-layer XOR MLP experiment (seed={arguments.seed})")
    print(
        f"\npart one: a 2 -> {arguments.hidden_units} -> 1 network on the same truth table the "
        "perceptron could not separate"
    )
    if headline.converged:
        print(
            f"  XOR: learned in {headline.epochs} epochs, "
            f"binary cross-entropy={headline.final_loss:.6f}"
        )
    else:
        print(f"  XOR: did NOT converge in {headline.epochs} epochs")
    for features, target in dataset:
        probability = model.predict_probability(features)
        print(
            f"  ({features[0]:.0f}, {features[1]:.0f}) -> {model.predict(features)} "
            f"(p={probability:.4f}, target={target})"
        )

    print(
        f"\npart two: how wide the hidden layer has to be, over {arguments.seeds} seeds per width"
    )
    print("  hidden units   solved   median epochs   failures end at")
    sweeps = tuple(
        sweep_width(
            width,
            seeds=arguments.seeds,
            learning_rate=arguments.learning_rate,
            max_epochs=arguments.max_epochs,
            target_loss=arguments.target_loss,
        )
        for width in (1, 2, 3, 4, 8)
    )
    for sweep in sweeps:
        epochs = "-" if sweep.median_epochs is None else str(sweep.median_epochs)
        if sweep.failure_accuracies:
            accuracies = ", ".join(f"{value:.2f}" for value in sweep.failure_accuracies)
            losses = ", ".join(f"{value:.4f}" for value in sweep.failure_losses)
            ending = f"accuracy {accuracies}; loss {losses}"
        else:
            ending = "-"
        print(
            f"  {sweep.hidden_units:>12}   {sweep.solved:>2}/{sweep.attempts:<4}"
            f"{epochs:>14}   {ending}"
        )

    largest = gradient_check(arguments.hidden_units, seed=arguments.seed)
    print(
        f"\n  the hand-derived gradient and central differences agree to {largest:.2e}, "
        "which is what makes the derivation checkable rather than merely plausible"
    )
    print(f"  chance on this dataset is a loss of ln 2 = {CHANCE_LOSS:.4f}")

    one_unit = next(sweep for sweep in sweeps if sweep.hidden_units == 1)
    two_units = next(sweep for sweep in sweeps if sweep.hidden_units == 2)
    four_units = next(sweep for sweep in sweeps if sweep.hidden_units == 4)
    if one_unit.solved != 0:
        raise RuntimeError("a single hidden unit cannot represent XOR, yet one run claimed to")
    if not 0.0 < two_units.solved_fraction < 1.0:
        raise RuntimeError("two hidden units should sometimes succeed and sometimes fail")
    if four_units.solved != four_units.attempts:
        raise RuntimeError("four hidden units should solve XOR from every seed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
