"""Run the reproducible perceptron experiment on AND, OR, and XOR."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass

from ml_scratch.pure.datasets import LogicGate, logic_gate_dataset
from ml_scratch.pure.perceptron import Perceptron


@dataclass(frozen=True)
class GateSummary:
    """Record measured outcomes from one logic-gate training run."""

    gate: LogicGate
    converged: bool
    epochs: int
    predictions: tuple[int, ...]
    targets: tuple[int, ...]


def run_gate(gate: LogicGate, *, seed: int, max_epochs: int) -> GateSummary:
    """Train and evaluate one gate using a deterministic local seed."""
    dataset = logic_gate_dataset(gate)
    model = Perceptron(feature_count=2, learning_rate=0.1, seed=seed)
    training = model.fit(dataset, max_epochs=max_epochs)
    predictions = model.predict_many(features for features, _ in dataset)
    targets = tuple(target for _, target in dataset)
    return GateSummary(gate, training.converged, training.epochs, predictions, targets)


def run_experiment(*, seed: int = 7, max_epochs: int = 100) -> tuple[GateSummary, ...]:
    """Run all current logic-gate cases and return their measured summaries."""
    summaries = tuple(
        run_gate(gate, seed=seed, max_epochs=max_epochs) for gate in ("and", "or", "xor")
    )
    for summary in summaries[:2]:
        if not summary.converged or summary.predictions != summary.targets:
            raise RuntimeError(f"{summary.gate.upper()} unexpectedly failed to converge")
    return summaries


def _format_summary(summary: GateSummary) -> str:
    gate = summary.gate.upper()
    pairs = ", ".join(
        f"{prediction}/{target}"
        for prediction, target in zip(summary.predictions, summary.targets, strict=True)
    )
    if summary.gate == "xor":
        return (
            f"{gate}: not linearly separable; single-layer perceptron "
            f"did not converge in {summary.epochs} epochs\n"
            f"  predictions/targets: {pairs}"
        )
    return f"{gate}: learned in {summary.epochs} epochs\n  predictions/targets: {pairs}"


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command-line experiment."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=7, help="deterministic initialization seed")
    parser.add_argument("--max-epochs", type=int, default=100, help="training epoch budget")
    arguments = parser.parse_args(argv)

    print(f"Pure-Python perceptron logic-gate experiment (seed={arguments.seed})")
    for summary in run_experiment(seed=arguments.seed, max_epochs=arguments.max_epochs):
        print(_format_summary(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
