"""Reproduce the AND/OR success and XOR failure using NumPy."""

import argparse

from ml_scratch.pure.datasets import logic_gate_dataset

from .model import Perceptron


def run_experiment(seed=7, max_epochs=100):
    results = []
    for gate in ("and", "or", "xor"):
        data = logic_gate_dataset(gate)
        model = Perceptron(2, seed=seed)
        result = model.fit(data, max_epochs=max_epochs)
        accuracy = model.accuracy(data)
        results.append((gate, result, accuracy))
        print(
            f"{gate.upper()}: epochs={result.epochs}, converged={result.converged}, "
            f"accuracy={accuracy:.4f}"
        )
    if any(not r.converged or a != 1 for _, r, a in results[:2]) or results[2][1].converged:
        raise RuntimeError("logic-gate experiment failed its criteria")
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    print(f"NumPy perceptron experiment (seed={args.seed})")
    run_experiment(args.seed)


if __name__ == "__main__":
    main()
