"""XOR learning, width sensitivity, and an independent central-difference check."""

import argparse

import numpy as np
from ml_scratch.pure.datasets import logic_gate_dataset

from .model import XorMlp


def gradient_check(model, data):
    analytic, numerical = model.gradient(data), model.numerical_gradient(data)
    return max(
        float(
            np.max(
                np.abs(np.asarray(getattr(analytic, name)) - np.asarray(getattr(numerical, name)))
            )
        )
        for name in ("input_weights", "hidden_biases", "output_weights", "output_bias")
    )


def run_experiment(seed=7, seeds=10):
    if seeds <= 0:
        raise ValueError("seeds must be positive")
    data = logic_gate_dataset("xor")
    model = XorMlp(seed=seed)
    error = gradient_check(model, data)
    result = model.fit(data)
    print(
        f"XOR: epochs={result.epochs}, converged={result.converged}, "
        f"BCE={result.loss_per_epoch[-1]:.6f}, accuracy={model.accuracy(data):.4f}"
    )
    print(f"maximum gradient error={error:.3e}")
    widths = []
    for width in (1, 2, 3, 4, 8):
        solved, epochs = 0, []
        for trial in range(seeds):
            network = XorMlp(hidden_units=width, seed=trial)
            trained = network.fit(data)
            if trained.converged and network.accuracy(data) == 1:
                solved += 1
                epochs.append(trained.epochs)
        widths.append((width, solved))
        median = float(np.median(epochs)) if epochs else None
        print(f"width={width}: solved={solved}/{seeds}, median epochs={median}")
    if not result.converged or model.accuracy(data) != 1 or error > 1e-7 or widths[0][1] != 0:
        raise RuntimeError("XOR experiment failed its criteria")
    return result, widths, error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--seeds", type=int, default=10)
    args = parser.parse_args()
    print(f"NumPy XOR MLP experiment (seed={args.seed}; width sweep seeds=0..{args.seeds - 1})")
    run_experiment(args.seed, args.seeds)


if __name__ == "__main__":
    main()
