# Perceptron and logic gates

[All topics](../../README.md#topics) · [Idea and mathematics](idea.md) · [Setup](../../docs/getting-started.md)

**Implementations:** [C++](cpp/README.md) · [Python](python/README.md) · [NumPy](numpy/README.md) · [PyTorch](pytorch/README.md) · [Backend status](../../README.md#topics)

This experiment trains the dependency-free perceptron on the complete AND and OR truth tables and
checks every prediction. It then uses the same training process on XOR to demonstrate the expected
non-convergence of a single linear decision boundary.

After [setup](../../docs/getting-started.md), run
`python -m ml_scratch.pure.perceptron.experiment`. The model and experiment are owned by the
[Python implementation](python/README.md).

The C++ version uses the same seed, learning rate, truth-table order, epoch budget, and stopping
rule. Build it from the repository root and run:

```bash
./build/cpp_perceptron_logic_gates
```

Its output reports AND and OR as learned only after checking all predictions. XOR always remains a
documented linear-separability failure rather than a successful training claim.

## NumPy reproduction

Measured on 2026-10-20 with CPython 3.14.7 and NumPy 2.5.4 on macOS 26.3.1 (arm64).
Command: `python -m ml_scratch.numpy.perceptron.experiment` after installing `.[numpy]`.
Reported quantities are measured losses, predictions, and iteration counts from the current run;
no runtime comparison is claimed. See the [NumPy implementation](numpy/README.md) for conventions.

At seed `7`, learning rate `0.1`, shuffling enabled, and a 100-epoch budget, AND converged
in 8 epochs and OR in 9, each with accuracy `1.0000`. XOR exhausted all 100 epochs without
converging and ended at accuracy `0.5000`. The complete four-row truth tables serve as both
training data and exhaustive evaluation; these are not held-out generalization measurements.

## PyTorch reproduction

Measured on 2026-10-21 using CPython 3.14.7 and PyTorch 2.14.1 on macOS 26.3.1 (arm64),
CPU float64. Command: `python -m ml_scratch.pytorch.perceptron.experiment` after installing
`.[pytorch]`. See the [PyTorch implementation](pytorch/README.md) for contracts and tests.

Seed `7`, rate `0.1`, shuffled online updates, and a 100-epoch budget produced the same outcomes
as NumPy: AND converged in 8 epochs, OR in 9, both with accuracy `1.0000`. XOR exhausted the
budget without converging and ended at `0.5000` accuracy. Evaluation uses the complete truth
tables, not a held-out generalization split.
