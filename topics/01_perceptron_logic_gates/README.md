# Perceptron and logic gates

[All topics](../../README.md#topics) · [Idea and mathematics](idea.md) · [Setup](../../docs/getting-started.md)

**Implementations:** [C++](cpp/README.md) · [Python](python/README.md) · [Backend status](../../README.md#topics)

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
