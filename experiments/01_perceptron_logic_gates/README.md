# Experiment 01: perceptron logic gates

This experiment trains the dependency-free perceptron on the complete AND and OR truth tables and
checks every prediction. It then uses the same training process on XOR to demonstrate the expected
non-convergence of a single linear decision boundary.

After installing the project, run `python run.py` from this directory or
`python experiments/01_perceptron_logic_gates/run.py` from the repository root. The canonical
implementation and CLI live in `ml_scratch.pure.perceptron_experiment` so experiment code does not
duplicate model logic.

The C++ version uses the same seed, learning rate, truth-table order, epoch budget, and stopping
rule. Build it from the repository root and run:

```bash
./build/cpp_perceptron_logic_gates
```

Its output reports AND and OR as learned only after checking all predictions. XOR always remains a
documented linear-separability failure rather than a successful training claim.
