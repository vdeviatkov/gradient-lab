# Experiment 04: binary logistic regression

[All topics](../README.md) · [C++ implementation](../../src/cpp/README.md) · [Pure Python](../../src/python/ml_scratch/pure/README.md) · [Mathematics](../../docs/mathematics/logistic_regression.md)

**Sources:** [C++ experiment](cpp_main.cpp) · [logistic_regression](../../src/cpp/src/logistic_regression.cpp) · [Python model](../../src/python/ml_scratch/pure/logistic_regression.py) · [Python runner](run.py)

The pure-Python and C++ experiments use a small, inspectable one-feature dataset with overlapping classes and a
3:1 negative-to-positive training imbalance. Training, validation, and test examples are fixed in
the source. Mini-batch order uses seed `20260904`.

The model minimizes binary cross-entropy on the training split until it reaches the configured
loss target or epoch limit. It selects a probability threshold from a fixed candidate grid by
maximizing validation F1, freezes that threshold, and evaluates it once on the test split. The
output also includes the conventional `0.5` threshold and an always-negative majority baseline to
show why accuracy alone can mislead on imbalanced data.

Build from the repository root and run:

```bash
cmake -S . -B build
cmake --build build
./build/cpp_logistic_regression
```

The executable prints only values measured during the current run. It exits unsuccessfully unless
training reaches its loss target and the selected classifier improves test F1 over both the
default threshold and the majority baseline.


Run the dependency-free Python version after installing the project:

```bash
python -m ml_scratch.pure.logistic_regression_experiment
# Equivalent entry points:
ml-logistic-regression
python experiments/04_logistic_regression/run.py
```

The Python CLI accepts `--seed` and prints it with every run. Both implementations use the same
fixed splits, candidate grid, zero initialization, learning rate `0.1`, batch size `4`, epoch
budget `2000`, and target BCE `0.30`. Their shuffle implementations differ, so equal seeds do not
imply identical training trajectories. Neither validation nor test samples enter training; only
validation labels select the threshold. These small hand-built splits illustrate the procedure,
not statistical evidence of generalization on a population.

Measured pure-Python run with seed `20260904`: training stops after 246 epochs at BCE `0.299995`,
weight `2.210806`, and bias `-2.255443`. Validation selects threshold `0.20` with F1 `0.666667`.

| Test classifier | Accuracy | Precision | Recall | F1 | TN / FP / FN / TP |
|---|---:|---:|---:|---:|---|
| Always negative | 0.80 | 0.00 | 0.00 | 0.00 | 8 / 0 / 2 / 0 |
| Threshold 0.50 | 0.70 | 0.333333 | 0.50 | 0.40 | 6 / 2 / 1 / 1 |
| Validation-selected 0.20 | 0.60 | 0.333333 | 1.00 | 0.50 | 4 / 4 / 0 / 2 |

The selected threshold detects both positives at the cost of two additional false positives.
Its F1 improves while accuracy decreases: the choice of metric determines which tradeoff wins.
Tests cover weighted gradients against central differences, extreme logits, hand-counted metrics,
threshold ties, partial batches, reset and stopping behavior, reproducibility, invalid inputs,
and the experiment's held-out improvement criteria.
