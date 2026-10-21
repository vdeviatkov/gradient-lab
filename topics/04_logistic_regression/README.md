# Binary logistic regression

[All topics](../../README.md#topics) · [Idea and mathematics](idea.md) · [Setup](../../docs/getting-started.md)

**Implementations:** [C++](cpp/README.md) · [Python](python/README.md) · [NumPy](numpy/README.md) · [PyTorch](pytorch/README.md) · [Backend status](../../README.md#topics)

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
python -m ml_scratch.pure.logistic_regression.experiment
# Equivalent entry points:
ml-logistic-regression
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

## NumPy reproduction

Measured on 2026-10-20 with CPython 3.14.7 and NumPy 2.5.4 on macOS 26.3.1 (arm64).
Command: `python -m ml_scratch.numpy.logistic_regression.experiment` after installing `.[numpy]`.
Reported quantities are measured losses, predictions, and iteration counts from the current run;
no runtime comparison is claimed. See the [NumPy implementation](numpy/README.md) for conventions.

Seed `20260904`, learning rate `0.1`, mini-batch size 4, positive class weight 1, and target
BCE `0.30` reproduced the Python experiment's fixed train/validation/test splits and threshold grid.
Training stopped at 246 epochs, BCE `0.299995`, weight `2.210806`, bias `-2.255443`.
Validation F1 selected threshold `0.20` with F1 `0.666667`.

| Test prediction rule | Accuracy | Recall | F1 |
|---|---:|---:|---:|
| Majority baseline | 0.800000 | 0.000000 | 0.000000 |
| Threshold 0.50 | 0.700000 | 0.500000 | 0.400000 |
| Validation-selected threshold 0.20 | 0.600000 | 1.000000 | 0.500000 |

The selected threshold improves the optimized metric, F1, while reducing accuracy on this small
held-out test split. The test set was not used to select it.

## PyTorch reproduction

Measured on 2026-10-21 using CPython 3.14.7 and PyTorch 2.14.1 on macOS 26.3.1 (arm64),
CPU float64. Command: `python -m ml_scratch.pytorch.logistic_regression.experiment` after installing
`.[pytorch]`. See the [PyTorch implementation](pytorch/README.md) for contracts and tests.

Seed `20260904`, rate `0.1`, mini-batch size 4, positive class weight 1, and target BCE `0.30`
reproduced the scalar and NumPy fixed-split experiment. Training stopped in 246 epochs with
BCE `0.299995`, weight `2.210806`, and bias `-2.255443`. Validation F1 selected threshold `0.20`
with F1 `0.666667`. Test F1 was `0.000000` for the majority baseline, `0.400000` at threshold
0.50, and `0.500000` at the selected threshold. Test recall rose from `0.5` to `1.0` while
accuracy fell from `0.7` to `0.6`; selection optimizes F1, not accuracy. Test data was not used
for selection. Weighted gradient tests compare the explicit derivatives with autograd and central
differences, including positive weight `3.7`.
