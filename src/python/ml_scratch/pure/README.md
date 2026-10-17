# Pure-Python implementation

[Python implementations](../../README.md) · [Setup](../../../../docs/getting-started.md) ·
[Tests](../../../../tests/python/README.md) · [Status](../../../../docs/status.md)

Dependency-free implementations with explicit operations and hand-derived gradients.

| Topic | Model | Experiment module | Topic README |
|---|---|---|---|
| Perceptron | [perceptron.py](perceptron.py) | `perceptron_experiment` | [01](../../../../experiments/01_perceptron_logic_gates/README.md) |
| XOR MLP | [mlp.py](mlp.py) | `xor_mlp_experiment` | [02](../../../../experiments/02_xor_mlp/README.md) |
| Linear regression | [linear_regression.py](linear_regression.py) | `linear_regression_experiment` | [03](../../../../experiments/03_linear_regression/README.md) |
| Binary logistic regression | [logistic_regression.py](logistic_regression.py) | `logistic_regression_experiment` | [04](../../../../experiments/04_logistic_regression/README.md) |

## Run

After an editable installation from the repository root:

```bash
python -m ml_scratch.pure.perceptron_experiment
python -m ml_scratch.pure.xor_mlp_experiment
python -m ml_scratch.pure.linear_regression_experiment
python -m ml_scratch.pure.logistic_regression_experiment
```

Equivalent installed commands are `ml-perceptron-logic-gates`, `ml-xor-mlp`,
`ml-linear-regression`, and `ml-logistic-regression`. Each command exposes `--help`.
All four use small local fixtures, so no dataset download is required.

## Conventions

Use local seeded random generators and record the seed with results. Keep the loss and gradient
conventions explicit, especially when comparing learning rates with C++. Validate gradients against
independent central differences and report non-convergence honestly. Shared immutable logic-gate
fixtures live in [datasets.py](datasets.py).
