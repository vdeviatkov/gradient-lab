# Python tests

[All tests](../README.md) · [Pure-Python implementation](../../src/python/ml_scratch/pure/README.md)

```bash
python -m pytest tests/python
```

| Topic | Model tests | Experiment tests |
|---|---|---|
| Perceptron and datasets | [Perceptron](test_perceptron.py), [datasets](test_datasets.py) | [Runner](test_perceptron_experiment.py) |
| XOR MLP | [Model](test_xor_mlp.py) | [Runner](test_xor_mlp_experiment.py) |
| Linear regression | [Model](test_linear_regression.py) | [Runner](test_linear_regression_experiment.py) |
| Binary logistic regression | [Model](test_logistic_regression.py) | [Runner](test_logistic_regression_experiment.py) |

Coverage includes hand-computed predictions and metrics, gradients against central differences,
training and non-convergence, stability and conditioning, malformed inputs, seeded reproducibility,
and each experiment's verification criteria. Pytest's source path configuration is maintained in
[pyproject.toml](../../pyproject.toml).
