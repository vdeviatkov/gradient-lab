# NumPy Binary logistic regression

[Topic and results](../README.md) · [Mathematics](../idea.md) ·
[NumPy conventions](../../../docs/implementations/numpy.md)

[Model](model.py) · [Public API](__init__.py) · [Experiment](experiment.py) ·
[Tests](tests/test_numpy_logistic_regression.py)

```bash
python -m pip install -e ".[dev,numpy]"
python -m ml_scratch.numpy.logistic_regression.experiment
python -m pytest topics/04_logistic_regression/numpy/tests
```

Scores use `X @ weights + bias`. The loss is mean weighted binary cross-entropy: positives
receive weight alpha, and normalization stays by sample count, not total class weight.
The explicit gradient uses `(sigmoid(z) - y) * where(y == 1, alpha, 1)`.
`logaddexp` and the target-specific logit sign avoid overflow and positive-loss cancellation.

The same configuration, mini-batch order, zero initialization on every `fit`, and stopping rules
as the scalar reference make training directly comparable. Single-sample prediction uses a
probability threshold, including equality in class 1. `predict_probabilities` handles a matrix.
Threshold selection maximizes validation F1, then prefers closest to 0.5, then the lower threshold.
Precision, recall, and F1 use zero when their denominator is zero.

```python
from ml_scratch.numpy.logistic_regression import BinaryLogisticRegression, LogisticTrainingConfig

model = BinaryLogisticRegression(1)
model.fit((((-1,), 0), ((1,), 1)), LogisticTrainingConfig(max_epochs=100))
assert model.predict([1]) == 1
```

The experiment reuses the scalar train/validation/test fixtures, chooses the threshold only on
validation, then compares test F1 with the default threshold and majority baseline.

Tests compare against the pure-Python reference and check independent numerical or hand-computed
results, input validation, deterministic behavior, and relevant failure/stopping cases.
