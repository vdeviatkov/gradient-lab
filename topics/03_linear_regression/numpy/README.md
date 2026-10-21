# NumPy Linear regression

[Topic and results](../README.md) · [Mathematics](../idea.md) ·
[NumPy conventions](../../../docs/implementations/numpy.md)

[Model](model.py) · [Public API](__init__.py) · [Experiment](experiment.py) ·
[Tests](tests/test_numpy_linear_regression.py)

```bash
python -m pip install -e ".[dev,numpy]"
python -m ml_scratch.numpy.linear_regression.experiment
python -m pytest topics/03_linear_regression/numpy/tests
```

The loss is full mean squared error, so the explicit gradient is
`2 * X.T @ residual / n` and the bias gradient is `2 * mean(residual)`. This matches Python;
the C++ reference uses half-MSE, so its learning rate must be halved for equivalent updates here.
Batch, stochastic, and mini-batch training use the same configuration objects and local Python
shuffle stream as the scalar model. `fit` continues from current parameters and reports divergence.

The closed form constructs the normal equations with matrix products, then uses explicit
Gaussian elimination with partial pivoting and a `1e-12` singular-pivot threshold. It does not call
`numpy.linalg.solve`, `lstsq`, or an inverse. Normal equations square the design condition number;
this is an educational reference, not a numerically robust least-squares solver for arbitrary data.
Hessian diagnostics keep the scalar power-iteration and shifted-matrix methods explicit.
As in that reference, the all-ones initial vector can miss an orthogonal eigenspace; the resulting
spectral estimates are diagnostics, not certified bounds for every matrix.

```python
from ml_scratch.numpy.linear_regression import LinearRegression

model = LinearRegression(1)
model.fit_closed_form((((0,), 1), ((1,), 3), ((2,), 5)))
assert abs(model.predict([3]) - 7) < 1e-12
```

`standardize` uses population standard deviation and maps constant columns to zero. It returns
the pair-dataset representation for compatibility. Fit preprocessing on training data only when
extending this helper to a held-out prediction pipeline. R² refuses constant-target datasets.
The experiment measures optimizer agreement, stability around `2/lambda_max`, feature scaling,
and near-collinearity. It reports finite-budget convergence instead of concealing failure.

Tests compare against the pure-Python reference and check independent numerical or hand-computed
results, input validation, deterministic behavior, and relevant failure/stopping cases.
