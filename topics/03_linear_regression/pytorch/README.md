# PyTorch linear regression

[Topic and results](../README.md) · [Mathematics](../idea.md) ·
[PyTorch conventions](../../../docs/implementations/pytorch.md)

[Model](model.py) · [Public API](__init__.py) · [Experiment](experiment.py) ·
[Tests](tests/test_torch_linear_regression.py)

```bash
python -m pip install -e ".[dev,pytorch]"
python -m ml_scratch.pytorch.linear_regression.experiment
python -m pytest topics/03_linear_regression/pytorch/tests
```

All numerical model tensors use CPU float64. Inputs accept the existing pair datasets or matrices;
external tensors are detached and normalized at the boundary. See the shared conventions for
return types, configuration reuse, and numerical-comparison limits.

The loss is full MSE, with manual derivatives `2 * X.T @ residual / n` and
`2 * mean(residual)`. Batch, SGD, and mini-batch updates use the scalar reference's configuration,
shuffle stream, and stopping rules; fitting continues from current weights. The C++ half-MSE
convention requires half its nominal learning rate for equivalent updates here.

```python
from ml_scratch.pytorch.linear_regression import LinearRegression

model = LinearRegression(1)
model.fit_closed_form((((0,), 1), ((1,), 3), ((2,), 5)))
assert abs(model.predict([3]) - 7) < 1e-12
```

Normal equations are built with tensor products and solved by explicit partial-pivot Gaussian
elimination. Singular pivots at absolute magnitude `1e-12` or less are refused. Normal equations
square the condition number; this educational solver deliberately exposes their limitations.
Hessian diagnostics use explicit power iteration and a spectral shift, matching the NumPy
reference, including the limitation that an all-ones start can miss an orthogonal eigenspace.
`standardize` uses population deviation and returns pair datasets; constant columns become zero.
R² refuses constant targets. Non-finite training MSE is reported as divergence.

`autograd_gradient` builds a fresh MSE graph on cloned parameters, without changing weights or
accumulating `.grad` fields. Tests compare it with manual gradients, scalar Python, and central
differences. They also check pivoting, singularity, all optimizers, Hessian estimates, scaling,
and the experiment's stability and convergence criteria.
