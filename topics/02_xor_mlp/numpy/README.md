# NumPy XOR MLP

[Topic and results](../README.md) · [Mathematics](../idea.md) ·
[NumPy conventions](../../../docs/implementations/numpy.md)

[Model](model.py) · [Public API](__init__.py) · [Experiment](experiment.py) ·
[Tests](tests/test_numpy_mlp.py)

```bash
python -m pip install -e ".[dev,numpy]"
python -m ml_scratch.numpy.mlp.experiment
python -m pytest topics/02_xor_mlp/numpy/tests
```

For `X` of shape `(n, d)`, input weights have shape `(h, d)` and hidden activations are
`H = tanh(X @ W.T + b)`. The output is one logit per row. The loss is mean binary cross-entropy,
evaluated as `logaddexp(0, -z)` for positive targets and `logaddexp(0, z)` for negatives.
Backpropagation explicitly computes `delta = (sigmoid(z) - y) / n`, then
`hidden_delta = delta[:, None] * output_weights * (1 - H**2)`.
The matrix gradients are `hidden_delta.T @ X` and `H.T @ delta`.

Initialization uses the same Glorot limits and local Python random stream as the scalar model;
full-batch descent and post-update loss stopping match it. `fit` continues from current parameters.
`parameters` is an immutable tuple snapshot; `set_parameters` copies its inputs. Gradient fields
contain arrays, while numerical gradients use central differences and restore parameters on exit.

```python
from ml_scratch.numpy.mlp import XorMlp
from ml_scratch.pure.datasets import logic_gate_dataset

model = XorMlp(seed=7)
result = model.fit(logic_gate_dataset("xor"))
assert result.converged
```

The experiment sweeps widths 1, 2, 3, 4, and 8 over seeds 0–9 by default. Pass `--seeds 50`
for the scalar experiment's larger sweep. No success rate is inferred from a single seed.

Tests compare against the pure-Python reference and check independent numerical or hand-computed
results, input validation, deterministic behavior, and relevant failure/stopping cases.
