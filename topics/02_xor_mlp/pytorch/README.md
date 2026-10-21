# PyTorch mlp

[Topic and results](../README.md) · [Mathematics](../idea.md) ·
[PyTorch conventions](../../../docs/implementations/pytorch.md)

[Model](model.py) · [Public API](__init__.py) · [Experiment](experiment.py) ·
[Tests](tests/test_torch_mlp.py)

```bash
python -m pip install -e ".[dev,pytorch]"
python -m ml_scratch.pytorch.mlp.experiment
python -m pytest topics/02_xor_mlp/pytorch/tests
```

All numerical model tensors use CPU float64. Inputs accept the existing pair datasets or matrices;
external tensors are detached and normalized at the boundary. See the shared conventions for
return types, configuration reuse, and numerical-comparison limits.

The forward pass is `H = tanh(X @ W.T + b)` followed by one affine logit per sample. Mean BCE
uses the target-specific logit sign and `logaddexp`. Explicit backpropagation computes
`delta = (sigmoid(z) - y) / n` and
`hidden_delta = delta[:, None] * output_weights * (1 - H**2)` before tensor matrix reductions.
Full-batch updates happen under `no_grad`; no layer or optimizer object replaces these equations.

```python
from ml_scratch.pytorch.mlp import XorMlp
from ml_scratch.pure.datasets import logic_gate_dataset

model = XorMlp(seed=7)
data = logic_gate_dataset("xor")
manual = model.gradient(data)
automatic = model.autograd_gradient(data)
result = model.fit(data)
assert result.converged
```

`autograd_gradient` differentiates an independent forward graph on cloned parameters and leaves
model state unchanged. `numerical_gradient` supplies central differences and restores parameters
on exit. `parameters` returns immutable Python tuples, avoiding aliases to tensor storage;
`set_parameters` validates and copies values. Initial Glorot weights and zero biases match scalar
Python. `fit` continues current parameters and tests loss after every update.
The experiment sweeps widths 1, 2, 3, 4, 8 over seeds 0–9 by default; `--seeds 50` matches the
scalar sweep budget. Tests compare every gradient field against autograd, central differences,
and the scalar derivation, then verify updates, stopping, and extreme logits.
