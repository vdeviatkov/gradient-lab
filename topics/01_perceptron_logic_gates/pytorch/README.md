# PyTorch perceptron

[Topic and results](../README.md) · [Mathematics](../idea.md) ·
[PyTorch conventions](../../../docs/implementations/pytorch.md)

[Model](model.py) · [Public API](__init__.py) · [Experiment](experiment.py) ·
[Tests](tests/test_torch_perceptron.py)

```bash
python -m pip install -e ".[dev,pytorch]"
python -m ml_scratch.pytorch.perceptron.experiment
python -m pytest topics/01_perceptron_logic_gates/pytorch/tests
```

All numerical model tensors use CPU float64. Inputs accept the existing pair datasets or matrices;
external tensors are detached and normalized at the boundary. See the shared conventions for
return types, configuration reuse, and numerical-comparison limits.

The model evaluates tensor dot products and updates all feature weights together. Online updates
remain sequential: a later sample sees earlier updates from the same epoch. The step activation
predicts 1 at zero. Local Python random generators preserve the scalar initialization and shuffle
order without modifying the global Python or Torch random state. `fit` continues current weights.

```python
from ml_scratch.pytorch.perceptron import Perceptron
from ml_scratch.pure.datasets import logic_gate_dataset

model = Perceptron(2, seed=7)
result = model.fit(logic_gate_dataset("and"))
assert result.converged
```

Tests cover scalar-equivalent histories, hand-computed updates, batch prediction, expected XOR
non-convergence, invalid data, and random-state isolation.
