# NumPy Perceptron

[Topic and results](../README.md) · [Mathematics](../idea.md) ·
[NumPy conventions](../../../docs/implementations/numpy.md)

[Model](model.py) · [Public API](__init__.py) · [Experiment](experiment.py) ·
[Tests](tests/test_numpy_perceptron.py)

```bash
python -m pip install -e ".[dev,numpy]"
python -m ml_scratch.numpy.perceptron.experiment
python -m pytest topics/01_perceptron_logic_gates/numpy/tests
```

The model computes scores with array dot products and updates all feature weights together.
Training remains online: every sample sees the parameters after the previous update. Replacing
this with a single batch update would teach a different algorithm. Zero score predicts class 1.
Local `random.Random` initialization and shuffling retain the scalar reference's draw order.
`fit` continues from the current weights, and reports mistake counts and non-convergence.

```python
from ml_scratch.numpy.perceptron import Perceptron
from ml_scratch.pure.datasets import logic_gate_dataset

model = Perceptron(2, seed=7)
result = model.fit(logic_gate_dataset("and"))
assert result.converged
assert model.predict([1, 1]) == 1
```

Tests compare against the pure-Python reference and check independent numerical or hand-computed
results, input validation, deterministic behavior, and relevant failure/stopping cases.
