# PyTorch decision trees

[Topic and results](../README.md) · [Mathematics](../idea.md) ·
[PyTorch conventions](../../../docs/implementations/pytorch.md)

[Model](model.py) · [Public API](__init__.py) · [Experiment](experiment.py) ·
[Tests](tests/test_torch_decision_tree.py)

```bash
python -m pip install -e ".[dev,pytorch]"
python -m ml_scratch.pytorch.decision_tree.experiment
# Equivalent installed command: ml-pytorch-decision-tree
python -m pytest topics/06_decision_tree/pytorch/tests
```

This is the [NumPy design](../numpy/README.md) in CPU tensors. A stable `torch.argsort`, `gather`,
and a `cumsum` of one-hot labels produce left-child class counts for every feature and threshold.
The impurity of all candidates is computed as one tensor, and a masked feature-major `argmax`
picks the split. The tree is stored as flat preorder tensors, prediction routes all samples one
level per step, and pruning scores every collapse from a single path matrix. Both backends use
the same tie rules, so they build the same trees as scalar Python.

```python
from ml_scratch.pytorch.decision_tree import DecisionTreeClassifier, DecisionTreeConfig

model = DecisionTreeClassifier(class_count=2)
model.fit([[0], [1], [2], [3]], [0, 0, 1, 1], DecisionTreeConfig(max_depth=2))
assert model.predict([2.5]) == 1
assert model.predict_distribution([2.5]).tolist() == [0.0, 1.0]
```

Class counts and node indices are int64, and impurities, thresholds, and distributions are
float64. Integer counts and sizes are cast to float64 before division. Torch would otherwise
promote integer/integer division to its default float32, and the rounded child weights can break
exact gain ties differently from the scalar reference. The randomized scalar-structure tests
detect this. Features are converted to detached CPU float64 at the boundary. Labels must be an
integer tensor or sequence, so float and boolean labels are rejected.

A tree is a discrete structure, so `fit` and `prune` run under `torch.no_grad()` and expose no
gradients. Single-sample `predict` returns a Python integer, and `accuracy` returns a Python float.
`predict_many` and `predict_distribution` return tensors.

[Tests](tests/test_torch_decision_tree.py) mirror the NumPy suite: scalar impurity agreement,
independent brute-force splits, and preorder structure and pruning equal to scalar Python on random
duplicate-heavy multiclass data. They also cover the reference experiment metrics and the
detachment of tensor inputs that require gradients.
