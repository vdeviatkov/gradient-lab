# Pure-Python decision trees

[Topic and measured results](../README.md) · [Mathematics](../idea.md) ·
[Python conventions](../../../docs/implementations/pure-python.md)

The standard-library-only [model](model.py) implements multiclass classification with Gini or
base-2 entropy, exhaustive sorted threshold sweeps, stopping controls, leaf class distributions,
and reduced-error pruning. The [public API](__init__.py) explicitly exports the model types.

```bash
python -m pip install -e ".[dev]"
python -m ml_scratch.pure.decision_tree.experiment
# Equivalent installed command: ml-decision-tree
python -m pytest topics/06_decision_tree/python/tests
```

```python
from ml_scratch.pure.decision_tree import DecisionTreeClassifier, DecisionTreeConfig

model = DecisionTreeClassifier(class_count=2)
model.fit([[0], [1], [2], [3]], [0, 0, 1, 1], DecisionTreeConfig(max_depth=2))
assert model.predict([2.5]) == 1
assert model.predict_distribution([2.5]) == (0.0, 1.0)
```

`predict` accepts one vector. `fit`, `accuracy`, and `prune` accept separate feature rows and integer
labels in `[0, class_count)`. `max_depth=0` means unlimited; the root is depth zero. Inspection
properties `node_count`, `leaf_count`, and `depth` describe the reachable tree.

A split must strictly exceed `min_impurity_decrease`; zero-gain XOR therefore remains a leaf.
Ties choose the lowest feature, then lowest threshold, and the lowest majority class. Prediction
routes equality left. Midpoints guard against overflow and rounding onto the upper value.
The sweep maintains class counts, costing O(d n log n + d n K) per node for d features and K classes.
Growth uses recursive calls and is intended for small educational datasets; extremely deep trees
can exceed Python's recursion limit.

Pruning considers all internal nodes each round, maximizing validation accuracy, then the number
of removed descendants, then choosing the first node in preorder. A collapsed node retains its
training class counts. Pruning never lowers validation accuracy, but does not guarantee better
test accuracy. Use a separate validation split and reserve the test split for reporting.

The [experiment](experiment.py) reproduces the C++ noisy-Gaussian experiment with seed `20260908`,
including its depth selection and success checks. Those checks describe this fixed experiment,
not guarantees for arbitrary datasets. See [recorded measurements](../README.md#python-reproduction).

[Model tests](tests/test_decision_tree.py) cover impurity, independent brute-force split selection,
multiclass prediction, ties, stopping limits, extreme thresholds, pruning, refitting, and invalid
inputs. [Experiment tests](tests/test_decision_tree_experiment.py) check reproducibility, split
disjointness, and the recorded C++ reference metrics.
