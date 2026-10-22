# NumPy decision trees

[Topic and results](../README.md) · [Mathematics](../idea.md) ·
[NumPy conventions](../../../docs/implementations/numpy.md)

[Model](model.py) · [Public API](__init__.py) · [Experiment](experiment.py) ·
[Tests](tests/test_numpy_decision_tree.py)

```bash
python -m pip install -e ".[dev,numpy]"
python -m ml_scratch.numpy.decision_tree.experiment
# Equivalent installed command: ml-numpy-decision-tree
python -m pytest topics/06_decision_tree/numpy/tests
```

The greedy split search runs as one array pass over all features and thresholds per node. A
stable `argsort` orders the node's samples along every feature at once. A cumulative sum of
one-hot labels then gives the left-child class counts for every candidate, shape `(n - 1, d, K)`,
and the right child is the parent counts minus that. Gini or base-2 entropy is evaluated on the
whole batch, and invalid candidates are masked to `-inf`: equal adjacent values, or a child below
`min_samples_leaf`. Taking `argmax` in feature-major order returns the first maximum, so ties
choose the lowest feature and then the lowest threshold, as in the scalar sweep. The cost per node
stays O(d n log n + d n K), with O(d n K) temporary storage. No library tree or impurity routine
replaces the algorithm.

```python
from ml_scratch.numpy.decision_tree import DecisionTreeClassifier, DecisionTreeConfig

model = DecisionTreeClassifier(class_count=2)
model.fit([[0], [1], [2], [3]], [0, 0, 1, 1], DecisionTreeConfig(max_depth=2))
assert model.predict([2.5]) == 1
assert model.predict_distribution([2.5]).tolist() == [0.0, 1.0]
```

The tree is stored as flat arrays indexed by node id: `feature`, `threshold`, `left`, `right`
(`-1` at leaves), `class_counts`, `impurity`, and `prediction`. Growth uses an explicit stack that
assigns ids in preorder, so it does not hit Python's recursion limit. `predict_many` routes every
sample one level per step with `where`. `predict` and `accuracy` keep the scalar meanings, and
`predict_distribution` returns an array. `DecisionTreeConfig` is the scalar configuration type.
Labels must be an integer array in `[0, class_count)`, so floats and booleans are rejected.

Reduced-error pruning scores every candidate collapse from one routing pass. That pass records a
boolean path matrix of which nodes each validation sample visits. Collapsing node `v` changes only
the samples that pass through `v`: each loses its current correctness and becomes correct exactly
when its label equals `prediction[v]`. So
`score(v) = correct - Σ_{s through v} correct_s + Σ_{s through v} [y_s = prediction[v]]`.
The scalar version re-evaluates the tree once per candidate. Selection is the same: highest
validation accuracy, then the most removed descendants, then the first node in preorder.
Collapsed subtrees remain in the arrays but are unreachable. `node_count`, `leaf_count`, `depth`,
and `reachable_nodes` describe only the current tree.

The [experiment](experiment.py) reuses the scalar seeded dataset (seed `20260908`), the depth
sweep, the shallowest-tie validation selection, and the success checks.

[Tests](tests/test_numpy_decision_tree.py) check batched impurities against the scalar functions
and the sweep against an independent brute-force split search. They require the full preorder
structure and the pruning sequence to equal the scalar tree's. That covers duplicate-heavy random
multiclass data under four configurations, and the experiment's unlimited tree. Tests also cover
ties, zero-gain XOR, stopping rules, extreme midpoints, invalid input, failed refits, and the
recorded reference metrics.
