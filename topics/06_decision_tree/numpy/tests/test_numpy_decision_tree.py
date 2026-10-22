import math
import random

import numpy as np
import pytest
from ml_scratch.numpy.decision_tree import (
    DecisionTreeClassifier,
    DecisionTreeConfig,
    entropy_impurity,
    gini_impurity,
)
from ml_scratch.numpy.decision_tree.experiment import run_experiment
from ml_scratch.pure.decision_tree import DecisionTreeClassifier as ScalarTree
from ml_scratch.pure.decision_tree import entropy_impurity as scalar_entropy
from ml_scratch.pure.decision_tree import gini_impurity as scalar_gini
from ml_scratch.pure.decision_tree.experiment import build_dataset


def scalar_preorder(tree):
    stack, nodes = [tree.root], []
    while stack:
        node = stack.pop()
        split = None if node.leaf else (node.feature, node.threshold)
        nodes.append((node.class_counts, split))
        if not node.leaf:
            stack.extend((node.right, node.left))
    return nodes


def array_preorder(tree):
    return [
        (
            tuple(tree.class_counts[i].tolist()),
            None if tree.left[i] < 0 else (int(tree.feature[i]), float(tree.threshold[i])),
        )
        for i in tree.reachable_nodes
    ]


def random_problem(seed, rows=60, width=3, classes=3):
    rng = random.Random(seed)
    # Small integer features force many duplicate values and exactly tied gains.
    x = [[rng.randrange(6) for _ in range(width)] for _ in range(rows)]
    return x, [rng.randrange(classes) for _ in x]


@pytest.mark.parametrize(
    "impurity,scalar,balanced",
    [(gini_impurity, scalar_gini, 0.5), (entropy_impurity, scalar_entropy, 1)],
)
def test_batched_impurity_matches_scalar(impurity, scalar, balanced):
    assert impurity([2, 2]) == pytest.approx(balanced)
    assert impurity([0, 4]) == 0
    batch = [[3, 0, 1], [1, 1, 1], [0, 0, 5], [7, 2, 4]]
    np.testing.assert_allclose(impurity(batch), [scalar(c) for c in batch], atol=1e-15)
    for counts in ([], [0, 0], [-1, 2], [1.5, 2], [[1, 1], [0, 0]]):
        with pytest.raises(ValueError):
            impurity(counts)


@pytest.mark.parametrize("criterion", ["gini", "entropy"])
def test_multiclass_and_distribution(criterion):
    tree = DecisionTreeClassifier(3).fit(
        [[0], [1], [2], [3], [4], [5]], [0, 0, 1, 1, 2, 2], DecisionTreeConfig(criterion=criterion)
    )
    assert [tree.predict([x]) for x in range(6)] == [0, 0, 1, 1, 2, 2]
    np.testing.assert_array_equal(tree.predict_many([[x] for x in range(6)]), [0, 0, 1, 1, 2, 2])
    np.testing.assert_array_equal(tree.predict_distribution([4]), [0, 0, 1])
    assert (tree.node_count, tree.leaf_count, tree.depth) == (5, 3, 2)


def test_ties_duplicates_and_zero_gain_xor():
    tree = DecisionTreeClassifier().fit([[0, 0], [1, 1]], [0, 1])
    assert (tree.feature[0], tree.threshold[0]) == (0, 0.5)
    assert tree.predict([0.5, 100]) == 0  # Equality goes left.
    tree.fit([[0], [0]], [1, 0])
    assert tree.predict([0]) == 0
    np.testing.assert_array_equal(tree.predict_distribution([0]), [0.5, 0.5])
    tree.fit([[0, 0], [0, 1], [1, 0], [1, 1]], [0, 1, 1, 0])
    assert tree.node_count == 1  # Greedy positive-gain splits cannot learn exact XOR.


@pytest.mark.parametrize(
    "config",
    [
        DecisionTreeConfig(max_depth=1),
        DecisionTreeConfig(min_samples_split=5),
        DecisionTreeConfig(min_samples_leaf=3),
        DecisionTreeConfig(min_impurity_decrease=0.5),
    ],
)
def test_stopping_rules(config):
    tree = DecisionTreeClassifier().fit([[0], [1], [2], [3]], [0, 0, 1, 1], config)
    assert tree.depth <= 1
    if config.max_depth != 1:
        assert tree.node_count == 1


@pytest.mark.parametrize("criterion", ["gini", "entropy"])
def test_vectorized_sweep_matches_independent_brute_force(criterion):
    x, y = random_problem(41, rows=30, width=2)
    impurity = scalar_gini if criterion == "gini" else scalar_entropy
    parent = impurity([y.count(c) for c in range(3)])
    candidates = []
    for j in range(2):
        values = sorted({row[j] for row in x})
        for a, b in zip(values, values[1:]):
            threshold = (a + b) / 2
            left = [label for row, label in zip(x, y) if row[j] <= threshold]
            right = [label for row, label in zip(x, y) if row[j] > threshold]
            gain = parent - len(left) / len(y) * impurity([left.count(c) for c in range(3)])
            gain -= len(right) / len(y) * impurity([right.count(c) for c in range(3)])
            candidates.append((gain, j, threshold))
    expected = max(candidates, key=lambda c: c[0])
    tree = DecisionTreeClassifier(3).fit(x, y, DecisionTreeConfig(criterion=criterion, max_depth=1))
    assert (tree.feature[0], tree.threshold[0]) == expected[1:]


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize(
    "config",
    [
        DecisionTreeConfig(),
        DecisionTreeConfig(criterion="entropy"),
        DecisionTreeConfig(max_depth=3, min_samples_leaf=2),
        DecisionTreeConfig(min_samples_split=6, min_impurity_decrease=0.01),
    ],
)
def test_whole_tree_and_pruning_match_scalar(seed, config):
    x, y = random_problem(seed)
    validation_x, validation_y = random_problem(seed + 100, rows=30)
    tree = DecisionTreeClassifier(3).fit(x, y, config)
    scalar = ScalarTree(3).fit(x, y, config)
    assert array_preorder(tree) == scalar_preorder(scalar)
    assert tree.prune(validation_x, validation_y) == scalar.prune(validation_x, validation_y)
    assert array_preorder(tree) == scalar_preorder(scalar)
    assert tree.accuracy(x, y) == scalar.accuracy(x, y)
    np.testing.assert_array_equal(tree.predict_many(x), [scalar.predict(row) for row in x])


def test_experiment_tree_matches_scalar_structure():
    data = build_dataset()
    tree = DecisionTreeClassifier().fit(data.train.features, data.train.labels)
    scalar = ScalarTree().fit(data.train.features, data.train.labels)
    assert array_preorder(tree) == scalar_preorder(scalar)
    np.testing.assert_allclose(tree.impurity[0], scalar.root.impurity, atol=1e-15)


def test_extreme_and_adjacent_thresholds():
    for a, b in [(-1e308, 1e308), (1.0, math.nextafter(1.0, math.inf))]:
        tree = DecisionTreeClassifier().fit([[a], [b]], [0, 1])
        assert math.isfinite(tree.threshold[0])
        assert tree.predict([a]) == 0
        assert tree.predict([b]) == 1


def test_pruning_collapse_keep_and_refit():
    tree = DecisionTreeClassifier().fit([[0], [1], [2], [3]], [0, 0, 1, 1])
    assert tree.prune([[0], [3]], [0, 1]) == 0
    assert tree.node_count == 3
    assert tree.prune([[0], [3]], [0, 0]) == 1
    assert tree.node_count == 1
    np.testing.assert_array_equal(tree.predict_distribution([3]), [0.5, 0.5])
    assert tree.prune([[0]], [0]) == 0
    tree.fit([[0], [1]], [0, 1])
    assert tree.node_count == 3


def test_invalid_data_unfitted_and_failed_fit_preserves_state():
    tree = DecisionTreeClassifier()
    with pytest.raises(RuntimeError):
        tree.predict([0])
    for x, y in [
        ([], []),
        ([[0]], []),
        ([[]], [0]),
        ([[0], [0, 1]], [0, 1]),
        ([[math.inf]], [0]),
        ([[0]], [2]),
        ([[0]], [-1]),
        ([[0]], [0.5]),
        ([[0]], [True]),
    ]:
        with pytest.raises(ValueError):
            tree.fit(x, y)
    tree.fit([[0], [1]], [0, 1])
    threshold = tree.threshold.copy()
    with pytest.raises(ValueError):
        tree.fit([[0], [1]], [0, 3])
    np.testing.assert_array_equal(tree.threshold, threshold)
    for row in ([0, 1], [math.nan]):
        with pytest.raises(ValueError):
            tree.predict(row)
    with pytest.raises(ValueError):
        tree.prune([[0]], [2])
    for class_count in (1, 2.0, True):
        with pytest.raises(ValueError):
            DecisionTreeClassifier(class_count)


def test_numpy_experiment_reproduces_reference_results():
    result = run_experiment()
    assert result.selected_depth == 2
    assert [r.leaves for r in result.depth_sweep] == [2, 4, 8, 12, 18, 25, 34, 40, 43, 45, 47, 48]
    unlimited, pruned = result.unlimited, result.pruned
    assert (unlimited.nodes, unlimited.leaves, unlimited.depth) == (95, 48, 12)
    assert (unlimited.train, unlimited.validation, unlimited.test) == pytest.approx(
        (1, 0.74, 121 / 150)
    )
    assert (pruned.nodes, pruned.leaves, pruned.depth, result.collapsed) == (13, 7, 5, 4)
    assert (pruned.train, pruned.validation, pruned.test) == pytest.approx(
        (257 / 300, 0.84, 131 / 150)
    )
    assert result.baseline == pytest.approx(73 / 150)
    assert run_experiment() == result
