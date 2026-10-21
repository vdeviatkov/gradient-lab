import math
import random

import pytest
from ml_scratch.pure.decision_tree import (
    DecisionTreeClassifier,
    DecisionTreeConfig,
    entropy_impurity,
    gini_impurity,
)


@pytest.mark.parametrize("impurity,balanced", [(gini_impurity, 0.5), (entropy_impurity, 1)])
def test_impurity(impurity, balanced):
    assert impurity([2, 2]) == pytest.approx(balanced)
    assert impurity([0, 4]) == 0
    for counts in ([], [0, 0], [-1, 2], [1.5, 2]):
        with pytest.raises(ValueError):
            impurity(counts)


@pytest.mark.parametrize("criterion", ["gini", "entropy"])
def test_multiclass_and_distribution(criterion):
    tree = DecisionTreeClassifier(3).fit(
        [[0], [1], [2], [3], [4], [5]], [0, 0, 1, 1, 2, 2], DecisionTreeConfig(criterion=criterion)
    )
    assert [tree.predict([x]) for x in range(6)] == [0, 0, 1, 1, 2, 2]
    assert tree.predict_distribution([4]) == (0, 0, 1)
    assert tree.node_count == 5
    assert tree.leaf_count == 3


def test_ties_duplicates_and_zero_gain_xor():
    tree = DecisionTreeClassifier().fit([[0, 0], [1, 1]], [0, 1])
    assert tree.root.feature == 0
    assert tree.root.threshold == 0.5
    assert tree.predict([0.5, 100]) == 0  # Equality goes left.
    tree.fit([[0], [0]], [1, 0])
    assert tree.predict([0]) == 0
    assert tree.predict_distribution([0]) == (0.5, 0.5)
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
def test_split_sweep_matches_independent_brute_force(criterion):
    rng = random.Random(41)
    x = [[rng.randrange(5), rng.randrange(5)] for _ in range(30)]
    y = [rng.randrange(3) for _ in x]
    impurity = gini_impurity if criterion == "gini" else entropy_impurity
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
    assert (tree.root.feature, tree.root.threshold) == expected[1:]


def test_extreme_and_adjacent_thresholds():
    for a, b in [(-1e308, 1e308), (1.0, math.nextafter(1.0, math.inf))]:
        tree = DecisionTreeClassifier().fit([[a], [b]], [0, 1])
        assert math.isfinite(tree.root.threshold)
        assert tree.predict([a]) == 0
        assert tree.predict([b]) == 1


def test_pruning_collapse_and_keep():
    tree = DecisionTreeClassifier().fit([[0], [1], [2], [3]], [0, 0, 1, 1])
    assert tree.prune([[0], [3]], [0, 1]) == 0
    assert tree.node_count == 3
    assert tree.prune([[0], [3]], [0, 0]) == 1
    assert tree.node_count == 1
    assert tree.predict_distribution([3]) == (0.5, 0.5)
    assert tree.prune([[0]], [0]) == 0
    tree.fit([[0], [1]], [0, 1])
    assert tree.node_count == 3


@pytest.mark.parametrize(
    "kwargs",
    [
        {"criterion": "other"},
        {"max_depth": -1},
        {"max_depth": 1.5},
        {"min_samples_split": 1},
        {"min_samples_leaf": 0},
        {"min_impurity_decrease": -1},
        {"min_impurity_decrease": math.nan},
    ],
)
def test_invalid_config(kwargs):
    with pytest.raises(ValueError):
        DecisionTreeConfig(**kwargs)


def test_invalid_data_and_unfitted():
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
        ([[0]], [0.5]),
    ]:
        with pytest.raises(ValueError):
            tree.fit(x, y)
    tree.fit([[0], [1]], [0, 1])
    for row in ([0, 1], [math.nan]):
        with pytest.raises(ValueError):
            tree.predict(row)
    with pytest.raises(ValueError):
        tree.prune([[0]], [2])
