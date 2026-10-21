"""Greedy multiclass trees with explicit impurity sweeps and reduced-error pruning."""

import math
from dataclasses import dataclass


def _counts(counts):
    if not counts or any(type(n) is not int or n < 0 for n in counts) or sum(counts) == 0:
        raise ValueError("counts must be non-negative integers with positive total")
    return sum(counts)


def gini_impurity(counts):
    total = _counts(counts)
    return max(0.0, 1 - sum((n / total) ** 2 for n in counts))


def entropy_impurity(counts):
    total = _counts(counts)
    return -sum((n / total) * math.log2(n / total) for n in counts if n)


@dataclass(frozen=True)
class DecisionTreeConfig:
    criterion: str = "gini"
    max_depth: int = 0  # Zero means unlimited, matching C++.
    min_samples_split: int = 2
    min_samples_leaf: int = 1
    min_impurity_decrease: float = 0.0

    def __post_init__(self):
        if self.criterion not in ("gini", "entropy"):
            raise ValueError("criterion must be gini or entropy")
        for name, minimum in (("max_depth", 0), ("min_samples_split", 2), ("min_samples_leaf", 1)):
            value = getattr(self, name)
            if type(value) is not int or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if not math.isfinite(self.min_impurity_decrease) or self.min_impurity_decrease < 0:
            raise ValueError("min_impurity_decrease must be finite and non-negative")


@dataclass
class DecisionTreeNode:
    class_counts: tuple[int, ...]
    impurity: float
    prediction: int
    feature: int = 0
    threshold: float = 0.0
    left: "DecisionTreeNode | None" = None
    right: "DecisionTreeNode | None" = None

    @property
    def leaf(self):
        return self.left is None


class DecisionTreeClassifier:
    def __init__(self, class_count=2):
        if type(class_count) is not int or class_count < 2:
            raise ValueError("class_count must be an integer >= 2")
        self.class_count = class_count
        self.root = None
        self.feature_count = 0

    def _validate(self, features, labels, width=None):
        if not features or len(features) != len(labels):
            raise ValueError("features and labels must be nonempty and have equal lengths")
        width = len(features[0]) if width is None else width
        if width == 0 or any(len(row) != width for row in features):
            raise ValueError("feature rows must have the same positive width")
        if any(not math.isfinite(v) for row in features for v in row):
            raise ValueError("features must be finite")
        if any(type(y) is not int or not 0 <= y < self.class_count for y in labels):
            raise ValueError("labels must be integers inside the class range")
        return width

    def fit(self, features, labels, config=None):
        config = config or DecisionTreeConfig()
        width = self._validate(features, labels)
        impurity = gini_impurity if config.criterion == "gini" else entropy_impurity

        def build(indices, depth):
            counts = [0] * self.class_count
            for i in indices:
                counts[labels[i]] += 1
            node = DecisionTreeNode(
                tuple(counts),
                impurity(counts),
                max(range(self.class_count), key=counts.__getitem__),
            )
            n = len(indices)
            if (
                node.impurity == 0
                or n < config.min_samples_split
                or (config.max_depth and depth >= config.max_depth)
            ):
                return node
            best = None
            for feature in range(width):
                ordered = sorted(indices, key=lambda i: (features[i][feature], i))
                left, right = [0] * self.class_count, counts.copy()
                for position in range(n - 1):
                    i = ordered[position]
                    left[labels[i]] += 1
                    right[labels[i]] -= 1
                    value = features[i][feature]
                    next_value = features[ordered[position + 1]][feature]
                    size = position + 1
                    if (
                        value == next_value
                        or size < config.min_samples_leaf
                        or n - size < config.min_samples_leaf
                    ):
                        continue
                    gain = (
                        node.impurity - size / n * impurity(left) - (n - size) / n * impurity(right)
                    )
                    if best is None or gain > best[0]:
                        # Avoid overflow for opposite signs and rounding onto the upper value.
                        threshold = value + (next_value - value) / 2
                        if not math.isfinite(threshold):
                            threshold = value / 2 + next_value / 2
                        if threshold >= next_value:
                            threshold = value
                        best = gain, feature, threshold
            if best is not None and best[0] > config.min_impurity_decrease:
                _, node.feature, node.threshold = best
                ordered = sorted(indices, key=lambda i: (features[i][node.feature], i))
                left_indices = [i for i in ordered if features[i][node.feature] <= node.threshold]
                right_indices = [i for i in ordered if features[i][node.feature] > node.threshold]
                node.left = build(left_indices, depth + 1)
                node.right = build(right_indices, depth + 1)
            return node

        root = build(list(range(len(features))), 0)
        self.root, self.feature_count = root, width
        return self

    def _require_fitted(self):
        if self.root is None:
            raise RuntimeError("the model must be fitted first")

    def _leaf_for(self, features):
        self._require_fitted()
        if len(features) != self.feature_count or any(not math.isfinite(v) for v in features):
            raise ValueError("features must be finite and match the fitted width")
        node = self.root
        while not node.leaf:
            node = node.left if features[node.feature] <= node.threshold else node.right
        return node

    def predict(self, features):
        """Predict one feature vector, matching the C++ API."""
        return self._leaf_for(features).prediction

    def predict_distribution(self, features):
        counts = self._leaf_for(features).class_counts
        return tuple(n / sum(counts) for n in counts)

    def accuracy(self, features, labels):
        self._require_fitted()
        self._validate(features, labels, self.feature_count)
        return sum(self.predict(x) == y for x, y in zip(features, labels, strict=True)) / len(
            labels
        )

    def _nodes(self):
        self._require_fitted()
        stack = [(self.root, 0)]
        while stack:
            node, depth = stack.pop()
            yield node, depth
            if not node.leaf:
                stack.extend(((node.right, depth + 1), (node.left, depth + 1)))

    @property
    def node_count(self):
        return sum(1 for _ in self._nodes())

    @property
    def leaf_count(self):
        return sum(node.leaf for node, _ in self._nodes())

    @property
    def depth(self):
        return max(depth for _, depth in self._nodes())

    def prune(self, features, labels):
        """Collapse whole subtrees using validation accuracy; never pass test data here."""
        self._require_fitted()
        self._validate(features, labels, self.feature_count)
        collapsed = 0
        while True:
            baseline = self.accuracy(features, labels)
            best = None
            for node, _ in list(self._nodes()):
                if node.leaf:
                    continue
                stack, removed = [node.left, node.right], 0
                while stack:
                    child = stack.pop()
                    removed += 1
                    if not child.leaf:
                        stack.extend((child.left, child.right))
                left, right = node.left, node.right
                node.left = node.right = None
                try:
                    score = self.accuracy(features, labels)
                finally:
                    node.left, node.right = left, right
                if score >= baseline and (best is None or (score, removed) > best[:2]):
                    best = score, removed, node
            if best is None:
                return collapsed
            best[2].left = best[2].right = None
            collapsed += 1
