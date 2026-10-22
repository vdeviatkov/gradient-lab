"""Greedy trees whose split search sweeps every feature and threshold in one array pass."""

import math

from ml_scratch.pure.decision_tree import DecisionTreeConfig

import numpy as np
from ml_scratch.numpy.validation import matrix, vector


def _gini(counts):
    p = counts / counts.sum(axis=-1, keepdims=True)
    return np.maximum(0.0, 1 - (p * p).sum(axis=-1))


def _entropy(counts):
    p = counts / counts.sum(axis=-1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        # 0 log 0 is taken as its limit, zero.
        terms = np.where(p > 0, p * np.log2(p), 0.0)
    return -terms.sum(axis=-1)


def _checked_counts(counts):
    c = np.asarray(counts)
    if (
        c.dtype.kind not in "iu"
        or c.ndim == 0
        or c.shape[-1] == 0
        or (c < 0).any()
        or (c.sum(axis=-1) == 0).any()
    ):
        raise ValueError("counts must be non-negative integers with positive total")
    return c


def gini_impurity(counts):
    """Gini impurity over the last axis; leading axes are a batch of nodes."""
    result = _gini(_checked_counts(counts))
    return float(result) if result.ndim == 0 else result


def entropy_impurity(counts):
    """Base-2 entropy over the last axis; leading axes are a batch of nodes."""
    result = _entropy(_checked_counts(counts))
    return float(result) if result.ndim == 0 else result


def _midpoint(value, next_value):
    # Avoid overflow for opposite signs and rounding onto the upper value.
    threshold = value + (next_value - value) / 2
    if not math.isfinite(threshold):
        threshold = value / 2 + next_value / 2
    return value if threshold >= next_value else threshold


def _best_split(x, one_hot, indices, counts, parent_impurity, impurity, config):
    """Return (gain, feature, threshold) of the best admissible split, or None.

    `indices` must be ascending, so a stable sort breaks equal values by sample index.
    """
    n = len(indices)
    values = x[indices]  # (n, d)
    order = np.argsort(values, axis=0, kind="stable")
    sorted_values = np.take_along_axis(values, order, axis=0)
    # Row s of the cumulative counts holds the class counts of the first s + 1 sorted samples,
    # for every feature at once: shape (n - 1, d, K). The right child is the complement.
    left = np.cumsum(one_hot[indices][order], axis=0)[:-1]
    right = counts - left
    sizes = np.arange(1, n)[:, None]
    gain = parent_impurity - sizes / n * impurity(left) - (n - sizes) / n * impurity(right)
    admissible = (
        (sorted_values[:-1] != sorted_values[1:])
        & (sizes >= config.min_samples_leaf)
        & (n - sizes >= config.min_samples_leaf)
    )
    if not admissible.any():
        return None
    # Feature-major order makes argmax's first maximum the lowest feature, then lowest threshold.
    gain = np.where(admissible, gain, -np.inf).T
    feature, position = np.unravel_index(np.argmax(gain), gain.shape)
    threshold = _midpoint(
        float(sorted_values[position, feature]), float(sorted_values[position + 1, feature])
    )
    return float(gain[feature, position]), int(feature), threshold


class DecisionTreeClassifier:
    """Multiclass tree stored as flat arrays indexed by node id in preorder.

    Leaves have `left == right == -1`. Collapsed subtrees stay in the arrays but are unreachable.
    """

    def __init__(self, class_count=2):
        if (
            isinstance(class_count, (bool, np.bool_))
            or not isinstance(class_count, (int, np.integer))
            or class_count < 2
        ):
            raise ValueError("class_count must be an integer >= 2")
        self.class_count = int(class_count)
        self.feature_count = 0
        self.feature = self.threshold = self.left = self.right = None
        self.class_counts = self.impurity = self.prediction = None

    def _validate(self, features, labels, width=None):
        x = matrix(features, width)
        y = np.asarray(labels)
        if y.shape != (len(x),):
            raise ValueError("features and labels must be nonempty and have equal lengths")
        if y.dtype.kind not in "iu" or ((y < 0) | (y >= self.class_count)).any():
            raise ValueError("labels must be integers inside the class range")
        return x, y.astype(np.int64)

    def fit(self, features, labels, config=None):
        config = config or DecisionTreeConfig()
        x, y = self._validate(features, labels)
        impurity = _gini if config.criterion == "gini" else _entropy
        one_hot = np.eye(self.class_count, dtype=np.int64)[y]
        feature, threshold, left, right, class_counts, impurities = [], [], [], [], [], []
        # Popping left before right assigns ids in preorder without recursion.
        stack = [(np.arange(len(x)), 0, None, None)]
        while stack:
            indices, depth, parent, links = stack.pop()
            node = len(class_counts)
            if parent is not None:
                links[parent] = node
            counts = np.bincount(y[indices], minlength=self.class_count)
            node_impurity = float(impurity(counts))
            class_counts.append(counts)
            impurities.append(node_impurity)
            feature.append(0)
            threshold.append(0.0)
            left.append(-1)
            right.append(-1)
            if (
                node_impurity == 0
                or len(indices) < config.min_samples_split
                or (config.max_depth and depth >= config.max_depth)
            ):
                continue
            best = _best_split(x, one_hot, indices, counts, node_impurity, impurity, config)
            if best is None or best[0] <= config.min_impurity_decrease:
                continue
            _, feature[node], threshold[node] = best
            goes_left = x[indices, feature[node]] <= threshold[node]
            stack.append((indices[~goes_left], depth + 1, node, right))
            stack.append((indices[goes_left], depth + 1, node, left))
        self.feature = np.array(feature, dtype=np.int64)
        self.threshold = np.array(threshold, dtype=np.float64)
        self.left = np.array(left, dtype=np.int64)
        self.right = np.array(right, dtype=np.int64)
        self.class_counts = np.array(class_counts, dtype=np.int64)
        self.impurity = np.array(impurities, dtype=np.float64)
        # argmax returns the first maximum, so count ties choose the lowest class.
        self.prediction = self.class_counts.argmax(axis=1)
        self.feature_count = x.shape[1]
        return self

    def _require_fitted(self):
        if self.left is None:
            raise RuntimeError("the model must be fitted first")

    def _route(self, x, record_path=False):
        """Advance every sample one level per step; optionally record each visited node."""
        rows = np.arange(len(x))
        node = np.zeros(len(x), dtype=np.int64)
        path = np.zeros((len(x), len(self.left)), dtype=bool) if record_path else None
        while True:
            if record_path:
                path[rows, node] = True
            internal = self.left[node] >= 0
            if not internal.any():
                return node, path
            goes_left = x[rows, self.feature[node]] <= self.threshold[node]
            child = np.where(goes_left, self.left[node], self.right[node])
            node = np.where(internal, child, node)

    def predict(self, features):
        """Predict one feature vector, matching the scalar API."""
        self._require_fitted()
        row = vector(features, self.feature_count)
        return int(self.prediction[self._route(row[None, :])[0][0]])

    def predict_many(self, features):
        self._require_fitted()
        return self.prediction[self._route(matrix(features, self.feature_count))[0]]

    def predict_distribution(self, features):
        self._require_fitted()
        row = vector(features, self.feature_count)
        counts = self.class_counts[self._route(row[None, :])[0][0]]
        return counts / counts.sum()

    def accuracy(self, features, labels):
        self._require_fitted()
        x, y = self._validate(features, labels, self.feature_count)
        return float(np.mean(self.prediction[self._route(x)[0]] == y))

    def _levels(self):
        frontier = np.array([0])
        while frontier.size:
            yield frontier
            internal = frontier[self.left[frontier] >= 0]
            frontier = np.concatenate((self.left[internal], self.right[internal]))

    @property
    def reachable_nodes(self):
        """Ids of the nodes in the current tree, ascending, which is preorder."""
        self._require_fitted()
        return np.sort(np.concatenate(list(self._levels())))

    @property
    def node_count(self):
        return len(self.reachable_nodes)

    @property
    def leaf_count(self):
        return int((self.left[self.reachable_nodes] < 0).sum())

    @property
    def depth(self):
        self._require_fitted()
        return sum(1 for _ in self._levels()) - 1

    def prune(self, features, labels):
        """Collapse whole subtrees using validation accuracy; never pass test data here."""
        self._require_fitted()
        x, y = self._validate(features, labels, self.feature_count)
        collapsed = 0
        while True:
            levels = list(self._levels())
            # Walking levels bottom-up finishes each child's subtree size before its parent's.
            subtree = np.ones(len(self.left), dtype=np.int64)
            for level in reversed(levels):
                internal = level[self.left[level] >= 0]
                subtree[internal] += subtree[self.left[internal]] + subtree[self.right[internal]]
            candidates = np.sort(np.concatenate(levels))
            candidates = candidates[self.left[candidates] >= 0]
            if not candidates.size:
                return collapsed
            leaves, path = self._route(x, record_path=True)
            correct = self.prediction[leaves] == y
            baseline = int(correct.sum())
            # Collapsing v changes only the samples whose path crosses v: they lose their current
            # correctness and are then right exactly when their label equals prediction[v].
            lost = (path & correct[:, None]).sum(axis=0)
            gained = (path & (self.prediction[None, :] == y[:, None])).sum(axis=0)
            score = (baseline - lost + gained)[candidates]
            removed = (subtree - 1)[candidates]
            keep = score >= baseline
            if not keep.any():
                return collapsed
            # Highest accuracy, then most removed nodes, then first in preorder.
            candidates, score, removed = candidates[keep], score[keep], removed[keep]
            best = score == score.max()
            candidates, removed = candidates[best], removed[best]
            node = candidates[removed == removed.max()][0]
            self.left[node] = self.right[node] = -1
            collapsed += 1
