"""Greedy trees whose split search sweeps every feature and threshold in one tensor pass."""

import math
from numbers import Integral

import torch
from ml_scratch.pure.decision_tree import DecisionTreeConfig

from ml_scratch.pytorch.validation import matrix, vector

_INTEGER_TYPES = (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)


def _gini(counts):
    # Integer division would produce the default float32 dtype; impurity is float64 throughout.
    p = counts.to(torch.float64) / counts.sum(dim=-1, keepdim=True)
    return torch.clamp(1 - (p * p).sum(dim=-1), min=0.0)


def _entropy(counts):
    p = counts.to(torch.float64) / counts.sum(dim=-1, keepdim=True)
    # 0 log 0 is taken as its limit, zero; where() discards the NaN from 0 * -inf.
    terms = torch.where(p > 0, p * torch.log2(p), torch.zeros_like(p))
    return -terms.sum(dim=-1)


def _checked_counts(counts):
    try:
        c = torch.as_tensor(counts, device="cpu")
    except (TypeError, RuntimeError, ValueError) as error:
        raise ValueError("counts must be non-negative integers with positive total") from error
    if (
        c.dtype not in _INTEGER_TYPES
        or c.ndim == 0
        or c.shape[-1] == 0
        or (c < 0).any()
        or (c.sum(dim=-1) == 0).any()
    ):
        raise ValueError("counts must be non-negative integers with positive total")
    return c


def gini_impurity(counts):
    """Gini impurity over the last dimension; leading dimensions are a batch of nodes."""
    result = _gini(_checked_counts(counts))
    return float(result) if result.ndim == 0 else result


def entropy_impurity(counts):
    """Base-2 entropy over the last dimension; leading dimensions are a batch of nodes."""
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
    order = torch.argsort(values, dim=0, stable=True)
    sorted_values = torch.gather(values, 0, order)
    # Row s of the cumulative counts holds the class counts of the first s + 1 sorted samples,
    # for every feature at once: shape (n - 1, d, K). The right child is the complement.
    left = torch.cumsum(one_hot[indices][order], dim=0)[:-1]
    right = counts - left
    # Float64 sizes: integer division would round the child weights to default float32.
    sizes = torch.arange(1, n, dtype=torch.float64)[:, None]
    gain = parent_impurity - sizes / n * impurity(left) - (n - sizes) / n * impurity(right)
    admissible = (
        (sorted_values[:-1] != sorted_values[1:])
        & (sizes >= config.min_samples_leaf)
        & (n - sizes >= config.min_samples_leaf)
    )
    if not admissible.any():
        return None
    # Feature-major order makes argmax's first maximum the lowest feature, then lowest threshold.
    gain = torch.where(admissible, gain, -math.inf).T
    feature, position = divmod(int(torch.argmax(gain)), n - 1)
    threshold = _midpoint(
        float(sorted_values[position, feature]), float(sorted_values[position + 1, feature])
    )
    return float(gain[feature, position]), feature, threshold


class DecisionTreeClassifier:
    """Multiclass tree stored as flat tensors indexed by node id in preorder.

    Leaves have `left == right == -1`. Collapsed subtrees stay in the tensors but are unreachable.
    """

    def __init__(self, class_count=2):
        if (
            isinstance(class_count, bool)
            or not isinstance(class_count, Integral)
            or class_count < 2
        ):
            raise ValueError("class_count must be an integer >= 2")
        self.class_count = int(class_count)
        self.feature_count = 0
        self.feature = self.threshold = self.left = self.right = None
        self.class_counts = self.impurity = self.prediction = None

    def _validate(self, features, labels, width=None):
        x = matrix(features, width)
        try:
            y = torch.as_tensor(labels, device="cpu").detach()
        except (TypeError, RuntimeError, ValueError) as error:
            raise ValueError("labels must be integers inside the class range") from error
        if y.shape != (len(x),):
            raise ValueError("features and labels must be nonempty and have equal lengths")
        if y.dtype not in _INTEGER_TYPES or ((y < 0) | (y >= self.class_count)).any():
            raise ValueError("labels must be integers inside the class range")
        return x, y.to(torch.int64)

    @torch.no_grad()
    def fit(self, features, labels, config=None):
        config = config or DecisionTreeConfig()
        x, y = self._validate(features, labels)
        impurity = _gini if config.criterion == "gini" else _entropy
        one_hot = torch.eye(self.class_count, dtype=torch.int64)[y]
        feature, threshold, left, right, class_counts, impurities = [], [], [], [], [], []
        # Popping left before right assigns ids in preorder without recursion.
        stack = [(torch.arange(len(x)), 0, None, None)]
        while stack:
            indices, depth, parent, links = stack.pop()
            node = len(class_counts)
            if parent is not None:
                links[parent] = node
            counts = torch.bincount(y[indices], minlength=self.class_count)
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
        self.feature = torch.tensor(feature, dtype=torch.int64)
        self.threshold = torch.tensor(threshold, dtype=torch.float64)
        self.left = torch.tensor(left, dtype=torch.int64)
        self.right = torch.tensor(right, dtype=torch.int64)
        self.class_counts = torch.stack(class_counts)
        self.impurity = torch.tensor(impurities, dtype=torch.float64)
        # argmax returns the first maximum, so count ties choose the lowest class.
        self.prediction = self.class_counts.argmax(dim=1)
        self.feature_count = x.shape[1]
        return self

    def _require_fitted(self):
        if self.left is None:
            raise RuntimeError("the model must be fitted first")

    def _route(self, x, record_path=False):
        """Advance every sample one level per step; optionally record each visited node."""
        rows = torch.arange(len(x))
        node = torch.zeros(len(x), dtype=torch.int64)
        path = torch.zeros((len(x), len(self.left)), dtype=torch.bool) if record_path else None
        while True:
            if record_path:
                path[rows, node] = True
            internal = self.left[node] >= 0
            if not internal.any():
                return node, path
            goes_left = x[rows, self.feature[node]] <= self.threshold[node]
            child = torch.where(goes_left, self.left[node], self.right[node])
            node = torch.where(internal, child, node)

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
        counts = self.class_counts[self._route(row[None, :])[0][0]].to(torch.float64)
        return counts / counts.sum()

    def accuracy(self, features, labels):
        self._require_fitted()
        x, y = self._validate(features, labels, self.feature_count)
        return int((self.prediction[self._route(x)[0]] == y).sum()) / len(y)

    def _levels(self):
        frontier = torch.tensor([0])
        while frontier.numel():
            yield frontier
            internal = frontier[self.left[frontier] >= 0]
            frontier = torch.cat((self.left[internal], self.right[internal]))

    @property
    def reachable_nodes(self):
        """Ids of the nodes in the current tree, ascending, which is preorder."""
        self._require_fitted()
        return torch.sort(torch.cat(list(self._levels()))).values

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

    @torch.no_grad()
    def prune(self, features, labels):
        """Collapse whole subtrees using validation accuracy; never pass test data here."""
        self._require_fitted()
        x, y = self._validate(features, labels, self.feature_count)
        collapsed = 0
        while True:
            levels = list(self._levels())
            # Walking levels bottom-up finishes each child's subtree size before its parent's.
            subtree = torch.ones(len(self.left), dtype=torch.int64)
            for level in reversed(levels):
                internal = level[self.left[level] >= 0]
                subtree[internal] += subtree[self.left[internal]] + subtree[self.right[internal]]
            candidates = torch.sort(torch.cat(levels)).values
            candidates = candidates[self.left[candidates] >= 0]
            if not candidates.numel():
                return collapsed
            leaves, path = self._route(x, record_path=True)
            correct = self.prediction[leaves] == y
            baseline = int(correct.sum())
            # Collapsing v changes only the samples whose path crosses v: they lose their current
            # correctness and are then right exactly when their label equals prediction[v].
            lost = (path & correct[:, None]).sum(dim=0)
            gained = (path & (self.prediction[None, :] == y[:, None])).sum(dim=0)
            score = (baseline - lost + gained)[candidates]
            removed = (subtree - 1)[candidates]
            keep = score >= baseline
            if not keep.any():
                return collapsed
            # Highest accuracy, then most removed nodes, then first in preorder.
            candidates, score, removed = candidates[keep], score[keep], removed[keep]
            best = score == score.max()
            candidates, removed = candidates[best], removed[best]
            node = int(candidates[removed == removed.max()][0])
            self.left[node] = self.right[node] = -1
            collapsed += 1
