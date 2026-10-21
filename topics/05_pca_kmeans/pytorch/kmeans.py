"""Lloyd iterations, explicit k-means++ sampling, restarts, and donor repair in tensors."""

import math
import random

import torch
from ml_scratch.pure.pca_kmeans.kmeans import (
    KMeansConfig,
    KMeansInitialization,
    KMeansRestartSummary,
    KMeansTrainingResult,
)

from ml_scratch.pytorch.validation import matrix, positive_integer, vector


def cluster_purity(assignments, true_labels):
    a = torch.as_tensor(assignments, device="cpu")
    y = torch.as_tensor(true_labels, device="cpu")
    if a.ndim != 1 or not a.numel() or a.shape != y.shape:
        raise ValueError("assignments and labels must be nonempty vectors of equal length")
    integer_types = (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)
    if any(v.dtype not in integer_types or (v < 0).any() for v in (a, y)):
        raise ValueError("assignments and labels must be non-negative integers")
    correct = sum(
        int(torch.unique(y[a == label], return_counts=True)[1].max()) for label in torch.unique(a)
    )
    return correct / len(a)


def _distances(x, centroids):
    difference = x[:, None, :] - centroids[None, :, :]
    result = (difference * difference).sum(dim=2)
    if not torch.isfinite(result).all():
        raise ArithmeticError("distances overflowed; rescale the features")
    return result


def _closest(x, centroids):
    distances = _distances(x, centroids)
    assignments = distances.argmin(dim=1)
    return assignments, distances[torch.arange(len(x)), assignments]


def _initialize(x, count, initialization, rng):
    if initialization is KMeansInitialization.RANDOM_SAMPLES:
        order = list(range(len(x)))
        for j in range(count):
            choice = j + int(rng.random() * (len(x) - j))
            order[j], order[choice] = order[choice], order[j]
        return x[order[:count]].clone()
    centroids = [x[int(rng.random() * len(x))].clone()]
    distances = torch.full((len(x),), math.inf, dtype=torch.float64)
    while len(centroids) < count:
        distances = torch.minimum(distances, _distances(x, centroids[-1][None, :])[:, 0])
        total = float(distances.sum())
        if not math.isfinite(total):
            raise ArithmeticError("seeding distances overflowed; rescale the features")
        choice = len(x) - 1
        if total == 0:
            choice = int(rng.random() * len(x))
        else:
            target = rng.random() * total
            for i, distance in enumerate(distances.tolist()):
                if target < distance:
                    choice = i
                    break
                target -= distance
        centroids.append(x[choice].clone())
    return torch.stack(centroids)


class KMeans:
    def __init__(self, cluster_count):
        positive_integer(cluster_count, "cluster_count")
        self.cluster_count = cluster_count
        self.centroids = self.labels = None

    @property
    def fitted(self):
        return self.centroids is not None

    def _require_fitted(self):
        if not self.fitted:
            raise ValueError("model must be fitted first")

    def predict(self, sample):
        self._require_fitted()
        row = vector(sample, self.centroids.shape[1])
        return int(_closest(row[None, :], self.centroids)[0][0])

    def predict_many(self, data):
        self._require_fitted()
        return _closest(matrix(data, self.centroids.shape[1]), self.centroids)[0]

    def inertia(self, data):
        self._require_fitted()
        total = float(_closest(matrix(data, self.centroids.shape[1]), self.centroids)[1].sum())
        if not math.isfinite(total):
            raise ArithmeticError("inertia overflowed; rescale the features")
        return total

    @torch.no_grad()
    def fit(self, data, config=KMeansConfig()):
        x = matrix(data)
        if self.cluster_count > len(x):
            raise ValueError("cluster_count must not exceed the sample count")
        positive_integer(config.max_iterations, "max_iterations")
        positive_integer(config.restarts, "restarts")
        if not isinstance(config.initialization, KMeansInitialization):
            raise ValueError("unknown initialization")
        if not math.isfinite(config.tolerance) or config.tolerance < 0:
            raise ValueError("tolerance must be finite and non-negative")
        if type(config.seed) is not int or not 0 <= config.seed < 2**32:
            raise ValueError("seed must be a uint32 integer")
        summaries, best_inertia, best_restart = [], math.inf, 0
        best_centroids = best_labels = None
        for restart in range(config.restarts):
            seed = (config.seed + restart) & 0xFFFFFFFF
            centroids = _initialize(
                x, self.cluster_count, config.initialization, random.Random(seed)
            )
            history, converged = [], False
            for iteration in range(1, config.max_iterations + 1):
                assignments, distances = _closest(x, centroids)
                counts = torch.bincount(assignments, minlength=self.cluster_count)
                sums = torch.zeros_like(centroids)
                sums.index_add_(0, assignments, x)
                for empty in torch.nonzero(counts == 0).flatten().tolist():
                    candidates = torch.nonzero(counts[assignments] > 1).flatten()
                    worst = int(candidates[torch.argmax(distances[candidates])])
                    donor = int(assignments[worst])
                    counts[donor] -= 1
                    counts[empty] = 1
                    sums[donor] -= x[worst]
                    sums[empty] = x[worst]
                    assignments[worst], distances[worst] = empty, 0
                updated = sums / counts[:, None]
                movement = float(((centroids - updated) ** 2).sum())
                centroids = updated
                inertia = float(_closest(x, centroids)[1].sum())
                if not math.isfinite(inertia):
                    raise ArithmeticError("inertia overflowed; rescale the features")
                history.append(inertia)
                if movement <= config.tolerance:
                    converged = True
                    break
            summaries.append(
                KMeansRestartSummary(seed, inertia, iteration, converged, tuple(history))
            )
            if inertia < best_inertia:
                best_inertia, best_restart = inertia, restart
                best_centroids = centroids.clone()
                best_labels = _closest(x, centroids)[0].clone()
        self.centroids, self.labels = best_centroids, best_labels
        best = summaries[best_restart]
        return KMeansTrainingResult(
            best.inertia, best.iterations, best.converged, best_restart, tuple(summaries)
        )
