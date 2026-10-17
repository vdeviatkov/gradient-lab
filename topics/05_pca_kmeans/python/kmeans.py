"""Lloyd's algorithm with explicit seeding, independent restarts, and empty-cluster repair."""

import math
import random
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum

from .validation import Matrix, feature_matrix, positive_integer, squared_distance, vector


class KMeansInitialization(Enum):
    RANDOM_SAMPLES = "random_samples"
    KMEANS_PLUS_PLUS = "kmeans_plus_plus"


@dataclass(frozen=True)
class KMeansConfig:
    initialization: KMeansInitialization = KMeansInitialization.KMEANS_PLUS_PLUS
    max_iterations: int = 300
    restarts: int = 10
    tolerance: float = 1e-12
    seed: int = 0


@dataclass(frozen=True)
class KMeansRestartSummary:
    seed: int
    inertia: float
    iterations: int
    converged: bool
    inertia_per_iteration: tuple[float, ...]


@dataclass(frozen=True)
class KMeansTrainingResult:
    inertia: float
    iterations: int
    converged: bool
    best_restart: int
    restarts: tuple[KMeansRestartSummary, ...]


def cluster_purity(assignments: Sequence[int], true_labels: Sequence[int]) -> float:
    """Evaluation only; use at fixed k because increasing k can inflate purity."""
    if not assignments or len(assignments) != len(true_labels):
        raise ValueError("assignments must not be empty and sizes must match")
    if any(isinstance(v, bool) or not isinstance(v, int) or v < 0
           for v in (*assignments, *true_labels)):
        raise ValueError("assignments and labels must be non-negative integers")
    counts = defaultdict(Counter)
    for assignment, label in zip(assignments, true_labels, strict=True):
        counts[assignment][label] += 1
    return sum(max(group.values()) for group in counts.values()) / len(assignments)


def _closest(sample, centroids):
    distances = [squared_distance(sample, centroid) for centroid in centroids]
    label = min(range(len(centroids)), key=distances.__getitem__)
    return label, distances[label]


def _initialize(data, count, initialization, rng):
    if initialization is KMeansInitialization.RANDOM_SAMPLES:
        order = list(range(len(data)))
        for j in range(count):
            choice = j + int(rng.random() * (len(data) - j))
            order[j], order[choice] = order[choice], order[j]
        return [list(data[index]) for index in order[:count]]
    centroids = [list(data[int(rng.random() * len(data))])]
    distances = [math.inf] * len(data)
    while len(centroids) < count:
        distances = [min(distance, squared_distance(row, centroids[-1]))
                     for row, distance in zip(data, distances, strict=True)]
        total = sum(distances)
        if not math.isfinite(total):
            raise ArithmeticError("seeding distances overflowed; rescale the features")
        choice = len(data) - 1
        if total == 0:
            choice = int(rng.random() * len(data))
        else:
            target = rng.random() * total
            for index, distance in enumerate(distances):
                if target < distance:
                    choice = index
                    break
                target -= distance
        centroids.append(list(data[choice]))
    return centroids


@dataclass
class KMeans:
    cluster_count: int
    centroids: tuple[tuple[float, ...], ...] = field(init=False, default=())
    labels: tuple[int, ...] = field(init=False, default=())

    def __post_init__(self):
        positive_integer(self.cluster_count, "cluster_count")

    @property
    def fitted(self):
        return bool(self.centroids)

    def _require_fitted(self):
        if not self.fitted:
            raise ValueError("model must be fitted first")

    def predict(self, sample: Sequence[float]) -> int:
        self._require_fitted()
        return _closest(vector(sample, len(self.centroids[0])), self.centroids)[0]

    def predict_many(self, data: Matrix) -> tuple[int, ...]:
        self._require_fitted()
        return tuple(self.predict(row) for row in feature_matrix(data))

    def inertia(self, data: Matrix) -> float:
        self._require_fitted()
        rows = feature_matrix(data)
        total = sum(_closest(vector(row, len(self.centroids[0])), self.centroids)[1] for row in rows)
        if not math.isfinite(total):
            raise ArithmeticError("inertia overflowed; rescale the features")
        return total

    def fit(self, data: Matrix, config: KMeansConfig = KMeansConfig()) -> KMeansTrainingResult:
        rows = feature_matrix(data)
        if self.cluster_count > len(rows):
            raise ValueError("cluster_count must not exceed the sample count")
        positive_integer(config.max_iterations, "max_iterations")
        positive_integer(config.restarts, "restarts")
        if not isinstance(config.initialization, KMeansInitialization):
            raise ValueError("unknown initialization")
        if not math.isfinite(config.tolerance) or config.tolerance < 0:
            raise ValueError("tolerance must be finite and non-negative")
        if isinstance(config.seed, bool) or not isinstance(config.seed, int) or not 0 <= config.seed < 2**32:
            raise ValueError("seed must be a uint32 integer")
        width = len(rows[0])
        summaries = []
        best_inertia, best_centroids, best_labels, best_restart = math.inf, (), (), 0
        for restart in range(config.restarts):
            seed = (config.seed + restart) & 0xffffffff
            centroids = _initialize(rows, self.cluster_count, config.initialization, random.Random(seed))
            history = []
            converged = False
            for iteration in range(1, config.max_iterations + 1):
                assignments, distances = zip(*(_closest(row, centroids) for row in rows), strict=True)
                assignments, distances = list(assignments), list(distances)
                sums = [[0.0] * width for _ in centroids]
                counts = [0] * self.cluster_count
                for row, label in zip(rows, assignments, strict=True):
                    counts[label] += 1
                    for j, value in enumerate(row):
                        sums[label][j] += value
                for empty in range(self.cluster_count):
                    if counts[empty]:
                        continue
                    candidates = [i for i, label in enumerate(assignments) if counts[label] > 1]
                    worst = max(candidates, key=distances.__getitem__)
                    donor = assignments[worst]
                    counts[donor] -= 1
                    counts[empty] = 1
                    for j, value in enumerate(rows[worst]):
                        sums[donor][j] -= value
                        sums[empty][j] = value
                    assignments[worst] = empty
                    distances[worst] = 0.0
                updated = [[value / count for value in row] for row, count in
                           zip(sums, counts, strict=True)]
                movement = sum(squared_distance(old, new) for old, new in
                               zip(centroids, updated, strict=True))
                centroids = updated
                inertia = sum(_closest(row, centroids)[1] for row in rows)
                if not math.isfinite(inertia):
                    raise ArithmeticError("inertia overflowed; rescale the features")
                history.append(inertia)
                if movement <= config.tolerance:
                    converged = True
                    break
            summaries.append(KMeansRestartSummary(seed, inertia, iteration, converged, tuple(history)))
            if inertia < best_inertia:
                best_inertia, best_restart = inertia, restart
                best_centroids = tuple(tuple(row) for row in centroids)
                best_labels = tuple(_closest(row, centroids)[0] for row in rows)
        self.centroids, self.labels = best_centroids, best_labels
        best = summaries[best_restart]
        return KMeansTrainingResult(best.inertia, best.iterations, best.converged,
                                    best_restart, tuple(summaries))
