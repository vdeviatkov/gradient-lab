"""Recover a latent plane and three groups without giving labels to either learner."""

import argparse
import math
from collections.abc import Sequence
from dataclasses import dataclass

from .kmeans import KMeans, KMeansConfig, KMeansInitialization, cluster_purity
from .model import PrincipalComponentAnalysis

SEED = 20260907


class _SplitMix64:
    """Dataset-only stream matching the C++ deterministic generator, including Box-Muller."""

    def __init__(self, seed: int):
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**64:
            raise ValueError("seed must be a uint64 integer")
        self.state = seed

    def uniform(self):
        mask = 2**64 - 1
        self.state = (self.state + 0x9E3779B97F4A7C15) & mask
        value = self.state
        value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & mask
        value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & mask
        value ^= value >> 31
        return (value >> 11) / 2**53

    def gaussian(self):
        first = max(self.uniform(), 1e-12)
        second = self.uniform()
        return math.sqrt(-2 * math.log(first)) * math.cos(2 * math.pi * second)


@dataclass(frozen=True)
class LabeledSplit:
    features: tuple[tuple[float, ...], ...]
    labels: tuple[int, ...]


@dataclass(frozen=True)
class Dataset:
    train: LabeledSplit
    validation: LabeledSplit
    test: LabeledSplit


def build_dataset(seed: int = SEED) -> Dataset:
    rng = _SplitMix64(seed)
    splits = [([], []) for _ in range(3)]
    for group, (center_x, center_y) in enumerate(((0.0, 0.0), (6.0, 0.5), (3.0, 5.0))):
        for index in range(40):
            x = center_x + 0.6 * rng.gaussian()
            y = center_y + 0.6 * rng.gaussian()
            row = (
                x + 0.05 * rng.gaussian(),
                y + 0.05 * rng.gaussian(),
                0.8 * x - 0.6 * y + 0.05 * rng.gaussian(),
                0.3 * x + 0.4 * y + 0.05 * rng.gaussian(),
            )
            features, labels = splits[0 if index < 24 else 1 if index < 32 else 2]
            features.append(row)
            labels.append(group)
    return Dataset(*(LabeledSplit(tuple(features), tuple(labels)) for features, labels in splits))


def select_cluster_count(validation_inertia: Sequence[float]) -> int:
    if len(validation_inertia) < 3 or any(
        not math.isfinite(x) or x < 0 for x in validation_inertia
    ):
        raise ValueError("elbow selection needs at least three finite non-negative inertias")
    # range index j represents k=j+1; first maximum gives the lower k on ties.
    return (
        max(
            range(1, len(validation_inertia) - 1),
            key=lambda j: (
                validation_inertia[j - 1] - 2 * validation_inertia[j] + validation_inertia[j + 1]
            ),
        )
        + 1
    )


@dataclass(frozen=True)
class ClusteringReport:
    train_inertia: float
    test_inertia: float
    test_purity: float
    iterations: int
    converged: bool


@dataclass(frozen=True)
class ExperimentResult:
    explained_variance: tuple[float, ...]
    variance_ratio: tuple[float, ...]
    reconstruction_errors: tuple[float, ...]
    validation_inertia: tuple[float, ...]
    selected_clusters: int
    baseline: ClusteringReport
    raw: ClusteringReport
    random_initialization: ClusteringReport
    projected: ClusteringReport


def run_experiment(seed: int = SEED) -> ExperimentResult:
    data = build_dataset(seed)
    spectrum = PrincipalComponentAnalysis(4)
    spectrum.fit(data.train.features)
    errors = []
    projected_train, projected_test = (), ()
    for rank in range(1, 5):
        model = PrincipalComponentAnalysis(rank)
        model.fit(data.train.features)
        errors.append(model.reconstruction_error(data.test.features))
        if rank == 2:
            projected_train = model.transform_many(data.train.features)
            projected_test = model.transform_many(data.test.features)
    config = KMeansConfig(seed=seed & 0xFFFFFFFF)
    validation_inertia = []
    for clusters in range(1, 7):
        model = KMeans(clusters)
        model.fit(data.train.features, config)
        validation_inertia.append(model.inertia(data.validation.features))
    selected = select_cluster_count(validation_inertia)

    def report(train, test, count, initialization=KMeansInitialization.KMEANS_PLUS_PLUS):
        model = KMeans(count)
        fitted = model.fit(train, KMeansConfig(seed=config.seed, initialization=initialization))
        return ClusteringReport(
            fitted.inertia,
            model.inertia(test),
            cluster_purity(model.predict_many(test), data.test.labels),
            fitted.iterations,
            fitted.converged,
        )

    return ExperimentResult(
        spectrum.explained_variance,
        spectrum.explained_variance_ratio,
        tuple(errors),
        tuple(validation_inertia),
        selected,
        report(data.train.features, data.test.features, 1),
        report(data.train.features, data.test.features, selected),
        report(
            data.train.features, data.test.features, selected, KMeansInitialization.RANDOM_SAMPLES
        ),
        report(projected_train, projected_test, selected),
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args(argv)
    result = run_experiment(args.seed)
    print(f"Pure-Python PCA and k-means experiment (seed={args.seed})")
    print("dataset: 120 samples, 4 features, 3 groups; splits train=72, validation=24, test=24")
    cumulative = 0.0
    for i, (variance, ratio) in enumerate(
        zip(result.explained_variance, result.variance_ratio, strict=True), 1
    ):
        cumulative += ratio
        print(
            f"  component {i}: explained variance={variance:.6f}, ratio={ratio:.6f}, "
            f"cumulative={cumulative:.6f}"
        )
    for rank, error in enumerate(result.reconstruction_errors, 1):
        print(f"  rank {rank}: test reconstruction MSE per feature={error:.6f}")
    for count, inertia in enumerate(result.validation_inertia, 1):
        print(f"  k={count}: validation inertia={inertia:.6f}")
    print(f"selected k={result.selected_clusters} by validation elbow")
    for label, report in (
        ("baseline", result.baseline),
        ("raw k-means++", result.raw),
        ("raw random", result.random_initialization),
        ("PCA-2 k-means++", result.projected),
    ):
        print(
            f"{label}: train inertia={report.train_inertia:.6f}, test inertia="
            f"{report.test_inertia:.6f}, test purity={report.test_purity:.6f}, "
            f"iterations={report.iterations}, converged={report.converged}"
        )
    if (
        sum(result.variance_ratio[:2]) < 0.99
        or result.selected_clusters != 3
        or result.raw.test_inertia >= result.baseline.test_inertia
        or result.raw.test_purity <= result.baseline.test_purity
        or result.projected.test_purity < result.raw.test_purity
        or not all(
            r.converged
            for r in (result.raw, result.random_initialization, result.projected, result.baseline)
        )
    ):
        raise RuntimeError("experiment failed its held-out recovery or convergence criteria")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
