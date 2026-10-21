"""Recover the seeded latent plane and clusters, selecting k on validation data."""

import argparse

from ml_scratch.pure.pca_kmeans.experiment import (
    SEED,
    ClusteringReport,
    ExperimentResult,
    build_dataset,
    select_cluster_count,
)

from .kmeans import KMeans, KMeansConfig, KMeansInitialization, cluster_purity
from .model import PrincipalComponentAnalysis


def run_experiment(seed=SEED):
    data = build_dataset(seed)
    spectrum = PrincipalComponentAnalysis(4)
    spectrum.fit(data.train.features)
    errors = []
    for rank in range(1, 5):
        pca = PrincipalComponentAnalysis(rank)
        pca.fit(data.train.features)
        errors.append(pca.reconstruction_error(data.test.features))
        if rank == 2:
            projected_train = pca.transform_many(data.train.features)
            projected_test = pca.transform_many(data.test.features)
    config = KMeansConfig(seed=seed & 0xFFFFFFFF)
    inertias = []
    for k in range(1, 7):
        model = KMeans(k)
        model.fit(data.train.features, config)
        inertias.append(model.inertia(data.validation.features))
    selected = select_cluster_count(inertias)

    def report(train, test, count, initialization=KMeansInitialization.KMEANS_PLUS_PLUS):
        model = KMeans(count)
        trained = model.fit(train, KMeansConfig(seed=config.seed, initialization=initialization))
        return ClusteringReport(
            trained.inertia,
            model.inertia(test),
            float(cluster_purity(model.predict_many(test), data.test.labels)),
            trained.iterations,
            trained.converged,
        )

    result = ExperimentResult(
        tuple(spectrum.explained_variance),
        tuple(spectrum.explained_variance_ratio),
        tuple(errors),
        tuple(inertias),
        selected,
        report(data.train.features, data.test.features, 1),
        report(data.train.features, data.test.features, selected),
        report(
            data.train.features, data.test.features, selected, KMeansInitialization.RANDOM_SAMPLES
        ),
        report(projected_train, projected_test, selected),
    )
    if (
        sum(result.variance_ratio[:2]) < 0.99
        or selected != 3
        or result.raw.test_inertia >= result.baseline.test_inertia
        or result.raw.test_purity <= result.baseline.test_purity
        or result.projected.test_purity < result.raw.test_purity
        or not all(
            r.converged
            for r in (result.baseline, result.raw, result.random_initialization, result.projected)
        )
    ):
        raise RuntimeError("PCA/k-means experiment failed its recovery or convergence criteria")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    print(f"NumPy PCA/k-means experiment (seed={args.seed})")
    result = run_experiment(args.seed)
    print("variance ratios:", ", ".join(f"{v:.6f}" for v in result.variance_ratio))
    print("test reconstruction MSEs:", ", ".join(f"{v:.6f}" for v in result.reconstruction_errors))
    print("validation inertias:", ", ".join(f"{v:.6f}" for v in result.validation_inertia))
    print(f"validation-selected k={result.selected_clusters}")
    for name in ("baseline", "raw", "random_initialization", "projected"):
        print(f"{name}: {getattr(result, name)}")


if __name__ == "__main__":
    main()
