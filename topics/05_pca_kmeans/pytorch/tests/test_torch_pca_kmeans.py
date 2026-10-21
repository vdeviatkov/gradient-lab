import pytest
import torch
from ml_scratch.pure.pca_kmeans import KMeans as ScalarKMeans
from ml_scratch.pure.pca_kmeans import PrincipalComponentAnalysis as ScalarPCA
from ml_scratch.pure.pca_kmeans.experiment import build_dataset
from ml_scratch.pure.pca_kmeans.experiment import run_experiment as scalar_experiment
from ml_scratch.pytorch.pca_kmeans import (
    KMeans,
    KMeansConfig,
    KMeansInitialization,
    PrincipalComponentAnalysis,
    cluster_purity,
    covariance_matrix,
    jacobi_eigen_decomposition,
)
from ml_scratch.pytorch.pca_kmeans.experiment import run_experiment


def test_jacobi_eigen_residual_orthogonality_and_independent_oracle():
    generator = torch.Generator().manual_seed(41)
    values = torch.randn((5, 5), generator=generator, dtype=torch.float64)
    a = values.T @ values
    decomposition = jacobi_eigen_decomposition(a)
    w, v = decomposition.eigenvalues, decomposition.eigenvectors
    torch.testing.assert_close(a @ v.T, v.T * w, atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(v @ v.T, torch.eye(5, dtype=torch.float64), atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(w, torch.linalg.eigvalsh(a).flip(0), atol=1e-10, rtol=1e-10)
    for row in v:
        assert row[torch.argmax(abs(row))] >= 0
    with pytest.raises(RuntimeError):
        jacobi_eigen_decomposition(a, max_sweeps=1, tolerance=1e-15)


def test_pca_covariance_scalar_and_reconstruction():
    data = build_dataset().train.features
    x = torch.tensor(data, dtype=torch.float64)
    torch.testing.assert_close(covariance_matrix(x), torch.cov(x.T))
    model, scalar = PrincipalComponentAnalysis(2), ScalarPCA(2)
    model.fit(x)
    scalar.fit(data)
    torch.testing.assert_close(
        model.explained_variance, torch.tensor(scalar.explained_variance, dtype=torch.float64)
    )
    torch.testing.assert_close(
        model.components, torch.tensor(scalar.components, dtype=torch.float64)
    )
    torch.testing.assert_close(
        model.transform_many(x), torch.tensor(scalar.transform_many(data), dtype=torch.float64)
    )
    assert model.reconstruction_error(x) == pytest.approx(scalar.reconstruction_error(data))
    full = PrincipalComponentAnalysis(4)
    full.fit(x)
    torch.testing.assert_close(full.inverse_transform(full.transform(x[0])), x[0])
    assert full.reconstruction_error(x) < 1e-25
    full.fit([[1, 2, 3, 4], [1, 2, 3, 4]])
    assert torch.equal(full.explained_variance_ratio, torch.zeros(4, dtype=torch.float64))


@pytest.mark.parametrize("initialization", list(KMeansInitialization))
def test_lloyd_scalar_history_restarts_and_repeatability(initialization):
    data = build_dataset().train.features
    config = KMeansConfig(initialization=initialization, seed=17, restarts=4)
    model, scalar = KMeans(3), ScalarKMeans(3)
    result, reference = model.fit(data, config), scalar.fit(data, config)
    torch.testing.assert_close(model.centroids, torch.tensor(scalar.centroids, dtype=torch.float64))
    assert model.labels.tolist() == list(scalar.labels)
    assert result.inertia == pytest.approx(reference.inertia)
    assert result.converged
    for restart in result.restarts:
        assert all(
            b <= a + 1e-10
            for a, b in zip(restart.inertia_per_iteration, restart.inertia_per_iteration[1:])
        )
    assert result.inertia == min(r.inertia for r in result.restarts)
    assert model.fit(data, config) == result
    assert [r.seed for r in result.restarts] == [17, 18, 19, 20]


def test_empty_clusters_duplicates_ties_budget_and_refit():
    model = KMeans(3)
    trained = model.fit([[0, 0]] * 5, KMeansConfig(restarts=1))
    assert trained.converged and trained.inertia == 0
    assert model.centroids.shape == (3, 2) and model.predict([0, 0]) == 0
    assert torch.equal(model.labels, model.predict_many([[0, 0]] * 5))
    trained = KMeans(3).fit(
        build_dataset().train.features, KMeansConfig(max_iterations=1, restarts=1)
    )
    assert not trained.converged and trained.iterations == 1
    model = KMeans(1)
    model.fit([[0], [2]], KMeansConfig(restarts=1))
    assert model.inertia([[0], [2]]) == pytest.approx(2)
    assert cluster_purity(torch.tensor([0, 0, 1]), torch.tensor([1, 1, 0])) == 1
    with pytest.raises(ValueError):
        cluster_purity([0.0], [1])


@pytest.mark.parametrize("bad", [[], [[1, 2], [1]], [[float("nan"), 1]], [[float("inf"), 0]]])
def test_invalid_data(bad):
    for model in (KMeans(1), PrincipalComponentAnalysis(1)):
        with pytest.raises(ValueError):
            model.fit(bad)


def test_invalid_controls_unfitted_and_overflow():
    with pytest.raises(ValueError):
        KMeans(1).predict([0])
    with pytest.raises(ValueError):
        PrincipalComponentAnalysis(1).transform([0])
    for values in ([[1, 2]], [[1, 2], [0, 1]]):
        with pytest.raises(ValueError):
            jacobi_eigen_decomposition(values)
    for config in (
        KMeansConfig(restarts=0),
        KMeansConfig(tolerance=-1),
        KMeansConfig(seed=-1),
        KMeansConfig(initialization="unknown"),
    ):
        with pytest.raises(ValueError):
            KMeans(1).fit([[0], [1]], config)
    with pytest.raises(ValueError):
        KMeans(3).fit([[0], [1]])
    with pytest.raises(ValueError):
        PrincipalComponentAnalysis(2).fit([[0], [1]])
    with pytest.raises(ArithmeticError):
        covariance_matrix([[1e308], [-1e308]])
    with pytest.raises(ArithmeticError):
        KMeans(1).fit([[1e308], [-1e308]])


def test_experiment_matches_scalar():
    result, reference = run_experiment(), scalar_experiment()
    assert result.selected_clusters == reference.selected_clusters == 3
    torch.testing.assert_close(
        torch.tensor(result.variance_ratio), torch.tensor(reference.variance_ratio)
    )
    torch.testing.assert_close(
        torch.tensor(result.reconstruction_errors), torch.tensor(reference.reconstruction_errors)
    )
    torch.testing.assert_close(
        torch.tensor(result.validation_inertia), torch.tensor(reference.validation_inertia)
    )
    assert result.raw.test_purity == result.projected.test_purity == 1
