import numpy as np
import pytest
from ml_scratch.numpy.pca_kmeans import (
    KMeans,
    KMeansConfig,
    KMeansInitialization,
    PrincipalComponentAnalysis,
    cluster_purity,
    covariance_matrix,
    jacobi_eigen_decomposition,
)
from ml_scratch.numpy.pca_kmeans.experiment import run_experiment
from ml_scratch.pure.pca_kmeans import KMeans as ScalarKMeans
from ml_scratch.pure.pca_kmeans import PrincipalComponentAnalysis as ScalarPCA
from ml_scratch.pure.pca_kmeans.experiment import build_dataset
from ml_scratch.pure.pca_kmeans.experiment import run_experiment as scalar_experiment


def test_jacobi_residual_orthogonality_and_independent_eigenvalues():
    rng = np.random.default_rng(41)
    values = rng.normal(size=(5, 5))
    a = values.T @ values
    decomposition = jacobi_eigen_decomposition(a)
    w, v = decomposition.eigenvalues, decomposition.eigenvectors
    np.testing.assert_allclose(a @ v.T, v.T * w, atol=1e-10)
    np.testing.assert_allclose(v @ v.T, np.eye(5), atol=1e-12)
    np.testing.assert_allclose(w, np.linalg.eigvalsh(a)[::-1], atol=1e-10)
    for row in v:
        assert row[np.argmax(abs(row))] >= 0
    with pytest.raises(RuntimeError):
        jacobi_eigen_decomposition(a, max_sweeps=1, tolerance=1e-15)


def test_covariance_pca_scalar_and_full_reconstruction():
    x = np.asarray(build_dataset().train.features)
    np.testing.assert_allclose(covariance_matrix(x), np.cov(x, rowvar=False), atol=1e-13)
    model, scalar = PrincipalComponentAnalysis(2), ScalarPCA(2)
    model.fit(x)
    scalar.fit(x.tolist())
    np.testing.assert_allclose(model.explained_variance, scalar.explained_variance, atol=1e-12)
    np.testing.assert_allclose(model.components, scalar.components, atol=1e-11)
    np.testing.assert_allclose(
        model.transform_many(x), scalar.transform_many(x.tolist()), atol=1e-11
    )
    assert model.reconstruction_error(x) == pytest.approx(scalar.reconstruction_error(x.tolist()))
    full = PrincipalComponentAnalysis(4)
    full.fit(x)
    np.testing.assert_allclose(full.inverse_transform(full.transform(x[0])), x[0], atol=1e-12)
    assert full.reconstruction_error(x) < 1e-25
    constant = PrincipalComponentAnalysis(2)
    constant.fit([[1, 2], [1, 2]])
    np.testing.assert_array_equal(constant.explained_variance_ratio, [0, 0])


@pytest.mark.parametrize("initialization", list(KMeansInitialization))
def test_lloyd_scalar_restarts_history_and_reproducibility(initialization):
    x = build_dataset().train.features
    config = KMeansConfig(initialization=initialization, seed=17, restarts=4)
    model, scalar = KMeans(3), ScalarKMeans(3)
    result, reference = model.fit(x, config), scalar.fit(x, config)
    np.testing.assert_allclose(model.centroids, scalar.centroids, atol=1e-12)
    np.testing.assert_array_equal(model.labels, scalar.labels)
    assert result.inertia == pytest.approx(reference.inertia)
    assert result.converged
    for restart in result.restarts:
        assert all(
            b <= a + 1e-10
            for a, b in zip(restart.inertia_per_iteration, restart.inertia_per_iteration[1:])
        )
    assert result.inertia == min(r.inertia for r in result.restarts)
    assert model.fit(x, config) == result
    assert [r.seed for r in result.restarts] == [17, 18, 19, 20]


def test_empty_clusters_duplicates_ties_and_exhausted_budget():
    model = KMeans(3)
    trained = model.fit([[0, 0]] * 5, KMeansConfig(restarts=1))
    assert trained.converged and trained.inertia == 0
    assert model.centroids.shape == (3, 2)
    assert model.predict([0, 0]) == 0
    np.testing.assert_array_equal(model.labels, model.predict_many([[0, 0]] * 5))
    trained = KMeans(3).fit(
        build_dataset().train.features, KMeansConfig(max_iterations=1, restarts=1)
    )
    assert not trained.converged and trained.iterations == 1
    model = KMeans(1)
    model.fit([[0], [2]], KMeansConfig(restarts=1))
    assert model.inertia([[0], [2]]) == pytest.approx(2)
    assert cluster_purity(np.array([0, 0, 1]), np.array([1, 1, 0])) == 1


@pytest.mark.parametrize("bad", [[], [[1, 2], [1]], [[np.nan, 1]], [[np.inf, 0]]])
def test_invalid_matrices(bad):
    for model in (KMeans(1), PrincipalComponentAnalysis(1)):
        with pytest.raises(ValueError):
            model.fit(bad)


def test_invalid_controls_unfitted_and_failed_fit_preserves_state():
    with pytest.raises(ValueError):
        KMeans(1).predict([0])
    with pytest.raises(ValueError):
        PrincipalComponentAnalysis(1).transform([0])
    for matrix in ([[1, 2]], [[1, 2], [0, 1]]):
        with pytest.raises(ValueError):
            jacobi_eigen_decomposition(matrix)
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
    model = PrincipalComponentAnalysis(1)
    model.fit([[0], [1]])
    mean = model.mean.copy()
    with pytest.raises(ValueError):
        model.fit([[1]])
    np.testing.assert_array_equal(model.mean, mean)
    with pytest.raises(ValueError):
        model.transform([0, 1])
    with pytest.raises(ArithmeticError):
        covariance_matrix([[1e308], [-1e308]])


def test_numpy_pca_kmeans_experiment_matches_scalar():
    result, scalar = run_experiment(), scalar_experiment()
    assert result.selected_clusters == scalar.selected_clusters == 3
    np.testing.assert_allclose(result.variance_ratio, scalar.variance_ratio, atol=1e-12)
    np.testing.assert_allclose(
        result.reconstruction_errors, scalar.reconstruction_errors, atol=1e-12
    )
    np.testing.assert_allclose(result.validation_inertia, scalar.validation_inertia, atol=1e-10)
    assert result.raw.test_purity == result.projected.test_purity == 1
    assert result.raw.test_inertia == pytest.approx(scalar.raw.test_inertia)
