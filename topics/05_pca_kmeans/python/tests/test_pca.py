import math

import pytest
from ml_scratch.pure.pca_kmeans import (
    PrincipalComponentAnalysis,
    covariance_matrix,
    jacobi_eigen_decomposition,
)

LINE = ((-2.0, -4.0), (-1.0, -2.0), (0.0, 0.0), (1.0, 2.0), (2.0, 4.0))


def test_covariance_is_unbiased_centered_and_symmetric():
    covariance = covariance_matrix(((1.0, 2.0), (2.0, 4.0), (3.0, 6.0)))
    assert covariance[0] == pytest.approx((1.0, 2.0))
    assert covariance[1] == pytest.approx((2.0, 4.0))
    shifted = covariance_matrix(((11.0, -18.0), (12.0, -16.0), (13.0, -14.0)))
    assert shifted == covariance


@pytest.mark.parametrize(
    "matrix", [((2.0, 1.0), (1.0, 2.0)), ((4.0, 1.0, -2.0), (1.0, 3.0, 0.5), (-2.0, 0.5, 5.0))]
)
def test_jacobi_eigenpairs_orthonormality_trace_and_reconstruction(matrix):
    result = jacobi_eigen_decomposition(matrix)
    assert list(result.eigenvalues) == sorted(result.eigenvalues, reverse=True)
    assert sum(result.eigenvalues) == pytest.approx(sum(matrix[j][j] for j in range(len(matrix))))
    for i, (value, direction) in enumerate(
        zip(result.eigenvalues, result.eigenvectors, strict=True)
    ):
        actual = [sum(a * v for a, v in zip(row, direction, strict=True)) for row in matrix]
        assert actual == pytest.approx([value * v for v in direction], abs=1e-10)
        assert direction[max(range(len(direction)), key=lambda j: abs(direction[j]))] >= 0
        for j, other in enumerate(result.eigenvectors):
            assert sum(x * y for x, y in zip(direction, other, strict=True)) == pytest.approx(
                float(i == j), abs=1e-12
            )
    for j, row in enumerate(matrix):
        reconstructed = [
            sum(
                value * v[j] * v[k]
                for value, v in zip(result.eigenvalues, result.eigenvectors, strict=True)
            )
            for k in range(len(matrix))
        ]
        assert reconstructed == pytest.approx(row, abs=1e-10)


def test_known_eigenvalues_and_ties_have_deterministic_order():
    result = jacobi_eigen_decomposition(((2.0, 1.0), (1.0, 2.0)))
    assert result.eigenvalues == pytest.approx((3.0, 1.0))
    tied = jacobi_eigen_decomposition(((2.0, 0.0, 0.0), (0.0, 2.0, 0.0), (0.0, 0.0, -1.0)))
    assert tied.eigenvectors == ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    assert jacobi_eigen_decomposition(((4.0,),)).eigenvalues == (4.0,)


def test_pca_recovers_the_known_line_and_held_out_reconstruction():
    model = PrincipalComponentAnalysis(1)
    model.fit(LINE)
    assert model.explained_variance == pytest.approx((12.5,))
    assert model.explained_variance_ratio == pytest.approx((1.0,))
    assert model.components[0] == pytest.approx((1 / math.sqrt(5), 2 / math.sqrt(5)))
    assert model.inverse_transform(model.transform((3.0, 6.0))) == pytest.approx((3.0, 6.0))
    assert model.reconstruction_error(((3.0, 6.0), (-3.0, -6.0))) < 1e-25
    mean = model.mean
    model.transform_many(((100.0, 200.0),))
    assert model.mean == mean  # Held-out data must not refit centering.


def test_projection_decorrelates_and_residual_matches_discarded_variance():
    data = ((1.0, 0.2), (2.0, 1.9), (3.0, 3.4), (4.0, 3.8), (5.0, 5.6))
    full = PrincipalComponentAnalysis(2)
    full.fit(data)
    projected = full.transform_many(data)
    covariance = covariance_matrix(projected)
    assert covariance[0][1] == pytest.approx(0.0, abs=1e-12)
    assert covariance[0][0] == pytest.approx(full.explained_variance[0])
    assert covariance[1][1] == pytest.approx(full.explained_variance[1])
    assert full.reconstruction_error(data) < 1e-25
    reduced = PrincipalComponentAnalysis(1)
    reduced.fit(data)
    # Sample covariance divides by n-1; reconstruction MSE divides by n*d.
    expected = full.explained_variance[1] * (len(data) - 1) / (len(data) * 2)
    assert reduced.reconstruction_error(data) == pytest.approx(expected, abs=1e-12)


def test_constant_features_have_zero_variance_without_dividing_by_zero():
    model = PrincipalComponentAnalysis(2)
    model.fit(((7.0, -3.0),) * 4)
    assert model.total_variance == 0
    assert model.explained_variance_ratio == (0.0, 0.0)
    assert model.reconstruction_error(((7.0, -3.0),)) == 0


def test_refits_are_deterministic_and_failed_fit_preserves_state():
    model = PrincipalComponentAnalysis(1)
    model.fit(LINE)
    state = (model.mean, model.components, model.explained_variance)
    model.fit(LINE)
    assert (model.mean, model.components, model.explained_variance) == state
    with pytest.raises(ValueError):
        model.fit(((1.0, 2.0),))
    assert (model.mean, model.components, model.explained_variance) == state


@pytest.mark.parametrize(
    "data",
    [(), ((),), ((1.0,),), ((1.0,), (1.0, 2.0)), ((math.nan,), (1.0,)), ((math.inf,), (1.0,))],
)
def test_covariance_rejects_invalid_data(data):
    with pytest.raises(ValueError):
        covariance_matrix(data)


@pytest.mark.parametrize(
    "kwargs", [{"max_sweeps": 0}, {"max_sweeps": 1.5}, {"tolerance": 0}, {"tolerance": math.nan}]
)
def test_invalid_eigensolver_config(kwargs):
    with pytest.raises(ValueError):
        jacobi_eigen_decomposition(((1.0, 0.0), (0.0, 1.0)), **kwargs)


def test_bad_matrix_and_nonconvergence_are_not_silent_success():
    with pytest.raises(ValueError, match="symmetric"):
        jacobi_eigen_decomposition(((1.0, 2.0), (3.0, 4.0)))
    with pytest.raises(ValueError, match="square"):
        jacobi_eigen_decomposition(((1.0, 2.0),))
    with pytest.raises(RuntimeError, match="converge"):
        jacobi_eigen_decomposition(
            ((4.0, 1.0, -2.0), (1.0, 3.0, 0.5), (-2.0, 0.5, 5.0)), max_sweeps=1
        )


@pytest.mark.parametrize("count", [0, -1, 1.5, True])
def test_invalid_component_count(count):
    with pytest.raises(ValueError):
        PrincipalComponentAnalysis(count)


def test_projection_validation_and_covariance_overflow():
    model = PrincipalComponentAnalysis(1)
    for call in (lambda: model.transform((0.0, 0.0)), lambda: model.inverse_transform((0.0,))):
        with pytest.raises(ValueError, match="fitted"):
            call()
    model.fit(LINE)
    for call in (
        lambda: model.transform((1.0,)),
        lambda: model.inverse_transform((1.0, 2.0)),
        lambda: model.transform((math.nan, 0.0)),
        lambda: PrincipalComponentAnalysis(3).fit(LINE),
    ):
        with pytest.raises(ValueError):
            call()
    with pytest.raises(ArithmeticError, match="overflow"):
        covariance_matrix(((-1e200,), (1e200,)))


def test_eigen_rotation_overflow_is_not_a_successful_decomposition():
    with pytest.raises(ArithmeticError, match="overflow"):
        jacobi_eigen_decomposition(((1e308, 1e308), (1e308, 1e308)))
