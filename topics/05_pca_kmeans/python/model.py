"""PCA with explicit covariance and cyclic Jacobi rotations; no linear-algebra dependency."""

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

from .validation import Matrix, feature_matrix, positive_integer, squared_distance, vector


@dataclass(frozen=True)
class SymmetricEigenDecomposition:
    eigenvalues: tuple[float, ...]
    # Each row is an eigenvector, ordered with its eigenvalue.
    eigenvectors: tuple[tuple[float, ...], ...]


def covariance_matrix(data: Matrix) -> tuple[tuple[float, ...], ...]:
    rows = feature_matrix(data)
    if len(rows) < 2:
        raise ValueError("covariance requires at least two samples")
    width = len(rows[0])
    means = [sum(row[j] for row in rows) / len(rows) for j in range(width)]
    covariance = [[0.0] * width for _ in range(width)]
    for row in rows:
        for j in range(width):
            for k in range(j, width):
                covariance[j][k] += (row[j] - means[j]) * (row[k] - means[k])
    for j in range(width):
        for k in range(j, width):
            covariance[j][k] /= len(rows) - 1
            covariance[k][j] = covariance[j][k]
    if any(not math.isfinite(value) for row in covariance for value in row):
        raise ArithmeticError("covariance overflowed; rescale the features")
    return tuple(tuple(row) for row in covariance)


def jacobi_eigen_decomposition(
    matrix: Matrix, *, max_sweeps: int = 100, tolerance: float = 1e-12
) -> SymmetricEigenDecomposition:
    """Diagonalize a symmetric matrix; fail explicitly if the sweep budget is exhausted.

    Each rotation is applied to columns and rows, implementing J^T A J directly.
    Columns of the accumulated rotation matrix are the resulting eigenvectors.
    """
    data = feature_matrix(matrix)
    size = len(data)
    if len(data[0]) != size:
        raise ValueError("matrix must be square")
    positive_integer(max_sweeps, "max_sweeps")
    if not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be finite and positive")
    for j in range(size):
        for k in range(j + 1, size):
            if abs(data[j][k] - data[k][j]) > 1e-9 * max(1, abs(data[j][k]), abs(data[k][j])):
                raise ValueError("matrix must be symmetric")
    # Within the symmetry tolerance, use the symmetric part rather than a nonsymmetric rotation.
    working = [[0.5 * data[j][k] + 0.5 * data[k][j] for k in range(size)] for j in range(size)]
    rotations = [[float(j == k) for k in range(size)] for j in range(size)]
    threshold = tolerance * max(1.0, max(abs(data[j][j]) for j in range(size)))
    # If every pair is below this threshold, the full off-diagonal Frobenius norm is too.
    pair_threshold = threshold / max(1.0, math.sqrt(size * (size - 1)))

    def off_diagonal_norm():
        norm = math.hypot(*(working[j][k] for j in range(size) for k in range(size) if j != k))
        if not math.isfinite(norm):
            raise ArithmeticError("eigendecomposition overflowed; rescale the matrix")
        return norm

    for _ in range(max_sweeps):
        if off_diagonal_norm() <= threshold:
            break
        for j in range(size):
            for k in range(j + 1, size):
                if abs(working[j][k]) <= pair_threshold:
                    continue
                theta = (working[k][k] - working[j][j]) / (2 * working[j][k])
                t = (1.0 if theta >= 0.0 else -1.0) / (abs(theta) + math.hypot(theta, 1.0))
                cosine = 1.0 / math.sqrt(1 + t * t)
                sine = t * cosine
                for row in working:
                    first, second = row[j], row[k]
                    row[j], row[k] = cosine * first - sine * second, sine * first + cosine * second
                for index in range(size):
                    first, second = working[j][index], working[k][index]
                    working[j][index] = cosine * first - sine * second
                    working[k][index] = sine * first + cosine * second
                    first, second = rotations[index][j], rotations[index][k]
                    rotations[index][j] = cosine * first - sine * second
                    rotations[index][k] = sine * first + cosine * second
    if off_diagonal_norm() > threshold:
        raise RuntimeError("Jacobi eigensolver did not converge within max_sweeps")
    if any(not math.isfinite(value) for row in working for value in row):
        raise ArithmeticError("eigendecomposition overflowed; rescale the matrix")
    order = sorted(range(size), key=lambda j: (-working[j][j], j))
    eigenvectors = []
    for j in order:
        direction = tuple(row[j] for row in rotations)
        dominant = max(range(size), key=lambda k: abs(direction[k]))
        if direction[dominant] < 0:
            direction = tuple(-value for value in direction)
        eigenvectors.append(direction)
    return SymmetricEigenDecomposition(tuple(working[j][j] for j in order), tuple(eigenvectors))


@dataclass
class PrincipalComponentAnalysis:
    component_count: int
    mean: tuple[float, ...] = field(init=False, default=())
    components: tuple[tuple[float, ...], ...] = field(init=False, default=())
    explained_variance: tuple[float, ...] = field(init=False, default=())
    explained_variance_ratio: tuple[float, ...] = field(init=False, default=())
    total_variance: float = field(init=False, default=0.0)

    def __post_init__(self):
        positive_integer(self.component_count, "component_count")

    @property
    def fitted(self) -> bool:
        return bool(self.mean)

    def _require_fitted(self):
        if not self.fitted:
            raise ValueError("model must be fitted first")

    def fit(self, data: Matrix) -> None:
        rows = feature_matrix(data)
        if self.component_count > len(rows[0]):
            raise ValueError("component_count must not exceed the feature count")
        decomposition = jacobi_eigen_decomposition(covariance_matrix(rows))
        variances = tuple(max(0.0, value) for value in decomposition.eigenvalues)
        total = sum(variances)
        self.mean = tuple(sum(row[j] for row in rows) / len(rows) for j in range(len(rows[0])))
        self.components = decomposition.eigenvectors[: self.component_count]
        self.explained_variance = variances[: self.component_count]
        self.total_variance = total
        self.explained_variance_ratio = tuple(
            value / total if total else 0.0 for value in self.explained_variance
        )

    def transform(self, sample: Sequence[float]) -> tuple[float, ...]:
        self._require_fitted()
        values = vector(sample, len(self.mean))
        return tuple(
            sum(w * (x - m) for w, x, m in zip(component, values, self.mean, strict=True))
            for component in self.components
        )

    def transform_many(self, data: Matrix) -> tuple[tuple[float, ...], ...]:
        self._require_fitted()
        return tuple(self.transform(row) for row in feature_matrix(data))

    def inverse_transform(self, projection: Sequence[float]) -> tuple[float, ...]:
        self._require_fitted()
        values = vector(projection, self.component_count)
        return tuple(
            mean
            + sum(z * component[j] for z, component in zip(values, self.components, strict=True))
            for j, mean in enumerate(self.mean)
        )

    def reconstruction_error(self, data: Matrix) -> float:
        self._require_fitted()
        rows = feature_matrix(data)
        return sum(
            squared_distance(row, self.inverse_transform(self.transform(row))) for row in rows
        ) / (len(rows) * len(self.mean))
