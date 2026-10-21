"""PCA using explicit sample covariance and vectorized cyclic Jacobi rotations."""

import math
from dataclasses import dataclass

import numpy as np
from ml_scratch.numpy.validation import matrix, positive, positive_integer, vector


@dataclass(frozen=True)
class SymmetricEigenDecomposition:
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray  # One eigenvector per row, descending eigenvalue.


def covariance_matrix(data):
    x = matrix(data)
    if len(x) < 2:
        raise ValueError("covariance requires at least two samples")
    with np.errstate(over="ignore", invalid="ignore"):
        centered = x - x.mean(axis=0)
        covariance = centered.T @ centered / (len(x) - 1)
    if not np.isfinite(covariance).all():
        raise ArithmeticError("covariance overflowed; rescale the features")
    return covariance


def jacobi_eigen_decomposition(values, *, max_sweeps=100, tolerance=1e-12):
    a = matrix(values)
    size = len(a)
    if a.shape != (size, size):
        raise ValueError("matrix must be square")
    positive_integer(max_sweeps, "max_sweeps")
    positive(tolerance, "tolerance")
    if np.any(np.abs(a - a.T) > 1e-9 * np.maximum(1, np.maximum(np.abs(a), np.abs(a.T)))):
        raise ValueError("matrix must be symmetric")
    a = 0.5 * a + 0.5 * a.T
    rotations = np.eye(size)
    threshold = tolerance * max(1.0, float(np.abs(np.diag(a)).max()))
    pair_threshold = threshold / max(1.0, math.sqrt(size * (size - 1)))

    def norm():
        off = a[~np.eye(size, dtype=bool)]
        result = float(np.hypot.reduce(off, initial=0.0))
        if not np.isfinite(result):
            raise ArithmeticError("eigendecomposition overflowed; rescale the matrix")
        return result

    for _ in range(max_sweeps):
        if norm() <= threshold:
            break
        for j in range(size):
            for k in range(j + 1, size):
                if abs(a[j, k]) <= pair_threshold:
                    continue
                theta = (a[k, k] - a[j, j]) / (2 * a[j, k])
                t = (1 if theta >= 0 else -1) / (abs(theta) + math.hypot(theta, 1))
                c = 1 / math.sqrt(1 + t * t)
                s = t * c
                first, second = a[:, j].copy(), a[:, k].copy()
                a[:, j], a[:, k] = c * first - s * second, s * first + c * second
                first, second = a[j].copy(), a[k].copy()
                a[j], a[k] = c * first - s * second, s * first + c * second
                first, second = rotations[:, j].copy(), rotations[:, k].copy()
                rotations[:, j], rotations[:, k] = c * first - s * second, s * first + c * second
    if norm() > threshold:
        raise RuntimeError("Jacobi eigensolver did not converge within max_sweeps")
    if not np.isfinite(a).all():
        raise ArithmeticError("eigendecomposition overflowed; rescale the matrix")
    order = np.argsort(-np.diag(a), kind="stable")
    directions = rotations[:, order].T.copy()
    dominant = np.argmax(np.abs(directions), axis=1)
    signs = np.where(directions[np.arange(size), dominant] < 0, -1, 1)
    return SymmetricEigenDecomposition(np.diag(a)[order].copy(), directions * signs[:, None])


class PrincipalComponentAnalysis:
    def __init__(self, component_count):
        positive_integer(component_count, "component_count")
        self.component_count = component_count
        self.mean = self.components = self.explained_variance = self.explained_variance_ratio = None
        self.total_variance = 0.0

    @property
    def fitted(self):
        return self.mean is not None

    def _require_fitted(self):
        if not self.fitted:
            raise ValueError("model must be fitted first")

    def fit(self, data):
        x = matrix(data)
        if self.component_count > x.shape[1]:
            raise ValueError("component_count must not exceed the feature count")
        decomposition = jacobi_eigen_decomposition(covariance_matrix(x))
        variances = np.maximum(0, decomposition.eigenvalues)
        total = float(variances.sum())
        self.mean = x.mean(axis=0)
        self.components = decomposition.eigenvectors[: self.component_count].copy()
        self.explained_variance = variances[: self.component_count].copy()
        self.total_variance = total
        self.explained_variance_ratio = (
            self.explained_variance / total if total else np.zeros(self.component_count)
        )

    def transform(self, sample):
        self._require_fitted()
        return (vector(sample, len(self.mean)) - self.mean) @ self.components.T

    def transform_many(self, data):
        self._require_fitted()
        return (matrix(data, len(self.mean)) - self.mean) @ self.components.T

    def inverse_transform(self, projection):
        self._require_fitted()
        return vector(projection, self.component_count) @ self.components + self.mean

    def reconstruction_error(self, data):
        self._require_fitted()
        x = matrix(data, len(self.mean))
        reconstructed = ((x - self.mean) @ self.components.T) @ self.components + self.mean
        return float(np.mean((x - reconstructed) ** 2))
