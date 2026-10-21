"""PCA using explicit sample covariance and vectorized cyclic Jacobi rotations."""

import math
from dataclasses import dataclass

import torch

from ml_scratch.pytorch.validation import matrix, positive, positive_integer, vector, zeros


@dataclass(frozen=True)
class SymmetricEigenDecomposition:
    eigenvalues: torch.Tensor
    eigenvectors: torch.Tensor  # One eigenvector per row, descending eigenvalue.


def covariance_matrix(data):
    x = matrix(data)
    if len(x) < 2:
        raise ValueError("covariance requires at least two samples")
    centered = x - x.mean(dim=0)
    covariance = centered.T @ centered / (len(x) - 1)
    if not torch.isfinite(covariance).all():
        raise ArithmeticError("covariance overflowed; rescale the features")
    return covariance


def jacobi_eigen_decomposition(values, *, max_sweeps=100, tolerance=1e-12):
    a = matrix(values)
    size = len(a)
    if a.shape != (size, size):
        raise ValueError("matrix must be square")
    positive_integer(max_sweeps, "max_sweeps")
    positive(tolerance, "tolerance")
    if torch.any(
        torch.abs(a - a.T) > 1e-9 * torch.clamp_min(torch.maximum(torch.abs(a), torch.abs(a.T)), 1)
    ):
        raise ValueError("matrix must be symmetric")
    a = 0.5 * a + 0.5 * a.T
    rotations = torch.eye(size, dtype=torch.float64)
    threshold = tolerance * max(1.0, float(torch.abs(torch.diag(a)).max()))
    pair_threshold = threshold / max(1.0, math.sqrt(size * (size - 1)))

    def norm():
        off = a[~torch.eye(size, dtype=torch.bool)]
        result = math.hypot(*off.tolist())
        if not math.isfinite(result):
            raise ArithmeticError("eigendecomposition overflowed; rescale the matrix")
        return result

    for _ in range(max_sweeps):
        if norm() <= threshold:
            break
        for j in range(size):
            for k in range(j + 1, size):
                if abs(a[j, k]) <= pair_threshold:
                    continue
                theta = float((a[k, k] - a[j, j]) / (2 * a[j, k]))
                t = (1 if theta >= 0 else -1) / (abs(theta) + math.hypot(theta, 1))
                c = 1 / math.sqrt(1 + t * t)
                s = t * c
                first, second = a[:, j].clone(), a[:, k].clone()
                a[:, j], a[:, k] = c * first - s * second, s * first + c * second
                first, second = a[j].clone(), a[k].clone()
                a[j], a[k] = c * first - s * second, s * first + c * second
                first, second = rotations[:, j].clone(), rotations[:, k].clone()
                rotations[:, j], rotations[:, k] = c * first - s * second, s * first + c * second
    if norm() > threshold:
        raise RuntimeError("Jacobi eigensolver did not converge within max_sweeps")
    if not torch.isfinite(a).all():
        raise ArithmeticError("eigendecomposition overflowed; rescale the matrix")
    order = torch.argsort(-torch.diag(a), stable=True)
    directions = rotations[:, order].T.clone()
    dominant = torch.argmax(torch.abs(directions), dim=1)
    signs = torch.where(directions[torch.arange(size), dominant] < 0, -1, 1)
    return SymmetricEigenDecomposition(torch.diag(a)[order].clone(), directions * signs[:, None])


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

    @torch.no_grad()
    def fit(self, data):
        x = matrix(data)
        if self.component_count > x.shape[1]:
            raise ValueError("component_count must not exceed the feature count")
        decomposition = jacobi_eigen_decomposition(covariance_matrix(x))
        variances = torch.clamp_min(decomposition.eigenvalues, 0)
        total = float(variances.sum())
        self.mean = x.mean(dim=0)
        self.components = decomposition.eigenvectors[: self.component_count].clone()
        self.explained_variance = variances[: self.component_count].clone()
        self.total_variance = total
        self.explained_variance_ratio = (
            self.explained_variance / total if total else zeros(self.component_count)
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
        return float(torch.mean((x - reconstructed) ** 2))
