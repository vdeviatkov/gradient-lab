"""Dependency-free PCA and k-means, with one canonical implementation per algorithm."""

from .kmeans import (
    KMeans,
    KMeansConfig,
    KMeansInitialization,
    KMeansRestartSummary,
    KMeansTrainingResult,
    cluster_purity,
)
from .model import (
    PrincipalComponentAnalysis,
    SymmetricEigenDecomposition,
    covariance_matrix,
    jacobi_eigen_decomposition,
)

__all__ = [
    "KMeans",
    "KMeansConfig",
    "KMeansInitialization",
    "KMeansRestartSummary",
    "KMeansTrainingResult",
    "PrincipalComponentAnalysis",
    "SymmetricEigenDecomposition",
    "cluster_purity",
    "covariance_matrix",
    "jacobi_eigen_decomposition",
]
