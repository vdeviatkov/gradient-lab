# NumPy PCA and k-means

[Topic and results](../README.md) · [Mathematics](../idea.md) ·
[NumPy conventions](../../../docs/implementations/numpy.md)

[Model](model.py) · [Public API](__init__.py) · [Experiment](experiment.py) ·
[Tests](tests/test_numpy_pca_kmeans.py)

```bash
python -m pip install -e ".[dev,numpy]"
python -m ml_scratch.numpy.pca_kmeans.experiment
python -m pytest topics/05_pca_kmeans/numpy/tests
```

PCA explicitly centers training data and builds sample covariance `X_centered.T @ X_centered /
(n - 1)`. Cyclic Jacobi rotations diagonalize it with vectorized row and column operations;
no `eigh`, SVD, or PCA estimator replaces the algorithm. Eigenvectors are rows in descending
variance order, with the largest-magnitude coordinate made nonnegative. Ties use stable ordering.
Repeated eigenvalues still define a subspace rather than a unique basis. Zero total variance gives
zero variance ratios. Reconstruction MSE is normalized by samples times original feature count.

K-means uses broadcast squared distances, lowest-index nearest-centroid ties, and `add.at`
centroid sums. Initialization implements random-sample selection or k-means++ explicitly using
the scalar model's local Python random stream. Each restart uses `(seed + restart) mod 2**32`;
the first strictly best inertia wins. Empty clusters take the farthest eligible sample from a
donor with at least two samples. Duplicate data can still yield duplicate centroids and unused
prediction labels. Convergence tests the sum of squared centroid movements against tolerance.
Returned labels always assign to the final nearest centroid, including when the budget expires.

```python
from ml_scratch.numpy.pca_kmeans import PrincipalComponentAnalysis, KMeans, KMeansConfig

rows = [[0, 0], [0, 1], [5, 5], [5, 6]]
pca = PrincipalComponentAnalysis(1)
pca.fit(rows)
model = KMeans(2)
result = model.fit(pca.transform_many(rows), KMeansConfig(seed=7))
assert result.converged
```

The experiment shares the scalar seeded dataset, fits PCA on training data, selects k with the
validation inertia elbow, and reports test purity only for evaluation. Purity is not a criterion
for selecting k: it can increase merely by making more clusters. Distance broadcasting uses
O(n k d) temporary storage, appropriate for these small teaching datasets rather than large data.

The separate [k-means source](kmeans.py) owns seeding, Lloyd iterations, restarts, and purity.

Tests compare against the pure-Python reference and check independent numerical or hand-computed
results, input validation, deterministic behavior, and relevant failure/stopping cases.
