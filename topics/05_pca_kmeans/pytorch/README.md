# PyTorch pca kmeans

[Topic and results](../README.md) · [Mathematics](../idea.md) ·
[PyTorch conventions](../../../docs/implementations/pytorch.md)

[Model](model.py) · [Public API](__init__.py) · [Experiment](experiment.py) ·
[Tests](tests/test_torch_pca_kmeans.py)

```bash
python -m pip install -e ".[dev,pytorch]"
python -m ml_scratch.pytorch.pca_kmeans.experiment
python -m pytest topics/05_pca_kmeans/pytorch/tests
```

All numerical model tensors use CPU float64. Inputs accept the existing pair datasets or matrices;
external tensors are detached and normalized at the boundary. See the shared conventions for
return types, configuration reuse, and numerical-comparison limits.

PCA centers training data and explicitly forms sample covariance with denominator `n - 1`.
Cyclic Jacobi rotations use cloned tensor columns and rows to avoid aliasing during in-place
updates. They return eigenvectors as rows in descending eigenvalue order, with each largest
coordinate nonnegative. Repeated eigenvalues define a subspace rather than a unique basis.
The implementation uses no eigensolver or SVD; test-only `torch.linalg` calls provide independent
oracles. Zero total variance produces zero variance ratios. Reconstruction MSE is per feature.

```python
from ml_scratch.pytorch.pca_kmeans import PrincipalComponentAnalysis, KMeans, KMeansConfig

rows = [[0, 0], [0, 1], [5, 5], [5, 6]]
pca = PrincipalComponentAnalysis(1)
pca.fit(rows)
model = KMeans(2)
result = model.fit(pca.transform_many(rows), KMeansConfig(seed=7))
assert result.converged
```

The separate [k-means source](kmeans.py) implements random-sample and k-means++ initialization,
Lloyd assignments, `index_add_` centroid sums, restarts, and empty-cluster repair. Distance
broadcasting costs O(n k d) temporary memory. Direct differences avoid cancellation near centroids.
Ties choose the lowest centroid index; an empty cluster takes the farthest eligible donor sample.
Duplicate data may produce duplicate centroids and unused prediction labels. Final labels always
refer to nearest final centroids, including after an exhausted budget. Convergence checks the sum
of squared centroid movements, and the first strictly best restart wins. Restart seeds increment
modulo 2**32, matching scalar Python.

These discrete algorithms do not expose autograd gradients. Tests verify eigen residuals,
orthogonality, covariance, reconstruction, scalar agreement, empty clusters, deterministic
restarts, monotone inertia, overflow, invalid data, and explicit non-convergence. The experiment
fits preprocessing on training data, chooses k from the validation elbow without labels, and uses
test purity only for reporting at that fixed k.
