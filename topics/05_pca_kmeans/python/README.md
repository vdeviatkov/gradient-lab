# PCA and k-means — pure Python

[Topic and results](../README.md) · [Idea and mathematics](../idea.md) ·
[Python setup](../../../docs/implementations/pure-python.md)

- [model.py](model.py): unbiased covariance, cyclic Jacobi eigendecomposition, PCA projection and reconstruction.
- [kmeans.py](kmeans.py): Lloyd updates, random/k-means++ seeding, restart selection, and purity evaluation.
- [validation.py](validation.py): shared input contracts for these two algorithms.
- [experiment.py](experiment.py): fixed synthetic splits and validation-only cluster-count selection.

After installing the repository, run `python -m ml_scratch.pure.pca_kmeans.experiment` or
`ml-pca-kmeans`. The default seed is `20260907`; `--seed` overrides and prints it.
No NumPy, linear-algebra library, or downloaded dataset is needed.

Python uses `transform`/`predict` for one sample and `transform_many`/`predict_many` for a matrix.
PCA exposes immutable component/statistic snapshots. K-means returns all restart summaries,
including an inertia history and explicit convergence state. Fits validate inputs before replacing
fitted state. Jacobi raises if its iteration budget cannot meet the residual tolerance.

## Tests

- [PCA and eigensolver](tests/test_pca.py)
- [k-means and evaluation](tests/test_kmeans.py)
- [Experiment and dataset agreement](tests/test_experiment.py)

Run `python -m pytest topics/05_pca_kmeans/python/tests` from the repository root.

## Comparability and numerical safety

The algorithms, unbiased covariance, descending eigenvalues, deterministic eigenvector signs,
mean updates, squared-movement stopping threshold, and restart selection follow C++. The dataset
stream matches C++; the learner RNG intentionally remains Python's local `random.Random`.

The Python Jacobi solver verifies its final off-diagonal Frobenius residual and raises on an
exhausted budget. Its per-pair skip threshold accounts for the matrix dimension, preventing many
small off-diagonal entries from jointly exceeding the global tolerance. C++'s standalone solver
currently returns its intermediate estimate on budget exhaustion. Both default runs converge on
the experiment. Python uses `hypot` for the rotation parameter and rejects arithmetic overflow.
Approximately symmetric inputs within the symmetry tolerance use their symmetric average.

Labels returned after fitting are recomputed against the final centroids. Equidistant centroids
choose the lower index. Duplicate points can leave fewer nonempty predicted groups than requested
centroids; this is reported as ordinary clustering behavior, not hidden by inventing groups.
