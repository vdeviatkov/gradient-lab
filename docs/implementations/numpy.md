# NumPy conventions

[Implementation guides](README.md) · [Topics and status](../../README.md#topics) ·
[Pure-Python reference](pure-python.md)

Install the optional array dependency:

```bash
python -m pip install -e ".[dev,numpy]"
python -m ml_scratch.numpy.perceptron.experiment
python -m ml_scratch.numpy.mlp.experiment
python -m ml_scratch.numpy.linear_regression.experiment
python -m ml_scratch.numpy.logistic_regression.experiment
python -m ml_scratch.numpy.pca_kmeans.experiment
python -m ml_scratch.numpy.decision_tree.experiment
python -m ml_scratch.numpy.backpropagation.experiment
python -m pytest topics/*/numpy/tests
```

Topics 01–07 own their models, experiments, README, exports, and tests under `numpy/`. The explicit
package mapping in [pyproject.toml](../../pyproject.toml) installs them as
`ml_scratch.numpy.<model>`. No source is mirrored into a second package tree. The corresponding
`ml-numpy-*` commands invoke the same experiment modules. Pure-Python imports do not need NumPy.

Use float64 arrays and vectorize numerical work while keeping the algorithm visible: gradients are
derived by hand (batched backpropagation is three matrix products per layer), normal equations use
explicit pivoted elimination, PCA uses cyclic Jacobi rotations, k-means implements seeding,
assignment, centroid updates, and restarts, and decision trees sweep every split with cumulative
class counts and score pruning from visited-node paths. Library solvers may serve as independent
test oracles; they do not replace the taught algorithms.

The supervised models accept the same `(features, target)` pair datasets as the scalar reference.
PCA and k-means accept rectangular matrices, and decision trees take separate feature and integer
label arrays like their scalar reference; single-vector methods keep their scalar API meaning.
Predictions, parameters, projections, and gradients use arrays where suitable. Configuration and
result value types are shared with scalar models; reusing them does not delegate model computation.
Shared [array validation and stable numerics](../../shared/python/ml_scratch/numpy/validation.py)
contain no topic algorithm. Experiment data and report types can be reused from scalar experiments.

Python's local seeded random generators preserve reference initialization and shuffle streams;
NumPy does the arithmetic. BLAS reductions and floating-point association can differ in the last
bits and can change exact ties or stopping epochs on boundary cases. Compare with tolerances,
aligned parameters, equal loss scaling, and recorded seeds. No speedup is claimed by these results.
The XOR CLI defaults to a smaller ten-seed sweep; `--seeds 50` uses the scalar experiment's budget.

CI installs NumPy and runs all backend tests. The tests include scalar comparisons, independent
central differences, pivoting and singularity, stable extreme logits, covariance/eigen residuals,
reconstruction, cluster repair, deterministic restarts, and the seeded held-out experiments.
