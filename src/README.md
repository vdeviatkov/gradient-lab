# Implementations

[Project home](../README.md) · [Full implementation status](../docs/status.md)

| Implementation | Responsibility | Guide |
|---|---|---|
| Pure Python | Dependency-free, explicit scalar operations | [README](python/ml_scratch/pure/README.md) |
| C++20 | Explicit data structures and compiled algorithms | [README](cpp/README.md) |
| NumPy | Planned vectorized equivalents | [README](python/ml_scratch/numpy_impl/README.md) |
| PyTorch | Planned tensors and autograd comparisons | [README](python/ml_scratch/pytorch_impl/README.md) |
| JAX | Planned functional transformations and JIT comparisons | [README](python/ml_scratch/jax_impl/README.md) |
| Triton | Planned GPU kernels where educationally useful | [README](triton/README.md) |

Each implementation follows the same topic progression. Equivalent models should share datasets,
objectives, stopping rules, and evaluation criteria; document numerical or convention differences.
A matching seed alone does not establish identical initialization across languages. See
[reproducibility](../docs/design/reproducibility.md).
