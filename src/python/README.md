# Python implementations

[Implementations](../README.md) · [Setup](../../docs/getting-started.md) ·
[Python tests](../../tests/python/README.md)

The installable `ml_scratch` package uses the `src` layout and is configured in
[pyproject.toml](../../pyproject.toml). Python 3.11 or newer is required.

| Backend | Status | Guide |
|---|---|---|
| Pure Python | First four topics implemented; no runtime dependencies | [README](ml_scratch/pure/README.md) |
| NumPy | Planned | [README](ml_scratch/numpy_impl/README.md) |
| PyTorch | Planned | [README](ml_scratch/pytorch_impl/README.md) |
| JAX | Planned | [README](ml_scratch/jax_impl/README.md) |

Models live in backend packages. The existing pure-Python `*_experiment.py` modules implement
installed CLIs; `experiments/*/run.py` files are thin source-tree entry points. Keep new training
algorithms out of those wrappers.
