# Implementation conventions

[Topics and implementation status](../../README.md#topics) · [Setup](../getting-started.md)

These guides describe shared tooling and conventions. Topic-specific code, tests, and instructions
live under `topics/<topic>/<backend>/`; follow the root topic table for a specific implementation.

| Backend | Guide |
|---|---|
| C++20 | [C++](cpp.md) |
| Pure Python | [Python](pure-python.md) |
| NumPy | [NumPy](numpy.md) |
| PyTorch | [PyTorch](pytorch.md) |
| JAX | [JAX](jax.md) |
| Triton | [Triton](triton.md) |

Equivalent implementations must match the educational algorithm and behavior contract. Document
loss scaling, initialization, stopping rules, and numerical differences before reporting speedups.

## Adding a backend

Place a topic’s new implementation under `topics/<topic>/<backend>/` with its README and tests.
Link the existing `idea.md`; do not copy the mathematics or mirror source into another package.
Match the topic’s behavior contract, record numerical differences and measurements, and update
only the root status matrix when the implementation is complete.
