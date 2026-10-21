# Backend conventions

[Topic/status matrix](../../README.md#topics) · [Setup](../getting-started.md)

Implemented backends have shared tooling guides:

- [C++20](cpp.md): compilation, includes, core/support linking, and source ownership.
- [Pure Python](pure-python.md): package mapping, model imports, and experiment CLIs.
- [NumPy](numpy.md): vectorized models, explicit algorithms, and reference comparisons.

## Additional backends

The root matrix records implementation status. These sections define scope without creating empty
backend folders or separate placeholder documents.

### NumPy

Topics 01–05 are implemented. See [NumPy conventions](numpy.md) for commands and numerical contracts.

### PyTorch

Use tensors and compare hand-derived gradients with automatic differentiation.

### JAX

Explore functional transformations, automatic differentiation, vectorization, and JIT.

### Triton

Teach GPU tiling, memory movement, and fusion where a standalone kernel adds educational value.

## Adding a backend

Put a new implementation, README, and tests under `topics/<topic>/<backend>/`. Link its existing
`idea.md` rather than copying mathematics or mirroring source into another package. Match the
topic's behavior contract; document loss scaling, initialization, stopping rules, and numerical
differences before comparing performance. Update the root status matrix when the milestone lands.
