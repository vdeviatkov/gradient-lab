# Design

[Documentation](../README.md) · [Mathematics](../../README.md#topics)

| Guide | Responsibility |
|---|---|
| [Repository layout](repository-layout.md) | Ownership of source, experiments, documentation, tests, and artifacts |
| [C++ boundaries](cpp_layout.md) | Core/support libraries, source ownership, and utility APIs |
| [Reproducibility](reproducibility.md) | Seeds, generators, shared conditions, and cross-language comparisons |

Prefer explicit algorithms and small interfaces. Share supporting operations when they are truly
common; keep experimental choices out of reusable model implementations.
