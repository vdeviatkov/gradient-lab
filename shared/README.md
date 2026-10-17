# Shared infrastructure

[Repository layout](../docs/architecture.md) · [Topics](../README.md#topics)

Only code used across topics belongs here. Algorithms stay in their owning topic, even when a
later topic reuses them.

| Location | Responsibility |
|---|---|
| `cpp/include/ml_scratch/` | Shared data types, numerical/activation primitives, and infrastructure interfaces |
| `cpp/src/` | Data validation, common distance/statistics operations, gradient clipping |
| `cpp/utils/` | Dataset readers, deterministic data generation, metrics, corpus handling, metadata |
| `python/ml_scratch/` | Base package, shared fixtures and stable numerical primitives |

The root CMake configuration adds these sources to the common core/support libraries. Each topic
adds its own sources and tests through its local CMakeLists.txt. Python's package-directory map in
[pyproject.toml](../pyproject.toml) installs topic-local packages without copying their source.
