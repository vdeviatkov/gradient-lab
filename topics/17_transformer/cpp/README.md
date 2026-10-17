# Transformer — C++

[Topic and results](../README.md) · [Idea](../idea.md) · [C++ setup and library boundaries](../../../docs/implementations/cpp.md)

## Implementation

Canonical algorithm source:

- [transformer.cpp](transformer.cpp)

Public interfaces:

- [transformer.hpp](include/ml_scratch/transformer.hpp)

Topic-specific support:

- [transformer_checkpoint.cpp](utils/transformer_checkpoint.cpp)
- [transformer_diagnostics.cpp](utils/transformer_diagnostics.cpp)

Reuses canonical implementations from [Optimizers](../../10_optimizers/cpp/README.md).

The [experiment driver](main.cpp) owns this topic’s data, configuration, evaluation, and reporting.

## Run

From the repository root after the shared C++ build:

```bash
./build/cpp_transformer
```

Dataset requirements and measured results are documented in the [topic README](../README.md).

## Tests

- [transformer_test.cpp](tests/transformer_test.cpp)

These tests are registered with CTest by the local [CMakeLists.txt](CMakeLists.txt).
