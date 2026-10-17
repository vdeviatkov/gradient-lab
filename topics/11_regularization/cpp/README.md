# Regularization and normalization — C++

[Topic and results](../README.md) · [Idea](../idea.md) · [C++ setup and library boundaries](../../../docs/implementations/cpp.md)

## Implementation

This topic applies existing models; it does not copy or fork their algorithms.

Reuses canonical implementations from [Backpropagation and gradient checking](../../07_backpropagation/cpp/README.md), [Optimizers](../../10_optimizers/cpp/README.md).

The [experiment driver](main.cpp) owns this topic’s data, configuration, evaluation, and reporting.

## Run

From the repository root after the shared C++ build:

```bash
./build/cpp_regularization
```

Dataset requirements and measured results are documented in the [topic README](../README.md).

## Tests

- [regularization_test.cpp](tests/regularization_test.cpp)

These tests are registered with CTest by the local [CMakeLists.txt](CMakeLists.txt).
