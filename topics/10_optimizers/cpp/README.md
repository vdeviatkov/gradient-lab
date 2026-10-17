# Optimizers — C++

[Topic and results](../README.md) · [Idea](../idea.md) · [C++ setup and library boundaries](../../../docs/implementations/cpp.md)

## Implementation

Canonical algorithm source:

- [optimizer.cpp](optimizer.cpp)

Public interfaces:

- [optimizer.hpp](include/ml_scratch/optimizer.hpp)

Topic-specific support:

- [reporting.cpp](utils/reporting.cpp)

Reuses canonical implementations from [Backpropagation and gradient checking](../../07_backpropagation/cpp/README.md).

The [experiment driver](main.cpp) owns this topic’s data, configuration, evaluation, and reporting.

## Run

From the repository root after the shared C++ build:

```bash
./build/cpp_optimizers
```

Dataset requirements and measured results are documented in the [topic README](../README.md).

## Tests

- [optimizer_test.cpp](tests/optimizer_test.cpp)

These tests are registered with CTest by the local [CMakeLists.txt](CMakeLists.txt).
