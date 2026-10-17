# Softmax regression on MNIST — C++

[Topic and results](../README.md) · [Idea](../idea.md) · [C++ setup and library boundaries](../../../docs/implementations/cpp.md)

## Implementation

Canonical algorithm source:

- [softmax_regression.cpp](softmax_regression.cpp)

Public interfaces:

- [softmax_regression.hpp](include/ml_scratch/softmax_regression.hpp)

Topic-specific support:

- [evaluation.cpp](utils/evaluation.cpp)

The [experiment driver](main.cpp) owns this topic’s data, configuration, evaluation, and reporting.

## Run

From the repository root after the shared C++ build:

```bash
./build/cpp_softmax_mnist
```

Dataset requirements and measured results are documented in the [topic README](../README.md).

## Tests

- [softmax_regression_test.cpp](tests/softmax_regression_test.cpp)

These tests are registered with CTest by the local [CMakeLists.txt](CMakeLists.txt).
