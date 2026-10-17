# Binary logistic regression — C++

[Topic and results](../README.md) · [Idea](../idea.md) · [C++ setup and library boundaries](../../../docs/implementations/cpp.md)

## Implementation

Canonical algorithm source:

- [logistic_regression.cpp](logistic_regression.cpp)

Public interfaces:

- [logistic_regression.hpp](include/ml_scratch/logistic_regression.hpp)

Topic-specific support:

- [evaluation.cpp](utils/evaluation.cpp)

The [experiment driver](main.cpp) owns this topic’s data, configuration, evaluation, and reporting.

## Run

From the repository root after the shared C++ build:

```bash
./build/cpp_logistic_regression
```

Dataset requirements and measured results are documented in the [topic README](../README.md).

## Tests

- [logistic_regression_test.cpp](tests/logistic_regression_test.cpp)

These tests are registered with CTest by the local [CMakeLists.txt](CMakeLists.txt).
