# Backpropagation and gradient checking — C++

[Topic and results](../README.md) · [Idea](../idea.md) · [C++ setup and library boundaries](../../../docs/implementations/cpp.md)

## Implementation

Canonical algorithm source:

- [neural_network.cpp](neural_network.cpp)

Public interfaces:

- [neural_network.hpp](include/ml_scratch/neural_network.hpp)
- [gradient_check.hpp](include/ml_scratch/utils/gradient_check.hpp)

Topic-specific support:

- [evaluation.cpp](utils/evaluation.cpp)
- [gradient_check.cpp](utils/gradient_check.cpp)
- [neural_network_checkpoint.cpp](utils/neural_network_checkpoint.cpp)

The [experiment driver](main.cpp) owns this topic’s data, configuration, evaluation, and reporting.

## Run

From the repository root after the shared C++ build:

```bash
./build/cpp_backpropagation
```

Dataset requirements and measured results are documented in the [topic README](../README.md).

## Tests

- [neural_network_test.cpp](tests/neural_network_test.cpp)

These tests are registered with CTest by the local [CMakeLists.txt](CMakeLists.txt).
