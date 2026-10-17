# Convolutional networks — C++

[Topic and results](../README.md) · [Idea](../idea.md) · [C++ setup and library boundaries](../../../docs/implementations/cpp.md)

## Implementation

Canonical algorithm source:

- [convolution.cpp](convolution.cpp)

Public interfaces:

- [convolution.hpp](include/ml_scratch/convolution.hpp)

Topic-specific support:

- [convolution_checkpoint.cpp](utils/convolution_checkpoint.cpp)
- [evaluation.cpp](utils/evaluation.cpp)

Reuses canonical implementations from [Backpropagation and gradient checking](../../07_backpropagation/cpp/README.md), [Optimizers](../../10_optimizers/cpp/README.md).

The [experiment driver](main.cpp) owns this topic’s data, configuration, evaluation, and reporting.

## Run

From the repository root after the shared C++ build:

```bash
./build/cpp_cnn_mnist
```

Dataset requirements and measured results are documented in the [topic README](../README.md).

## Tests

- [convolution_test.cpp](tests/convolution_test.cpp)

These tests are registered with CTest by the local [CMakeLists.txt](CMakeLists.txt).
