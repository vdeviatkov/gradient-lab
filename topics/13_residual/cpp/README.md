# Residual networks — C++

[Topic and results](../README.md) · [Idea](../idea.md) · [C++ setup and library boundaries](../../../docs/implementations/cpp.md)

## Implementation

This topic applies existing models; it does not copy or fork their algorithms.

Reuses canonical implementations from [Backpropagation and gradient checking](../../07_backpropagation/cpp/README.md), [Convolutional networks](../../12_cnn_mnist/cpp/README.md).

The [experiment driver](main.cpp) owns this topic’s data, configuration, evaluation, and reporting.

## Run

From the repository root after the shared C++ build:

```bash
./build/cpp_resnet_cifar10
```

Dataset requirements and measured results are documented in the [topic README](../README.md).

## Tests

- [residual_test.cpp](tests/residual_test.cpp)

These tests are registered with CTest by the local [CMakeLists.txt](CMakeLists.txt).
