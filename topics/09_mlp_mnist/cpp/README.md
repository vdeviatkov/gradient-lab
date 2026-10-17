# MLP on MNIST — C++

[Topic and results](../README.md) · [Idea](../idea.md) · [C++ setup and library boundaries](../../../docs/implementations/cpp.md)

## Implementation

This topic applies existing models; it does not copy or fork their algorithms.

Reuses canonical implementations from [Backpropagation and gradient checking](../../07_backpropagation/cpp/README.md), [Softmax regression on MNIST](../../08_softmax_mnist/cpp/README.md).

The [experiment driver](main.cpp) owns this topic’s data, configuration, evaluation, and reporting.

## Run

From the repository root after the shared C++ build:

```bash
./build/cpp_mlp_mnist
```

Dataset requirements and measured results are documented in the [topic README](../README.md).

## Verification

The driver checks its experiment criteria. Reused algorithms are tested in their owning topics;
shared integration tests are described in the [test guide](../../../tests/README.md).
