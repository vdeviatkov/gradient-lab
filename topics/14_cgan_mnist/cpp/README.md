# Conditional GAN — C++

[Topic and results](../README.md) · [Idea](../idea.md) · [C++ setup and library boundaries](../../../docs/implementations/cpp.md)

## Implementation

Canonical algorithm source:

- [gan.cpp](gan.cpp)

Public interfaces:

- [gan.hpp](include/ml_scratch/gan.hpp)

Topic-specific support:

- [gan_checkpoint.cpp](utils/gan_checkpoint.cpp)
- [reporting.cpp](utils/reporting.cpp)

Reuses canonical implementations from [Backpropagation and gradient checking](../../07_backpropagation/cpp/README.md), [Optimizers](../../10_optimizers/cpp/README.md).

The [experiment driver](main.cpp) owns this topic’s data, configuration, evaluation, and reporting.

## Run

From the repository root after the shared C++ build:

```bash
./build/cpp_cgan_mnist
```

Dataset requirements and measured results are documented in the [topic README](../README.md).

## Tests

- [gan_test.cpp](tests/gan_test.cpp)

These tests are registered with CTest by the local [CMakeLists.txt](CMakeLists.txt).
