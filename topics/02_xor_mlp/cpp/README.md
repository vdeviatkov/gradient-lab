# XOR multilayer perceptron — C++

[Topic and results](../README.md) · [Idea](../idea.md) · [C++ setup and library boundaries](../../../docs/implementations/cpp.md)

## Implementation

Canonical algorithm source:

- [xor_mlp.cpp](xor_mlp.cpp)

Public interfaces:

- [xor_mlp.hpp](include/ml_scratch/xor_mlp.hpp)

The [experiment driver](main.cpp) owns this topic’s data, configuration, evaluation, and reporting.

## Run

From the repository root after the shared C++ build:

```bash
./build/cpp_xor_mlp
```

Dataset requirements and measured results are documented in the [topic README](../README.md).

## Tests

- [logic_models_test.cpp](tests/logic_models_test.cpp)

These tests are registered with CTest by the local [CMakeLists.txt](CMakeLists.txt).
