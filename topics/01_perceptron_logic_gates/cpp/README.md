# Perceptron and logic gates — C++

[Topic and results](../README.md) · [Idea](../idea.md) · [C++ setup and library boundaries](../../../docs/implementations/cpp.md)

## Implementation

Canonical algorithm source:

- [perceptron.cpp](perceptron.cpp)

Public interfaces:

- [perceptron.hpp](include/ml_scratch/perceptron.hpp)

Topic-specific support:

- [evaluation.cpp](utils/evaluation.cpp)
- [logic_gates.cpp](utils/logic_gates.cpp)

The [experiment driver](main.cpp) owns this topic’s data, configuration, evaluation, and reporting.

## Run

From the repository root after the shared C++ build:

```bash
./build/cpp_perceptron_logic_gates
```

Dataset requirements and measured results are documented in the [topic README](../README.md).

## Verification

The driver checks its experiment criteria. Reused algorithms are tested in their owning topics;
shared integration tests are described in the [test guide](../../../tests/README.md).
