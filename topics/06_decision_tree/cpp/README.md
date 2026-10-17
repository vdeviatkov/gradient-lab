# Decision trees — C++

[Topic and results](../README.md) · [Idea](../idea.md) · [C++ setup and library boundaries](../../../docs/implementations/cpp.md)

## Implementation

Canonical algorithm source:

- [decision_tree.cpp](decision_tree.cpp)

Public interfaces:

- [decision_tree.hpp](include/ml_scratch/decision_tree.hpp)

Topic-specific support:

- [tree_diagnostics.cpp](utils/tree_diagnostics.cpp)

The [experiment driver](main.cpp) owns this topic’s data, configuration, evaluation, and reporting.

## Run

From the repository root after the shared C++ build:

```bash
./build/cpp_decision_tree
```

Dataset requirements and measured results are documented in the [topic README](../README.md).

## Tests

- [decision_tree_test.cpp](tests/decision_tree_test.cpp)

These tests are registered with CTest by the local [CMakeLists.txt](CMakeLists.txt).
