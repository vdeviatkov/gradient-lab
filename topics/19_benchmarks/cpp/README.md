# Implementation benchmarks — C++

[Topic and results](../README.md) · [Idea](../idea.md) · [C++ setup and library boundaries](../../../docs/implementations/cpp.md)

## Implementation

This topic applies existing models; it does not copy or fork their algorithms.

Topic-specific support:

- [benchmark.cpp](utils/benchmark.cpp)

The [experiment driver](main.cpp) owns this topic’s data, configuration, evaluation, and reporting.

## Run

From the repository root after the shared C++ build:

```bash
./build/cpp_benchmarks
```

Dataset requirements and measured results are documented in the [topic README](../README.md).

## Tests

- [benchmark_test.cpp](tests/benchmark_test.cpp)

These tests are registered with CTest by the local [CMakeLists.txt](CMakeLists.txt).
