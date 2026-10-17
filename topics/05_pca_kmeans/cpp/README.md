# PCA and k-means — C++

[Topic and results](../README.md) · [Idea](../idea.md) · [C++ setup and library boundaries](../../../docs/implementations/cpp.md)

## Implementation

Canonical algorithm source:

- [kmeans.cpp](kmeans.cpp)
- [pca.cpp](pca.cpp)

Public interfaces:

- [kmeans.hpp](include/ml_scratch/kmeans.hpp)
- [pca.hpp](include/ml_scratch/pca.hpp)

Topic-specific support:

- [evaluation.cpp](utils/evaluation.cpp)

The [experiment driver](main.cpp) owns this topic’s data, configuration, evaluation, and reporting.

## Run

From the repository root after the shared C++ build:

```bash
./build/cpp_pca_kmeans
```

Dataset requirements and measured results are documented in the [topic README](../README.md).

## Tests

- [kmeans_test.cpp](tests/kmeans_test.cpp)
- [pca_test.cpp](tests/pca_test.cpp)

These tests are registered with CTest by the local [CMakeLists.txt](CMakeLists.txt).
