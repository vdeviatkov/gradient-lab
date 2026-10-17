# Shared C++ integration tests

[All tests](../README.md) · [C++ conventions](../../docs/implementations/cpp.md)

Model-specific tests live in each topic's `cpp/tests/` directory and are registered by its local
CMakeLists.txt. This folder owns integration and common-infrastructure checks:

- [numerics_test.cpp](numerics_test.cpp): stable primitives and activation derivatives.
- [core_test.cpp](core_test.cpp): training/inference across model families, linking core alone.
- [utilities_test.cpp](utilities_test.cpp): reference solver and diagnostic contracts.
- [text_corpus_test.cpp](text_corpus_test.cpp): shared corpus and n-gram integration.
- [smoke_test.cpp](smoke_test.cpp): project metadata.

Run all topic and shared tests with `ctest --test-dir build --output-on-failure` after building.
No external test framework or dataset download is required.
