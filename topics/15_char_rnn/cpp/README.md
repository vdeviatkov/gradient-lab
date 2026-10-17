# Character RNN — C++

[Topic and results](../README.md) · [Idea](../idea.md) · [C++ setup and library boundaries](../../../docs/implementations/cpp.md)

## Implementation

Canonical algorithm source:

- [char_ngram.cpp](char_ngram.cpp)
- [rnn.cpp](rnn.cpp)

Public interfaces:

- [char_ngram.hpp](include/ml_scratch/char_ngram.hpp)

- [rnn.hpp](include/ml_scratch/rnn.hpp)

Topic-specific support:

- [reporting.cpp](utils/reporting.cpp)
- [rnn_checkpoint.cpp](utils/rnn_checkpoint.cpp)
- [rnn_diagnostics.cpp](utils/rnn_diagnostics.cpp)

Reuses canonical implementations from [Optimizers](../../10_optimizers/cpp/README.md).

The [experiment driver](main.cpp) owns this topic’s data, configuration, evaluation, and reporting.

## Run

From the repository root after the shared C++ build:

```bash
./build/cpp_char_rnn
```

Dataset requirements and measured results are documented in the [topic README](../README.md).

## Tests

- [rnn_test.cpp](tests/rnn_test.cpp)

These tests are registered with CTest by the local [CMakeLists.txt](CMakeLists.txt).
