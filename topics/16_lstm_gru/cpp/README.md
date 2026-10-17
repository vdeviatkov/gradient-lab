# LSTM and GRU — C++

[Topic and results](../README.md) · [Idea](../idea.md) · [C++ setup and library boundaries](../../../docs/implementations/cpp.md)

## Implementation

This topic applies existing models; it does not copy or fork their algorithms.

Reuses canonical implementations from [Character RNN](../../15_char_rnn/cpp/README.md).

The [experiment driver](main.cpp) owns this topic’s data, configuration, evaluation, and reporting.

## Run

From the repository root after the shared C++ build:

```bash
./build/cpp_lstm_gru
```

Dataset requirements and measured results are documented in the [topic README](../README.md).

## Verification

The driver checks its experiment criteria. Reused algorithms are tested in their owning topics;
shared integration tests are described in the [test guide](../../../tests/README.md).
