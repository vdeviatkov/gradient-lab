# C++ conventions

[Implementation guides](README.md) · [Topics](../../README.md#topics)

Requires CMake 3.20+ and a C++20 compiler; the implementations use only the standard library.
Build and run from the repository root:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

Each `topics/<topic>/cpp/` contains its implementation README, experiment `main.cpp`, optional
model and public interfaces, utilities, tests, and local CMake registration. When a topic applies
an existing model, its README links that model's owner instead of introducing another copy.
Header includes remain `ml_scratch/<module>.hpp`; CMake collects topic-owned include directories.

## Library boundaries

The aggregate `ml_scratch_cpp` target holds training/inference and necessary shared operations.
`ml_scratch_utils` holds reference solvers, diagnostics, evaluation, readers, and checkpoints and
links the core transitively. Core algorithms cannot depend on support implementations. Link only
the core when no utilities are needed. Topic executables retain their `build/cpp_*` paths.

Shared operations live in [shared/cpp](../../shared/cpp). Gradient clipping is common to RNN,
Transformer, and DQN; model algorithms remain topic-owned. Use [testing](../testing.md) for quality checks.

The count-based baseline now has its own topic-15 header, `ml_scratch/char_ngram.hpp`;
include it explicitly when using `CharNgram`. `text_corpus.hpp` contains only corpus infrastructure.

Training-dependent evaluation stays core: XOR convergence checks accuracy and loss, tree pruning
compares validation accuracy, and Transformer training computes windowed validation loss.
The [core-only test](../../tests/cpp/core_test.cpp) enforces this runtime dependency boundary.
