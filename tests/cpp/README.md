# C++ tests

[All tests](../README.md) · [C++ implementation](../../src/cpp/README.md)

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

Each `*_test.cpp` is a standalone executable registered with CTest. Focus a test by name with
`ctest --test-dir build -R cpp_linear_regression_test --output-on-failure`.

| Area | Test sources |
|---|---|
| Library boundaries and utility contracts | [Core-only linkage](core_test.cpp), [utilities](utilities_test.cpp) |
| Introductory and classical models | [Logic models](logic_models_test.cpp), [linear](linear_regression_test.cpp), [logistic](logistic_regression_test.cpp), [PCA](pca_test.cpp), [k-means](kmeans_test.cpp), [trees](decision_tree_test.cpp) |
| Dense networks and optimization | [Backpropagation](neural_network_test.cpp), [softmax](softmax_regression_test.cpp), [optimizers](optimizer_test.cpp), [regularization](regularization_test.cpp) |
| Vision and generation | [Convolution](convolution_test.cpp), [residual](residual_test.cpp), [GAN](gan_test.cpp) |
| Sequence modeling | [Text corpus and n-grams](text_corpus_test.cpp), [recurrent cells](rnn_test.cpp), [Transformer](transformer_test.cpp) |
| Reinforcement learning | [Snake environment](snake_test.cpp), [agents and replay](reinforcement_test.cpp) |
| Measurement and metadata | [Benchmark harness](benchmark_test.cpp), [project metadata](smoke_test.cpp) |

The core-only test links `ml_scratch_cpp` without utilities and exercises training paths across the
models. Other tests link `ml_scratch_utils` to cover evaluation, diagnostics, readers, and checkpoint
round trips. No external test framework or dataset download is required.
