# C++ implementation

[Implementations](../README.md) · [Setup](../../docs/getting-started.md) ·
[All experiments](../../experiments/README.md) · [Tests](../../tests/cpp/README.md)

All 19 topics are implemented in C++20 using only the standard library. Gradients, matrix operations,
solvers, and learning rules are written explicitly.

## Build and run

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
ctest --test-dir build --output-on-failure
./build/cpp_linear_regression
```

Commands assume the repository root. Dataset requirements are listed in the
[experiment catalog](../../experiments/README.md).

## Layout and linking

| Path / target | Responsibility |
|---|---|
| `include/ml_scratch/` | Public model interfaces and shared types |
| `src/` / `ml_scratch_cpp` | Training, inference, and required shared operations |
| `utils/` / `ml_scratch_utils` | Reference solvers, evaluation, checkpoints, I/O, diagnostics, reporting |
| `internal/` | Private declarations shared across implementation units |
| [Experiment drivers](../../experiments/README.md) | Dataset/configuration choices and result reporting |

Link `ml_scratch_cpp` for core training and inference. Link `ml_scratch_utils` when using experiment
support; it brings in the core transitively. The core does not depend on utilities.
Reference regression fitting and numerical gradient checks are free utilities; some existing
checkpoint and evaluation member APIs are implemented in the support library. See the
[boundary and API guide](../../docs/design/cpp_layout.md) for details and examples.

## Topic map

| Topic | Core source | Experiment |
|---|---|---|
| Perceptron and logic gates | [perceptron.cpp](src/perceptron.cpp) | [README](../../experiments/01_perceptron_logic_gates/README.md) |
| XOR multilayer perceptron | [xor_mlp.cpp](src/xor_mlp.cpp) | [README](../../experiments/02_xor_mlp/README.md) |
| Linear regression | [linear_regression.cpp](src/linear_regression.cpp) | [README](../../experiments/03_linear_regression/README.md) |
| Binary logistic regression | [logistic_regression.cpp](src/logistic_regression.cpp) | [README](../../experiments/04_logistic_regression/README.md) |
| PCA and k-means | [pca.cpp](src/pca.cpp), [kmeans.cpp](src/kmeans.cpp) | [README](../../experiments/05_pca_kmeans/README.md) |
| Decision trees | [decision_tree.cpp](src/decision_tree.cpp) | [README](../../experiments/06_decision_tree/README.md) |
| Backpropagation and gradient checking | [neural_network.cpp](src/neural_network.cpp) | [README](../../experiments/07_backpropagation/README.md) |
| Softmax regression on MNIST | [softmax_regression.cpp](src/softmax_regression.cpp) | [README](../../experiments/08_softmax_mnist/README.md) |
| MLP on MNIST | [neural_network.cpp](src/neural_network.cpp) | [README](../../experiments/09_mlp_mnist/README.md) |
| Optimizers | [optimizer.cpp](src/optimizer.cpp), [neural_network.cpp](src/neural_network.cpp) | [README](../../experiments/10_optimizers/README.md) |
| Regularization and normalization | [neural_network.cpp](src/neural_network.cpp) | [README](../../experiments/11_regularization/README.md) |
| Convolutional networks | [convolution.cpp](src/convolution.cpp) | [README](../../experiments/12_cnn_mnist/README.md) |
| Residual networks | [convolution.cpp](src/convolution.cpp), [neural_network.cpp](src/neural_network.cpp) | [README](../../experiments/13_residual/README.md) |
| Conditional GAN | [gan.cpp](src/gan.cpp) | [README](../../experiments/14_cgan_mnist/README.md) |
| Character RNN | [rnn.cpp](src/rnn.cpp), [char_ngram.cpp](src/char_ngram.cpp) | [README](../../experiments/15_char_rnn/README.md) |
| LSTM and GRU | [rnn.cpp](src/rnn.cpp) | [README](../../experiments/16_lstm_gru/README.md) |
| Transformer | [transformer.cpp](src/transformer.cpp) | [README](../../experiments/17_transformer/README.md) |
| Snake reinforcement learning | [reinforcement.cpp](src/reinforcement.cpp), [snake.cpp](src/snake.cpp) | [README](../../experiments/18_snake_rl/README.md) |
| Implementation benchmarks | [Benchmark utility](utils/benchmark.cpp) | [README](../../experiments/19_benchmarks/README.md) |

Shared data validation lives in [dataset.cpp](src/dataset.cpp), and shared gradient clipping in
[gradient.cpp](src/gradient.cpp). Public declarations mirror source module names under
[`include/ml_scratch/`](include/ml_scratch/).
