# ML from Scratch

Machine learning from first principles, with explicit mathematics, reproducible experiments,
and automated tests. Each topic owns its idea, implementation, tests, and measured report.

[Setup](docs/getting-started.md) · [Documentation](docs/README.md) ·
[Implementation guides](docs/implementations/README.md) · [Contributing](CONTRIBUTING.md) · [MIT license](LICENSE)

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
python -m ml_scratch.pure.logistic_regression.experiment
```

For C++: `cmake -S . -B build -DCMAKE_BUILD_TYPE=Release`, then
`cmake --build build --parallel` and `./build/cpp_logistic_regression`.
See [setup](docs/getting-started.md) for prerequisites and datasets.

## Topics

Topic READMEs contain the experiment and results; **Idea** links explain the algorithm.
Implementation links open the topic's source/test guide. **Planned** means unimplemented;
**—** means a standalone implementation is not applicable. This is the canonical status matrix.

| # | Topic | Idea | [C++](docs/implementations/cpp.md) | [Python](docs/implementations/pure-python.md) | [NumPy](docs/implementations/README.md#numpy) | [PyTorch](docs/implementations/README.md#pytorch) | [JAX](docs/implementations/README.md#jax) | [Triton](docs/implementations/README.md#triton) |
|---:|---|---|---|---|---|---|---|---|
| 01 | [Perceptron and logic gates](topics/01_perceptron_logic_gates/README.md) | [Idea](topics/01_perceptron_logic_gates/idea.md) | [Code](topics/01_perceptron_logic_gates/cpp/README.md) | [Code](topics/01_perceptron_logic_gates/python/README.md) | [Code](topics/01_perceptron_logic_gates/numpy/README.md) | Planned | Planned | — |
| 02 | [XOR multilayer perceptron](topics/02_xor_mlp/README.md) | [Idea](topics/02_xor_mlp/idea.md) | [Code](topics/02_xor_mlp/cpp/README.md) | [Code](topics/02_xor_mlp/python/README.md) | [Code](topics/02_xor_mlp/numpy/README.md) | Planned | Planned | — |
| 03 | [Linear regression](topics/03_linear_regression/README.md) | [Idea](topics/03_linear_regression/idea.md) | [Code](topics/03_linear_regression/cpp/README.md) | [Code](topics/03_linear_regression/python/README.md) | [Code](topics/03_linear_regression/numpy/README.md) | Planned | Planned | — |
| 04 | [Binary logistic regression](topics/04_logistic_regression/README.md) | [Idea](topics/04_logistic_regression/idea.md) | [Code](topics/04_logistic_regression/cpp/README.md) | [Code](topics/04_logistic_regression/python/README.md) | [Code](topics/04_logistic_regression/numpy/README.md) | Planned | Planned | — |
| 05 | [PCA and k-means](topics/05_pca_kmeans/README.md) | [Idea](topics/05_pca_kmeans/idea.md) | [Code](topics/05_pca_kmeans/cpp/README.md) | [Code](topics/05_pca_kmeans/python/README.md) | [Code](topics/05_pca_kmeans/numpy/README.md) | Planned | Planned | — |
| 06 | [Decision trees](topics/06_decision_tree/README.md) | [Idea](topics/06_decision_tree/idea.md) | [Code](topics/06_decision_tree/cpp/README.md) | [Code](topics/06_decision_tree/python/README.md) | Planned | Planned | Planned | — |
| 07 | [Backpropagation and gradient checking](topics/07_backpropagation/README.md) | [Idea](topics/07_backpropagation/idea.md) | [Code](topics/07_backpropagation/cpp/README.md) | Planned | Planned | Planned | Planned | — |
| 08 | [Softmax regression on MNIST](topics/08_softmax_mnist/README.md) | [Idea](topics/08_softmax_mnist/idea.md) | [Code](topics/08_softmax_mnist/cpp/README.md) | Planned | Planned | Planned | Planned | Planned |
| 09 | [MLP on MNIST](topics/09_mlp_mnist/README.md) | [Idea](topics/09_mlp_mnist/idea.md) | [Code](topics/09_mlp_mnist/cpp/README.md) | Planned | Planned | Planned | Planned | Planned |
| 10 | [Optimizers](topics/10_optimizers/README.md) | [Idea](topics/10_optimizers/idea.md) | [Code](topics/10_optimizers/cpp/README.md) | Planned | Planned | Planned | Planned | Planned |
| 11 | [Regularization and normalization](topics/11_regularization/README.md) | [Idea](topics/11_regularization/idea.md) | [Code](topics/11_regularization/cpp/README.md) | Planned | Planned | Planned | Planned | Planned |
| 12 | [Convolutional networks](topics/12_cnn_mnist/README.md) | [Idea](topics/12_cnn_mnist/idea.md) | [Code](topics/12_cnn_mnist/cpp/README.md) | Planned | Planned | Planned | Planned | Planned |
| 13 | [Residual networks](topics/13_residual/README.md) | [Idea](topics/13_residual/idea.md) | [Code](topics/13_residual/cpp/README.md) | Planned | Planned | Planned | Planned | Planned |
| 14 | [Conditional GAN](topics/14_cgan_mnist/README.md) | [Idea](topics/14_cgan_mnist/idea.md) | [Code](topics/14_cgan_mnist/cpp/README.md) | Planned | Planned | Planned | Planned | Planned |
| 15 | [Character RNN](topics/15_char_rnn/README.md) | [Idea](topics/15_char_rnn/idea.md) | [Code](topics/15_char_rnn/cpp/README.md) | Planned | Planned | Planned | Planned | Planned |
| 16 | [LSTM and GRU](topics/16_lstm_gru/README.md) | [Idea](topics/16_lstm_gru/idea.md) | [Code](topics/16_lstm_gru/cpp/README.md) | Planned | Planned | Planned | Planned | Planned |
| 17 | [Transformer](topics/17_transformer/README.md) | [Idea](topics/17_transformer/idea.md) | [Code](topics/17_transformer/cpp/README.md) | Planned | Planned | Planned | Planned | Planned |
| 18 | [Snake reinforcement learning](topics/18_snake_rl/README.md) | [Idea](topics/18_snake_rl/idea.md) | [Code](topics/18_snake_rl/cpp/README.md) | Planned | Planned | Planned | Planned | Planned |
| 19 | [Implementation benchmarks](topics/19_benchmarks/README.md) | [Idea](topics/19_benchmarks/idea.md) | [Code](topics/19_benchmarks/cpp/README.md) | Planned | Planned | Planned | Planned | Planned |

## Repository layout

Each topic owns its README, idea, and backend code/tests. Shared infrastructure lives in `shared/`.
See [ownership rules](docs/architecture.md) and [quality checks](docs/testing.md).
