# ML from Scratch

Machine learning algorithms built from first principles, with explicit mathematics, deterministic
experiments, and automated tests. Pure Python and C++ are implemented; NumPy, PyTorch, JAX, and
Triton are planned. Frameworks will not replace the algorithm being taught.

[Documentation](docs/README.md) · [Getting started](docs/getting-started.md) ·
[Full status](docs/status.md) · [Contributing](CONTRIBUTING.md) · [MIT license](LICENSE)

## Quick start

Python 3.11+; the pure-Python models have no runtime dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
python -m ml_scratch.pure.logistic_regression_experiment
```

CMake 3.20+ and a C++20 compiler:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
./build/cpp_logistic_regression
```

## Topics

Each topic links to its experiment README and available implementation source. Backend column
headings link to implementation guides. **Planned** means no implementation yet; **—** means a
standalone implementation is not applicable.

| # | Topic | [C++](src/cpp/README.md) | [Python](src/python/ml_scratch/pure/README.md) | [NumPy](src/python/ml_scratch/numpy_impl/README.md) | [PyTorch](src/python/ml_scratch/pytorch_impl/README.md) | [JAX](src/python/ml_scratch/jax_impl/README.md) | [Triton](src/triton/README.md) |
|---:|---|---|---|---|---|---|---|
| 01 | [Perceptron and logic gates](experiments/01_perceptron_logic_gates/README.md) | [Source](src/cpp/src/perceptron.cpp) | [Source](src/python/ml_scratch/pure/perceptron.py) | Planned | Planned | Planned | — |
| 02 | [XOR multilayer perceptron](experiments/02_xor_mlp/README.md) | [Source](src/cpp/src/xor_mlp.cpp) | [Source](src/python/ml_scratch/pure/mlp.py) | Planned | Planned | Planned | — |
| 03 | [Linear regression](experiments/03_linear_regression/README.md) | [Model](src/cpp/src/linear_regression.cpp) · [Reference](src/cpp/utils/regression.cpp) | [Source](src/python/ml_scratch/pure/linear_regression.py) | Planned | Planned | Planned | — |
| 04 | [Binary logistic regression](experiments/04_logistic_regression/README.md) | [Source](src/cpp/src/logistic_regression.cpp) | [Source](src/python/ml_scratch/pure/logistic_regression.py) | Planned | Planned | Planned | — |
| 05 | [PCA and k-means](experiments/05_pca_kmeans/README.md) | [PCA](src/cpp/src/pca.cpp) · [k-means](src/cpp/src/kmeans.cpp) | Planned | Planned | Planned | Planned | — |
| 06 | [Decision trees](experiments/06_decision_tree/README.md) | [Source](src/cpp/src/decision_tree.cpp) | Planned | Planned | Planned | Planned | — |
| 07 | [Backpropagation and gradient checking](experiments/07_backpropagation/README.md) | [Network](src/cpp/src/neural_network.cpp) · [Checks](src/cpp/utils/gradient_check.cpp) | Planned | Planned | Planned | Planned | — |
| 08 | [Softmax regression on MNIST](experiments/08_softmax_mnist/README.md) | [Source](src/cpp/src/softmax_regression.cpp) | Planned | Planned | Planned | Planned | Planned |
| 09 | [MLP on MNIST](experiments/09_mlp_mnist/README.md) | [Source](src/cpp/src/neural_network.cpp) | Planned | Planned | Planned | Planned | Planned |
| 10 | [Optimizers](experiments/10_optimizers/README.md) | [Updates](src/cpp/src/optimizer.cpp) · [Training](src/cpp/src/neural_network.cpp) | Planned | Planned | Planned | Planned | Planned |
| 11 | [Regularization and normalization](experiments/11_regularization/README.md) | [Source](src/cpp/src/neural_network.cpp) | Planned | Planned | Planned | Planned | Planned |
| 12 | [Convolutional networks](experiments/12_cnn_mnist/README.md) | [Source](src/cpp/src/convolution.cpp) | Planned | Planned | Planned | Planned | Planned |
| 13 | [Residual networks](experiments/13_residual/README.md) | [CNN](src/cpp/src/convolution.cpp) · [MLP](src/cpp/src/neural_network.cpp) | Planned | Planned | Planned | Planned | Planned |
| 14 | [Conditional GAN](experiments/14_cgan_mnist/README.md) | [Source](src/cpp/src/gan.cpp) | Planned | Planned | Planned | Planned | Planned |
| 15 | [Character RNN](experiments/15_char_rnn/README.md) | [RNN](src/cpp/src/rnn.cpp) · [n-gram](src/cpp/src/char_ngram.cpp) | Planned | Planned | Planned | Planned | Planned |
| 16 | [LSTM and GRU](experiments/16_lstm_gru/README.md) | [Source](src/cpp/src/rnn.cpp) | Planned | Planned | Planned | Planned | Planned |
| 17 | [Transformer](experiments/17_transformer/README.md) | [Source](src/cpp/src/transformer.cpp) | Planned | Planned | Planned | Planned | Planned |
| 18 | [Snake reinforcement learning](experiments/18_snake_rl/README.md) | [Agents](src/cpp/src/reinforcement.cpp) · [Game](src/cpp/src/snake.cpp) | Planned | Planned | Planned | Planned | Planned |
| 19 | [Implementation benchmarks](experiments/19_benchmarks/README.md) | [Source](src/cpp/utils/benchmark.cpp) | Planned | Planned | Planned | Planned | Planned |

## Development

Run `pytest`, `ruff check .`, `python scripts/check_docs.py`, and
`ctest --test-dir build --output-on-failure`. See [testing](docs/testing.md) for setup and coverage.

[Repository layout](docs/design/repository-layout.md) · [Mathematics](docs/mathematics/README.md) ·
[Experiment findings](docs/findings.md) · [Roadmap](docs/roadmap.md) · [Results policy](results/README.md)
