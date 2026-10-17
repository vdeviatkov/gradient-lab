# Experiments

[Project home](../README.md) · [Setup](../docs/getting-started.md) ·
[Status by implementation](../docs/status.md) · [Mathematics](../docs/mathematics/README.md)

Experiments own dataset splits, configuration, threshold/model selection, reporting, and verification
criteria. They call reusable implementations in `src/`; they do not duplicate model algorithms.
Run the commands from the repository root after following the setup guide.

| # | Topic README | C++ executable | External data |
|---:|---|---|---|
| 01 | [Perceptron and logic gates](01_perceptron_logic_gates/README.md) | `cpp_perceptron_logic_gates` | None |
| 02 | [XOR multilayer perceptron](02_xor_mlp/README.md) | `cpp_xor_mlp` | None |
| 03 | [Linear regression](03_linear_regression/README.md) | `cpp_linear_regression` | None |
| 04 | [Binary logistic regression](04_logistic_regression/README.md) | `cpp_logistic_regression` | None |
| 05 | [PCA and k-means](05_pca_kmeans/README.md) | `cpp_pca_kmeans` | None |
| 06 | [Decision trees](06_decision_tree/README.md) | `cpp_decision_tree` | None |
| 07 | [Backpropagation and gradient checking](07_backpropagation/README.md) | `cpp_backpropagation` | None |
| 08 | [Softmax regression on MNIST](08_softmax_mnist/README.md) | `cpp_softmax_mnist` | MNIST |
| 09 | [MLP on MNIST](09_mlp_mnist/README.md) | `cpp_mlp_mnist` | MNIST |
| 10 | [Optimizers](10_optimizers/README.md) | `cpp_optimizers` | None |
| 11 | [Regularization and normalization](11_regularization/README.md) | `cpp_regularization` | None |
| 12 | [Convolutional networks](12_cnn_mnist/README.md) | `cpp_cnn_mnist` | MNIST |
| 13 | [Residual networks](13_residual/README.md) | `cpp_resnet_cifar10` | CIFAR-10; MNIST baseline |
| 14 | [Conditional GAN](14_cgan_mnist/README.md) | `cpp_cgan_mnist` | MNIST |
| 15 | [Character RNN](15_char_rnn/README.md) | `cpp_char_rnn` | Tiny Shakespeare |
| 16 | [LSTM and GRU](16_lstm_gru/README.md) | `cpp_lstm_gru` | Tiny Shakespeare |
| 17 | [Transformer](17_transformer/README.md) | `cpp_transformer` | Tiny Shakespeare |
| 18 | [Snake reinforcement learning](18_snake_rl/README.md) | `cpp_snake_rl` | None |
| 19 | [Implementation benchmarks](19_benchmarks/README.md) | `cpp_benchmarks` | None |

Python runners are currently available for topics 01–04. See the
[pure-Python README](../src/python/ml_scratch/pure/README.md) for module and installed commands.
Topics 13 and 14 include longer training runs; consult their READMEs before running them.
