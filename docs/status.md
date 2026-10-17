# Implementation status

[Documentation](README.md) · [Topic catalog](../experiments/README.md) · [Roadmap](roadmap.md)

| Milestone | [Pure Python](../src/python/ml_scratch/pure/README.md) | [NumPy](../src/python/ml_scratch/numpy_impl/README.md) | [PyTorch](../src/python/ml_scratch/pytorch_impl/README.md) | [JAX](../src/python/ml_scratch/jax_impl/README.md) | [C++](../src/cpp/README.md) | [Triton](../src/triton/README.md) |
|---|---:|---:|---:|---:|---:|---:|
| Perceptron: AND / OR | ✅ Complete | Planned | Planned | Planned | ✅ Complete | — |
| Perceptron: XOR limitation | ✅ Demonstrated | Planned | Planned | Planned | ✅ Demonstrated | — |
| Two-layer MLP: XOR | ✅ Complete | Planned | Planned | Planned | ✅ Complete | — |
| Linear regression | ✅ Complete | Planned | Planned | Planned | ✅ Complete | — |
| Binary logistic regression | ✅ Complete | Planned | Planned | Planned | ✅ Complete | — |
| PCA and k-means | Planned | Planned | Planned | Planned | ✅ Complete | — |
| Decision tree classification | Planned | Planned | Planned | Planned | ✅ Complete | — |
| Backpropagation and gradient checking | Planned | Planned | Planned | Planned | ✅ Complete | — |
| Softmax regression: MNIST | Planned | Planned | Planned | Planned | ✅ Complete | Planned |
| MLP: MNIST digit recognition | Planned | Planned | Planned | Planned | ✅ Complete | Planned |
| Optimizer comparisons | Planned | Planned | Planned | Planned | ✅ Complete | Planned |
| Regularization, initialization, and normalization | Planned | Planned | Planned | Planned | ✅ Complete | Planned |
| CNN: MNIST digit recognition | Planned | Planned | Planned | Planned | ✅ Complete | Planned |
| Residual network: image classification | Planned | Planned | Planned | Planned | ✅ Complete | Planned |
| Conditional GAN: MNIST digit generation | Planned | Planned | Planned | Planned | ✅ Complete | Planned |
| RNN: character-level sequence modeling | Planned | Planned | Planned | Planned | ✅ Complete | Planned |
| LSTM and GRU: character-level sequence modeling | Planned | Planned | Planned | Planned | ✅ Complete | Planned |
| Tiny Transformer: character-level language modeling | Planned | Planned | Planned | Planned | ✅ Complete | Planned |
| Snake agent: Q-learning and DQN | Planned | Planned | Planned | Planned | ✅ Complete | Planned |
| Implementation benchmarks | Planned | Planned | Planned | Planned | ✅ Complete | Planned |

`—` means that a standalone Triton kernel would not add meaningful educational value for that
milestone.

The matrix records completed behavior, including the perceptron’s expected XOR limitation.
A complete milestone requires documented mathematics, tests, a runnable experiment, and measured
results. Update this matrix and the root README implementation summary when a milestone lands.
